//! Agent MCP module.
//!
//! Support for the Model Context Protocol (MCP). This module defines the
//! `McpClient` trait and `McpServerConfig` for connecting to MCP servers
//! over various transports (stdio, SSE, HTTP, WebSocket). It also includes
//! a lifecycle state machine for tracking connection state.
//!
//! The crate defines the transport types and lifecycle states. The binary
//! spec defines handshake timeouts and which servers to launch.

use crate::agent_tools::ToolDefinition;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::process::Stdio;
use tokio::sync::Mutex;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::timeout;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for MCP operations.
#[derive(Debug, Clone)]
pub enum McpError {
    Transport(String),
    Protocol(String),
    NotFound(String),
    Timeout,
    Shutdown,
    Internal(String),
}

impl McpError {
    /// Returns true if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            McpError::Transport(_) | McpError::Protocol(_) | McpError::Timeout
        )
    }
}

impl std::fmt::Display for McpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpError::Transport(msg) => write!(f, "Transport error: {}", msg),
            McpError::Protocol(msg) => write!(f, "Protocol error: {}", msg),
            McpError::NotFound(msg) => write!(f, "Not found: {}", msg),
            McpError::Timeout => write!(f, "Operation timed out"),
            McpError::Shutdown => write!(f, "Client shutdown"),
            McpError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for McpError {}

impl From<std::io::Error> for McpError {
    fn from(e: std::io::Error) -> Self {
        McpError::Transport(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Lifecycle phase — full 11-phase state machine
// ---------------------------------------------------------------------------

/// Lifecycle phase of the MCP client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpLifecyclePhase {
    ConfigLoad,
    ServerRegistration,
    SpawnConnect,
    InitializeHandshake,
    ToolDiscovery,
    ResourceDiscovery,
    Ready,
    Invocation,
    ErrorSurfacing,
    Shutdown,
    Cleanup,
}

impl McpLifecyclePhase {
    /// Returns all 11 phases in order.
    pub fn all() -> [Self; 11] {
        [
            Self::ConfigLoad,
            Self::ServerRegistration,
            Self::SpawnConnect,
            Self::InitializeHandshake,
            Self::ToolDiscovery,
            Self::ResourceDiscovery,
            Self::Ready,
            Self::Invocation,
            Self::ErrorSurfacing,
            Self::Shutdown,
            Self::Cleanup,
        ]
    }
}

impl std::fmt::Display for McpLifecyclePhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::ConfigLoad => "config_load",
            Self::ServerRegistration => "server_registration",
            Self::SpawnConnect => "spawn_connect",
            Self::InitializeHandshake => "initialize_handshake",
            Self::ToolDiscovery => "tool_discovery",
            Self::ResourceDiscovery => "resource_discovery",
            Self::Ready => "ready",
            Self::Invocation => "invocation",
            Self::ErrorSurfacing => "error_surfacing",
            Self::Shutdown => "shutdown",
            Self::Cleanup => "cleanup",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// Transport trait
// ---------------------------------------------------------------------------

/// Trait for pluggable MCP transport implementations.
///
/// Any implementation of this trait can be used with [`McpClientImpl`]
/// to connect to MCP servers over different transport mechanisms
/// (stdio, SSE, HTTP, WebSocket, etc.).
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Establishes a connection to the MCP server.
    async fn connect(&self) -> Result<(), McpError>;

    /// Closes the connection to the MCP server.
    async fn disconnect(&self) -> Result<(), McpError>;

    /// Sends a message to the MCP server.
    async fn send(&self, msg: &[u8]) -> Result<(), McpError>;

    /// Receives a message from the MCP server.
    async fn receive(&self) -> Result<Vec<u8>, McpError>;
}

// ---------------------------------------------------------------------------
// Stdio bootstrap — construction data for spawning a stdio MCP server
// ---------------------------------------------------------------------------

/// Bootstrap configuration for spawning an MCP server over stdio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StdioBootstrap {
    /// Command to run (e.g. `npx`, `uvx`, `/usr/local/bin/mcp-server`).
    pub command: String,
    /// Positional arguments passed to the command.
    pub args: Vec<String>,
    /// Environment variables to set for the child process.
    pub env: BTreeMap<String, String>,
}

impl StdioBootstrap {
    /// Creates a new stdio bootstrap with the given command and arguments.
    pub fn new(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: command.into(),
            args,
            env: BTreeMap::new(),
        }
    }

    /// Adds an environment variable.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Client trait
// ---------------------------------------------------------------------------

/// Trait for MCP client operations.
#[async_trait]
pub trait McpClient: Send + Sync {
    /// Discovers available tools on the MCP server.
    async fn discover_tools(&self) -> Result<Vec<ToolDefinition>, McpError>;

    /// Calls a tool by name with the given JSON input.
    async fn call_tool(&self, name: &str, input: &str) -> Result<String, McpError>;

    /// Lists available resources on the MCP server.
    async fn list_resources(&self) -> Result<Vec<String>, McpError>;

    /// Reads a resource by URI.
    async fn read_resource(&self, uri: &str) -> Result<String, McpError>;

    /// Initiates a graceful shutdown of the MCP server.
    async fn shutdown(&self) -> Result<(), McpError>;
}

// ---------------------------------------------------------------------------
// MCP client implementation
// ---------------------------------------------------------------------------

/// MCP client implementation wrapping a pluggable transport.
pub struct McpClientImpl<T: McpTransport> {
    transport: T,
}

impl<T: McpTransport> McpClientImpl<T> {
    /// Creates a new MCP client with the given transport.
    pub fn new(transport: T) -> Self {
        Self { transport }
    }
}

/// Inner mutable state (stdin/stdout handles and initialization flag).
struct Inner {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    initialized: bool,
    next_request_id: u64,
}

/// Stdio-based MCP client — holds the child process and async state.
/// All mutation is gated through a Mutex for interior mutability.
pub struct McpStdioClient {
    child: Mutex<Child>,
    inner: Mutex<Inner>,
}

/// Protocol constants.
const MCP_INITIALIZE_TIMEOUT_MS: u64 = 10_000;
const MCP_LIST_TOOLS_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_PROTOCOL_VERSION: &str = "2025-03-26";
const DEFAULT_CLIENT_NAME: &str = "ai-agent";

// ---------------------------------------------------------------------------
// JSON-RPC wire types (private inner module)
// ---------------------------------------------------------------------------

mod mcp_stdio_types {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    #[serde(untagged)]
    pub enum JsonRpcId {
        Number(u64),
        String(String),
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JsonRpcRequest<T = JsonValue> {
        pub jsonrpc: String,
        pub id: JsonRpcId,
        pub method: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub params: Option<T>,
    }

    impl<T> JsonRpcRequest<T> {
        pub fn new(id: JsonRpcId, method: impl Into<String>, params: Option<T>) -> Self {
            Self {
                jsonrpc: "2.0".to_string(),
                id,
                method: method.into(),
                params,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JsonRpcError {
        pub code: i64,
        pub message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<JsonValue>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JsonRpcResponse<T = JsonValue> {
        pub jsonrpc: String,
        pub id: JsonRpcId,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<T>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<JsonRpcError>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpInitializeParams {
        pub protocol_version: String,
        pub capabilities: JsonValue,
        pub client_info: McpInitializeClientInfo,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpInitializeResult {
        pub protocol_version: String,
        pub capabilities: JsonValue,
        pub server_info: McpInitializeServerInfo,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpInitializeServerInfo {
        pub name: String,
        pub version: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpInitializeClientInfo {
        pub name: String,
        pub version: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpListToolsParams {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cursor: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpTool {
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        #[serde(rename = "inputSchema", skip_serializing_if = "Option::is_none")]
        pub input_schema: Option<JsonValue>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpListToolsResult {
        pub tools: Vec<McpTool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub next_cursor: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpToolCallParams {
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<JsonValue>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpToolCallResult {
        #[serde(default)]
        pub content: Vec<McpToolCallContent>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum McpToolCallContent {
        #[serde(rename = "text")]
        Text { text: String },
        #[serde(rename = "resource")]
        Resource { resource: McpResourceContents },
        #[serde(rename = "image")]
        Image { data: String, mime_type: Option<String> },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpListResourcesParams {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cursor: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct McpResource {
        pub uri: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpListResourcesResult {
        pub resources: Vec<McpResource>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub next_cursor: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpReadResourceParams {
        pub uri: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct McpResourceContents {
        pub uri: String,
        #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub text: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub blob: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct McpReadResourceResult {
        pub contents: Vec<McpResourceContents>,
    }
}

// ---------------------------------------------------------------------------
// McpStdioClient — stdio process management
// ---------------------------------------------------------------------------

impl McpStdioClient {
    /// Spawns a new MCP stdio client from the given bootstrap config.
    pub async fn spawn(bootstrap: &StdioBootstrap) -> Result<Self, McpError> {
        let mut command = Command::new(&bootstrap.command);
        command
            .args(&bootstrap.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        for (key, value) in &bootstrap.env {
            command.env(key, value);
        }

        let mut child = command
            .spawn()
            .map_err(|e| McpError::Transport(format!("failed to spawn {}: {e}", bootstrap.command)))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("child process missing stdin pipe".into()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("child process missing stdout pipe".into()))?;

        Ok(Self {
            child: Mutex::new(child),
            inner: Mutex::new(Inner {
                stdin,
                stdout: BufReader::new(stdout),
                initialized: false,
                next_request_id: 1,
            }),
        })
    }

    /// Sends the JSON-RPC `initialize` handshake once (idempotent).
    pub async fn ensure_initialized(&self) -> Result<(), McpError> {
        let initialized = {
            let mut inner = self.inner.lock().await;
            if inner.initialized {
                return Ok(());
            }
            inner.initialized = true;
            inner.next_request_id += 1;
            inner.next_request_id - 1
        };

        let id = mcp_stdio_types::JsonRpcId::Number(initialized);
        let params = mcp_stdio_types::McpInitializeParams {
            protocol_version: DEFAULT_PROTOCOL_VERSION.to_string(),
            capabilities: JsonValue::Object(serde_json::Map::new()),
            client_info: mcp_stdio_types::McpInitializeClientInfo {
                name: DEFAULT_CLIENT_NAME.to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        };

        self.request_jsonrpc(id, "initialize", Some(params)).await
    }

    /// Sends a JSON-RPC request and reads the response, with a timeout.
    async fn request_jsonrpc<TParams: Serialize>(
        &self,
        id: mcp_stdio_types::JsonRpcId,
        method: &str,
        params: Option<TParams>,
    ) -> Result<(), McpError> {
        let request = mcp_stdio_types::JsonRpcRequest::new(id.clone(), method.to_string(), params);

        let bytes = serde_json::to_vec(&request)
            .map_err(|e| McpError::Protocol(format!("JSON serialization failed: {e}")))?;

        {
            let mut inner = self.inner.lock().await;
            inner.stdin.write_all(
                format!("Content-Length: {}\r\n\r\n", bytes.len()).as_bytes(),
            ).await
            .map_err(|e| McpError::Transport(format!("write header failed: {e}")))?;
            inner.stdin.write_all(&bytes).await
            .map_err(|e| McpError::Transport(format!("write payload failed: {e}")))?;
            inner.stdin.flush().await
            .map_err(|e| McpError::Transport(format!("flush failed: {e}")))?;
        }

        let response_bytes = {
            let mut inner = self.inner.lock().await;
            let mut content_length: Option<usize> = None;
            loop {
                let mut line = String::new();
                let n = inner.stdout.read_line(&mut line).await
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, e))?;
                if n == 0 {
                    return Err(McpError::Transport("MCP stdio stream closed".into()));
                }
                if line == "\r\n" {
                    break;
                }
                let header = line.trim_end_matches(['\r', '\n']);
                if let Some((name, value)) = header.split_once(':') {
                    if name.trim().eq_ignore_ascii_case("Content-Length") {
                        content_length = Some(
                            value.trim().parse::<usize>().map_err(|e|
                                McpError::Protocol(format!("invalid Content-Length: {e}"))
                            )?
                        );
                    }
                }
            }
            let len = content_length.ok_or_else(|| McpError::Protocol("missing Content-Length".into()))?;
            let mut payload = vec![0_u8; len];
            inner.stdout.read_exact(&mut payload).await
                .map_err(|e| McpError::Transport(format!("read payload failed: {e}")))?;
            payload
        };

        let response: mcp_stdio_types::JsonRpcResponse<JsonValue> =
            serde_json::from_slice(&response_bytes)
                .map_err(|e| McpError::Protocol(format!("JSON deserialization failed: {e}")))?;

        if response.error.is_some() {
            return Err(McpError::Protocol("server returned JSON-RPC error".into()));
        }

        Ok(())
    }

    /// Sends `tools/list` and returns all tools (paginated).
    async fn do_tools_list(&self) -> Result<Vec<mcp_stdio_types::McpTool>, McpError> {
        self.ensure_initialized().await?;

        let mut all_tools = Vec::new();
        let mut cursor = None;

        loop {
            let (id, params) = {
                let mut inner = self.inner.lock().await;
                let id = inner.next_request_id;
                inner.next_request_id += 1;
                let params = mcp_stdio_types::McpListToolsParams { cursor };
                (mcp_stdio_types::JsonRpcId::Number(id), params)
            };

            let response: mcp_stdio_types::JsonRpcResponse<mcp_stdio_types::McpListToolsResult> =
                timeout(
                    Duration::from_millis(MCP_LIST_TOOLS_TIMEOUT_MS),
                    self.jsonrpc_request(id, "tools/list", Some(params)),
                )
                .await
                .map_err(|_| McpError::Timeout)?
                .map_err(|e| {
                    if matches!(e, McpError::Transport(_)) {
                        McpError::Transport(format!("list_tools: {e}"))
                    } else {
                        e
                    }
                })?;

            if let Some(err) = response.error {
                return Err(McpError::Protocol(format!(
                    "tools/list error: {} (code {})",
                    err.message, err.code
                )));
            }

            let result = response.result
                .ok_or_else(|| McpError::Protocol("tools/list returned no result".into()))?;

            all_tools.extend(result.tools);

            match result.next_cursor {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }

        Ok(all_tools)
    }

    /// Sends `tools/call` and returns the result serialized as JSON.
    async fn do_tool_call(
        &self,
        name: &str,
        arguments: Option<JsonValue>,
    ) -> Result<String, McpError> {
        self.ensure_initialized().await?;

        let (id, params) = {
            let mut inner = self.inner.lock().await;
            let id = inner.next_request_id;
            inner.next_request_id += 1;
            (
                mcp_stdio_types::JsonRpcId::Number(id),
                mcp_stdio_types::McpToolCallParams {
                    name: name.to_string(),
                    arguments,
                },
            )
        };

        let response: mcp_stdio_types::JsonRpcResponse<mcp_stdio_types::McpToolCallResult> =
            timeout(
                Duration::from_millis(MCP_LIST_TOOLS_TIMEOUT_MS),
                self.jsonrpc_request(id, "tools/call", Some(params)),
            )
            .await
            .map_err(|_| McpError::Timeout)?
            .map_err(|e| {
                if matches!(e, McpError::Transport(_)) {
                    McpError::Transport(format!("call_tool: {e}"))
                } else {
                    e
                }
            })?;

        if let Some(err) = response.error {
            return Err(McpError::Protocol(format!(
                "tools/call error: {} (code {})",
                err.message, err.code
            )));
        }

        let result = response.result
            .ok_or_else(|| McpError::Protocol("tools/call returned no result".into()))?;

        serde_json::to_string(&result)
            .map_err(|e| McpError::Internal(format!("failed to serialize call result: {e}")))
    }

    /// Sends `resources/list` and returns all resource URIs (paginated).
    async fn do_list_resources(&self) -> Result<Vec<String>, McpError> {
        self.ensure_initialized().await?;

        let mut uris = Vec::new();
        let mut cursor = None;

        loop {
            let (id, params) = {
                let mut inner = self.inner.lock().await;
                let id = inner.next_request_id;
                inner.next_request_id += 1;
                let params = mcp_stdio_types::McpListResourcesParams { cursor };
                (mcp_stdio_types::JsonRpcId::Number(id), params)
            };

            let response:
                mcp_stdio_types::JsonRpcResponse<mcp_stdio_types::McpListResourcesResult> =
                timeout(
                    Duration::from_millis(MCP_LIST_TOOLS_TIMEOUT_MS),
                    self.jsonrpc_request(id, "resources/list", Some(params)),
                )
                .await
                .map_err(|_| McpError::Timeout)?
                .map_err(|e| {
                    if matches!(e, McpError::Transport(_)) {
                        McpError::Transport(format!("list_resources: {e}"))
                    } else {
                        e
                    }
                })?;

            if let Some(err) = response.error {
                return Err(McpError::Protocol(format!(
                    "resources/list error: {} (code {})",
                    err.message, err.code
                )));
            }

            let result = response.result
                .ok_or_else(|| McpError::Protocol("resources/list returned no result".into()))?;

            for res in result.resources {
                uris.push(res.uri);
            }

            match result.next_cursor {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }

        Ok(uris)
    }

    /// Sends `resources/read` and returns the resource content as a string.
    async fn do_read_resource(&self, uri: &str) -> Result<String, McpError> {
        self.ensure_initialized().await?;

        let (id, params) = {
            let mut inner = self.inner.lock().await;
            let id = inner.next_request_id;
            inner.next_request_id += 1;
            (
                mcp_stdio_types::JsonRpcId::Number(id),
                mcp_stdio_types::McpReadResourceParams {
                    uri: uri.to_string(),
                },
            )
        };

        let response:
            mcp_stdio_types::JsonRpcResponse<mcp_stdio_types::McpReadResourceResult> =
            timeout(
                Duration::from_millis(MCP_LIST_TOOLS_TIMEOUT_MS),
                self.jsonrpc_request(id, "resources/read", Some(params)),
            )
            .await
            .map_err(|_| McpError::Timeout)?
            .map_err(|e| {
                if matches!(e, McpError::Transport(_)) {
                    McpError::Transport(format!("read_resource: {e}"))
                } else {
                    e
                }
            })?;

        if let Some(err) = response.error {
            return Err(McpError::Protocol(format!(
                "resources/read error: {} (code {})",
                err.message, err.code
            )));
        }

        let result = response.result
            .ok_or_else(|| McpError::Protocol("resources/read returned no result".into()))?;

        let parts: Vec<String> = result
            .contents
            .iter()
            .map(|c| c.text.as_deref().unwrap_or("").to_string())
            .collect();

        Ok(parts.join("\n"))
    }

    /// Gracefully terminates the child process.
    pub async fn shutdown(&self) -> Result<(), McpError> {
        let mut child = self.child.lock().await;
        if child.try_wait().map(|s| s.is_none()).unwrap_or(false) {
            match child.kill().await {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => {}
                Err(e) => return Err(McpError::Transport(format!("kill failed: {e}"))),
            }
        }
        let _ = child.wait().await;
        Ok(())
    }

    /// Low-level JSON-RPC request: sends a request and returns the deserialized response.
    async fn jsonrpc_request<TParams: Serialize, TResult: for<'de> Deserialize<'de>>(
        &self,
        id: mcp_stdio_types::JsonRpcId,
        method: &str,
        params: Option<TParams>,
    ) -> Result<mcp_stdio_types::JsonRpcResponse<TResult>, McpError> {
        let request = mcp_stdio_types::JsonRpcRequest::new(id, method.to_string(), params);

        let bytes = serde_json::to_vec(&request)
            .map_err(|e| McpError::Protocol(format!("JSON serialization failed: {e}")))?;

        // Send
        {
            let mut inner = self.inner.lock().await;
            inner.stdin.write_all(
                format!("Content-Length: {}\r\n\r\n", bytes.len()).as_bytes(),
            ).await
            .map_err(|e| McpError::Transport(format!("write header failed: {e}")))?;
            inner.stdin.write_all(&bytes).await
            .map_err(|e| McpError::Transport(format!("write payload failed: {e}")))?;
            inner.stdin.flush().await
            .map_err(|e| McpError::Transport(format!("flush failed: {e}")))?;
        }

        // Receive
        let response_bytes = {
            let mut inner = self.inner.lock().await;
            let mut content_length: Option<usize> = None;
            loop {
                let mut line = String::new();
                let n = inner.stdout.read_line(&mut line).await
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, e))?;
                if n == 0 {
                    return Err(McpError::Transport("MCP stdio stream closed".into()));
                }
                if line == "\r\n" {
                    break;
                }
                let header = line.trim_end_matches(['\r', '\n']);
                if let Some((name, value)) = header.split_once(':') {
                    if name.trim().eq_ignore_ascii_case("Content-Length") {
                        content_length = Some(
                            value.trim().parse::<usize>().map_err(|e|
                                McpError::Protocol(format!("invalid Content-Length: {e}"))
                            )?
                        );
                    }
                }
            }
            let len = content_length
                .ok_or_else(|| McpError::Protocol("missing Content-Length".into()))?;
            let mut payload = vec![0_u8; len];
            inner.stdout.read_exact(&mut payload).await
                .map_err(|e| McpError::Transport(format!("read payload failed: {e}")))?;
            payload
        };

        serde_json::from_slice(&response_bytes)
            .map_err(|e| McpError::Protocol(format!("JSON deserialization failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// McpStdioClient — McpTransport and McpClient trait implementations
// ---------------------------------------------------------------------------

#[async_trait]
impl McpTransport for McpStdioClient {
    async fn connect(&self) -> Result<(), McpError> {
        Ok(())
    }

    async fn disconnect(&self) -> Result<(), McpError> {
        self.shutdown().await
    }

    async fn send(&self, _msg: &[u8]) -> Result<(), McpError> {
        Err(McpError::Internal("send not implemented for stdio".into()))
    }

    async fn receive(&self) -> Result<Vec<u8>, McpError> {
        Err(McpError::Internal("receive not implemented for stdio".into()))
    }
}

#[async_trait]
impl McpClient for McpStdioClient {
    async fn discover_tools(&self) -> Result<Vec<ToolDefinition>, McpError> {
        let tools = self.do_tools_list().await?;

        Ok(tools
            .into_iter()
            .map(|tool| {
                let input_schema = tool
                    .input_schema
                    .as_ref()
                    .map(|s| serde_json::to_string(s).unwrap_or_default())
                    .unwrap_or_default();

                ToolDefinition {
                    name: tool.name,
                    description: tool.description.unwrap_or_default(),
                    input_schema,
                }
            })
            .collect())
    }

    async fn call_tool(&self, name: &str, input: &str) -> Result<String, McpError> {
        let arguments: Option<JsonValue> = if input.is_empty() {
            None
        } else {
            Some(
                serde_json::from_str(input)
                    .map_err(|e| McpError::Protocol(format!("invalid tool input JSON: {e}")))?,
            )
        };

        self.do_tool_call(name, arguments).await
    }

    async fn list_resources(&self) -> Result<Vec<String>, McpError> {
        self.do_list_resources().await
    }

    async fn read_resource(&self, uri: &str) -> Result<String, McpError> {
        self.do_read_resource(uri).await
    }

    async fn shutdown(&self) -> Result<(), McpError> {
        self.shutdown().await
    }
}