//! Canonical error taxonomy for the `ai-agent` crate.
//!
//! Every error in this crate is domain-specific — no generic `Box<Error>` or
//! `"unknown error"` strings. Binary authors handle errors by matching on
//! structured variants and reading the enclosed context.
//!
//! ## Error hierarchy
//!
//! The hierarchy flows from low-level subsystem errors up to the top-level
//! [`AgentError`]:
//!
//! - **`ConfigError`** — configuration loading, parsing, and validation
//! - **`McpError`** — Model Context Protocol transport and protocol errors
//! - **`PluginError`** — plugin lifecycle (init, shutdown, hooks)
//! - **`SessionError`** — session persistence and conversation state
//! - **`ToolError`** — tool registry, execution, and permissions
//! - **`AgentError`** — top-level orchestrator; wraps all of the above
//!
//! ## Retryability
//!
//! Every error type in this module implements `is_retryable() -> bool`.
//! Binary authors use this to decide whether to retry an operation:
//!
//! | Error type       | Retryable variants                                 | Non-retryable variants                      |
//! |------------------|----------------------------------------------------|----------------------------------------------|
//! | `ConfigError`    | `IO`, `Internal`                                   | `Parse`, `FormatNotSupported`, `NotFound`, `Validation` |
//! | `AgentError`     | `Session`, `Internal`                              | `Context`, `Permission`, `Tool`, `Config`    |
//! | `McpError`       | `Transport`, `Protocol`, `Timeout`                 | `NotFound`, `Shutdown`, `Internal`            |
//! | `PluginError`    | `Initialization`, `Hook`, `Config`                | `Shutdown`, `Internal`                        |
//! | `SessionError`   | `SaveFailed`, `LoadFailed`, `ForkFailed`, `SearchFailed` | `NotFound`, `Internal`             |
//! | `ToolError`      | `Timeout`, `Internal`                              | `Execution`, `NotFound`, `PermissionDenied`  |
//!
//! Transient errors (network blips, rate limits, timeouts) are retryable.
//! Permanent errors (missing credentials, permission denied, not found) are not.
//!
//! ## Handling strategy
//!
//! Binary authors should typically:
//! 1. Propagate `AgentError` from the agent loop upward.
//! 2. Use `is_retryable()` to decide between retry and abort.
//! 3. Surface non-retryable errors to the user with the variant message.
//! 4. Log all errors (including retryable ones) for observability.

#![allow(unused_imports)]

// ============================================================================
// ConfigError
// ============================================================================

/// Errors emitted by the configuration subsystem ([`agent_config`][crate::agent_config]).
///
/// Triggered during config file discovery, parsing, format resolution,
/// and validation. Binary authors encounter this error when initializing
/// the agent with [`ConfigLoaderRegistry::load`][crate::agent_config::ConfigLoaderRegistry::load].
///
/// | Variant             | Trigger                                                | Retryable? |
/// |---------------------|--------------------------------------------------------|------------|
/// | `Parse`             | JSON/TOML/YAML syntax error                             | No         |
/// | `FormatNotSupported`| No loader registered for the file extension            | No         |
/// | `NotFound`          | Config file or required key absent                     | No         |
/// | `IO`                | Disk read failure (permissions, ENOMEM, device error)   | Yes        |
/// | `Validation`        | Schema check failed (e.g., unknown field)               | No         |
/// | `Internal`          | Loader registry empty or invariant violated            | Yes        |
///
/// **Binary author action:** Show a clear error to the user pointing them
/// to the config file path. Retry is unlikely to help unless the underlying
/// I/O condition clears.
pub use crate::agent_config::ConfigError;

// ============================================================================
// AgentError
// ============================================================================

/// Top-level error emitted by the agent core ([`agent_core`][crate::agent_core]).
///
/// Covers every subsystem that can fail during agent loop execution:
/// session, permissions, tools, config, and internal invariants.
/// Binary authors receive this error from [`AgentContext::run_turn`][crate::agent_core::AgentContext::run_turn]
/// and the [`Agent::run`][crate::agent_core::Agent::run] method.
///
/// | Variant     | Trigger                                                     | Retryable? |
/// |-------------|-------------------------------------------------------------|------------|
/// | `Context`   | Session not found, invalid session state, or bad input     | No         |
/// | `Session`   | Session save/load/fork failed (wrapped [`SessionError`]) | Yes |
/// | `Permission`| Tool call denied by the permission policy                    | No         |
/// | `Tool`      | Tool execution failed (wrapped [`ToolError`]) | No |
/// | `Config`    | Configuration invalid or unavailable (wrapped [`ConfigError`]) | No |
/// | `Internal`  | Invariant violated or unexpected panic in crate internals     | Yes        |
///
/// **Binary author action:** Propagate to the top level. Distinguish
/// retryable vs. non-retryable via `is_retryable()`. Present permission and
/// tool errors to the user with the embedded context. Log internal errors
/// and consider retrying once.
pub use crate::agent_core::AgentError;

// ============================================================================
// McpError
// ============================================================================

/// Errors emitted by the MCP subsystem ([`agent_mcp`][crate::agent_mcp]).
///
/// Triggered when connecting to, communicating with, or shutting down
/// a Model Context Protocol server. Binary authors encounter this error
/// when the MCP client discovers tools, calls a tool, or manages server
/// lifecycle.
///
/// | Variant    | Trigger                                                        | Retryable? |
/// |------------|----------------------------------------------------------------|------------|
/// | `Transport`| Stdio spawn failed, connection dropped, socket error            | Yes        |
/// | `Protocol` | Malformed request/response, invalid JSON-RPC frame              | Yes        |
/// | `NotFound` | Requested tool or resource does not exist on server            | No         |
/// | `Timeout`  | Server did not respond within the configured timeout           | Yes        |
/// | `Shutdown` | Server exited or graceful shutdown was requested                | No         |
/// | `Internal` | Internal invariant violated in MCP client                       | Yes        |
///
/// **Binary author action:** Transport and timeout errors are good candidates
/// for retry with backoff. NotFound and Shutdown indicate a configuration
/// or lifecycle problem that retry will not fix.
pub use crate::agent_mcp::McpError;

// ============================================================================
// PluginError
// ============================================================================

/// Errors emitted by the plugin subsystem ([`agent_plugin`][crate::agent_plugin]).
///
/// Triggered during plugin initialization, shutdown, hook execution,
/// and registration. Binary authors encounter this error when loading
/// plugins at startup via [`PluginManager::register`][crate::agent_plugin::PluginManager::register]
/// or when hooks fire during agent execution.
///
/// | Variant          | Trigger                                                          | Retryable? |
/// |------------------|------------------------------------------------------------------|------------|
/// | `Initialization` | Plugin failed to initialize (missing dep, bad manifest)          | Yes        |
/// | `Shutdown`       | Plugin returned an error during graceful shutdown                  | No         |
/// | `Hook`           | A registered hook panicked or returned an error                    | Yes        |
/// | `Config`         | Plugin manifest or runtime config is invalid                       | No         |
/// | `Internal`       | Plugin manager invariant violated                                 | Yes        |
///
/// **Binary author action:** Initialization errors may succeed on retry if
/// dependencies become available. Hook errors are typically transient but can
/// indicate a misbehaving plugin. Config errors require the plugin manifest
/// or binary configuration to be corrected.
pub use crate::agent_plugin::PluginError;

// ============================================================================
// SessionError
// ============================================================================

/// Errors emitted by the session subsystem ([`agent_session`][crate::agent_session]).
///
/// Triggered during session save, load, fork, and search operations.
/// Binary authors encounter this error from any [`SessionStore`][crate::agent_session::SessionStore]
/// implementation (JSONL, SQLite, etc.).
///
/// | Variant       | Trigger                                                         | Retryable? |
/// |---------------|-----------------------------------------------------------------|------------|
/// | `NotFound`    | Requested session ID does not exist in the store                 | No         |
/// | `SaveFailed`  | Write to disk failed (full disk, ENOMEM, permissions)            | Yes        |
/// | `LoadFailed`  | Read failed (corrupt file, I/O error)                            | Yes        |
/// | `ForkFailed`  | Fork operation failed (source not found, write failure)           | Yes        |
/// | `SearchFailed`| Query execution failed (invalid query, I/O error)               | Yes        |
/// | `Internal`    | Session store invariant violated                                 | No         |
///
/// **Binary author action:** Save, load, fork, and search failures are good
/// candidates for retry. NotFound requires the caller to handle the missing
/// session case. Internal errors indicate a bug in the store implementation.
pub use crate::agent_session::SessionError;

// ============================================================================
// ToolError
// ============================================================================

/// Errors emitted by the tool subsystem ([`agent_tools`][crate::agent_tools]).
///
/// Triggered during tool registration, execution, permission checks,
/// and timeout handling. Binary authors encounter this error from
/// [`ToolExecutor::execute_batch`][crate::agent_tools::ToolExecutor::execute_batch]
/// and when checking tool availability.
///
/// | Variant           | Trigger                                                        | Retryable? |
/// |-------------------|----------------------------------------------------------------|------------|
/// | `Execution`       | Tool panicked, returned an error, or produced invalid output   | No         |
/// | `NotFound`        | Requested tool name is not registered                           | No         |
/// | `Timeout`         | Tool did not complete within the configured timeout              | Yes        |
/// | `PermissionDenied`| Permission policy blocked the tool call                       | No         |
/// | `Internal`        | Tool registry invariant violated                               | Yes        |
///
// **Binary author action:** Timeout is retryable (tool may succeed on a second
// call). Execution, NotFound, and PermissionDenied are not — they require
// the caller to adjust the request. Log internal errors and consider
// retrying once with backoff.
pub use crate::agent_tools::ToolError;
