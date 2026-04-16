//! Agent tools module.
//!
//! Tool registry and execution layer. This module defines the `Tool` trait
//! that all callable tools must implement, along with the `ToolExecutor` trait
//! for dispatching tool calls. Tool definitions are schema-driven
//! (JSON Schema for input validation) and carry an associated permission level
//! used by the permission policy engine.
//!

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

/// Error type for tool execution failures.
#[derive(Debug, Clone)]
pub enum ToolError {
    Execution(String),
    NotFound(String),
    Timeout,
    PermissionDenied(String),
    Internal(String),
}

impl ToolError {
    /// Returns true if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(self, ToolError::Timeout | ToolError::Internal(_))
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::Execution(msg) => write!(f, "Execution error: {}", msg),
            ToolError::NotFound(msg) => write!(f, "Not found: {}", msg),
            ToolError::Timeout => write!(f, "Tool execution timed out"),
            ToolError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            ToolError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for ToolError {}

/// Permission level required to execute a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PermissionLevel {
    Ask,
    Low,
    Medium,
    High,
    Critical,
}

/// Serializable tool descriptor.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: String,
}

/// Core trait for callable tools. All tool execution is synchronous.
pub trait Tool: Send + Sync {
    /// Returns the tool name.
    fn name(&self) -> String;

    /// Returns the tool description.
    fn description(&self) -> String;

    /// Returns the JSON schema for the tool's input.
    fn input_schema(&self) -> String;

    /// Executes the tool with the given JSON input. Always synchronous.
    fn execute(&self, input: &str) -> Result<String, ToolError>;
}

/// Trait for dispatching and managing tool executions.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Executes a batch of tool calls. Short-circuits on fatal errors (NotFound, Execution)
    /// but NOT on Timeout. Returns a map of execution_id -> result.
    async fn execute_batch(
        &self,
        tools: &[Arc<dyn Tool>],
        inputs: HashMap<String, String>,
    ) -> HashMap<String, Result<String, ToolError>>;

    /// Lists all available tools.
    fn list_tools(&self) -> Vec<ToolDefinition>;

    /// Checks if a tool with the given name exists.
    fn has_tool(&self, name: &str) -> bool;

    /// Cancels a running execution by ID.
    async fn cancel(&self, execution_id: Uuid) -> Result<(), ToolError>;
}
