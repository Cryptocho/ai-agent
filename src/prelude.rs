//! Prelude — commonly used types re-exported for convenience.
//!
//! Binary authors can import everything they need with a single glob:
//!
//! ```
//! use ai_agent::prelude::*;
//! ```

#![allow(unused_imports)]
//! Prelude re-exports are intentionally unused within the crate — they are
//! intended for consumption by downstream binary authors via `use ai_agent::prelude::*`.
// ============================================================================
// agent_types
// ============================================================================
pub use crate::agent_types::SearchHit;
pub use crate::agent_types::SessionId;
pub use crate::agent_types::TokenUsage;

// ============================================================================
// agent_provider
// ============================================================================
pub use crate::agent_provider::ChatRequest;
pub use crate::agent_provider::ChatResponse;
pub use crate::agent_provider::CredentialProvider;
pub use crate::agent_provider::LlmClient;
pub use crate::agent_provider::ModelEvent;
pub use crate::agent_provider::ProviderFallbacks;
pub use crate::agent_provider::ProviderKind;

// ============================================================================
// agent_tools
// ============================================================================
pub use crate::agent_tools::Tool;
pub use crate::agent_tools::ToolDefinition;
pub use crate::agent_tools::ToolExecutor;
pub use crate::agent_tools::PermissionLevel;
// ToolError re-exported via agent_error below

// ============================================================================
// agent_session
// ============================================================================
pub use crate::agent_session::ContentBlock;
pub use crate::agent_session::Message;
pub use crate::agent_session::Role;
pub use crate::agent_session::Session;
pub use crate::agent_session::SessionSearchQuery;
pub use crate::agent_session::SessionStore;
// SessionError re-exported via agent_error below

// ============================================================================
// agent_permissions
// ============================================================================
pub use crate::agent_permissions::PermissionMode;
pub use crate::agent_permissions::PermissionOutcome;
pub use crate::agent_permissions::PermissionPolicy;
pub use crate::agent_permissions::PermissionPrompter;

// ============================================================================
// agent_hooks
// ============================================================================
pub use crate::agent_hooks::HookContext;
pub use crate::agent_hooks::HookEvent;
pub use crate::agent_hooks::HookResult;
pub use crate::agent_hooks::HookRunner;

// ============================================================================
// agent_mcp
// ============================================================================
pub use crate::agent_mcp::McpClient;
pub use crate::agent_mcp::McpTransport;

// ============================================================================
// agent_plugin
// ============================================================================
pub use crate::agent_plugin::Plugin;
pub use crate::agent_plugin::PluginManager;
pub use crate::agent_plugin::PluginManifest;

// ============================================================================
// agent_config
// ============================================================================
pub use crate::agent_config::AgentConfig;
pub use crate::agent_config::ConfigLoader;

// ============================================================================
// agent_core
// ============================================================================
pub use crate::agent_core::Agent;
pub use crate::agent_core::AgentContext;
// AgentError re-exported via agent_error below (avoids duplicate re-export)
pub use crate::agent_core::Event;
pub use crate::agent_core::IterationBudget;

// ============================================================================
// agent_error (re-exports from source modules)
// ============================================================================
pub use crate::agent_error::ConfigError;
pub use crate::agent_error::McpError;
pub use crate::agent_error::PluginError;
pub use crate::agent_error::SessionError;
pub use crate::agent_error::ToolError;
