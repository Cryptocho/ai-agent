//! Agent session module.
//!
//! Session and conversation state management. A `Session` tracks the full
//! conversation history including messages, metadata, compaction records,
//! and any forks. The `SessionStore` trait provides a pluggable interface for
//! persistence, supporting any backend (JSONL, SQLite, etc.) without the
//! crate itself making assumptions about storage format.
//!

use crate::agent_types::{SearchHit, SessionId};
use async_trait::async_trait;
use std::collections::HashMap;
use std::fmt;

/// Errors that can occur during session operations.
#[derive(Debug, Clone)]
pub enum SessionError {
    /// Session not found.
    NotFound(String),
    /// Failed to save session.
    SaveFailed(String),
    /// Failed to load session.
    LoadFailed(String),
    /// Failed to fork session.
    ForkFailed(String),
    /// Failed to search sessions.
    SearchFailed(String),
    /// Internal error.
    Internal(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::NotFound(s) => write!(f, "Session not found: {}", s),
            SessionError::SaveFailed(s) => write!(f, "Failed to save session: {}", s),
            SessionError::LoadFailed(s) => write!(f, "Failed to load session: {}", s),
            SessionError::ForkFailed(s) => write!(f, "Failed to fork session: {}", s),
            SessionError::SearchFailed(s) => write!(f, "Search failed: {}", s),
            SessionError::Internal(s) => write!(f, "Internal error: {}", s),
        }
    }
}

impl std::error::Error for SessionError {}

impl SessionError {
    /// Returns true if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            SessionError::NotFound(_) => false,
            SessionError::SaveFailed(_) => true,
            SessionError::LoadFailed(_) => true,
            SessionError::ForkFailed(_) => true,
            SessionError::SearchFailed(_) => true,
            SessionError::Internal(_) => false,
        }
    }
}

/// Role of a message sender.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Content block within a message.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    ToolUse {
        tool_name: String,
        tool_input: String,
        tool_call_id: Option<String>,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

/// A single message in the conversation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub name: Option<String>,
    pub cache_control: Option<String>,
}

/// A session representing a conversation thread.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Session {
    pub id: SessionId,
    pub messages: Vec<Message>,
    pub metadata: HashMap<String, String>,
    /// Parent session ID for fork tracking.
    pub parent_id: Option<SessionId>,
}

/// Query for searching sessions.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionSearchQuery {
    pub text_match: String,
    pub limit: usize,
    pub offset: usize,
}

/// Session persistence trait.
///
/// Provides an interface for saving, loading, forking, and searching sessions.
/// Implementors may use any storage backend (JSONL, SQLite, etc.).
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Save a session to storage.
    async fn save(&self, session: &Session) -> Result<(), SessionError>;

    /// Load a session by ID.
    async fn load(&self, id: &SessionId) -> Result<Session, SessionError>;

    /// Fork a session, creating a deep copy with a new ID.
    ///
    /// The forked session contains all messages from the original up to the fork point,
    /// and its `parent_id` is set to the original session's ID.
    async fn fork(&self, id: &SessionId, new_id: SessionId) -> Result<Session, SessionError>;

    /// Search sessions by text content.
    async fn search(&self, query: SessionSearchQuery) -> Result<Vec<SearchHit>, SessionError>;
}
