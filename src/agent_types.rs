//! Agent types module.
//!
//! This module defines foundational types shared across all other modules in
//! the crate. It serves as the canonical home for primitive types that have no
//! other logical module boundary.
//!
//! ## Primary Types
//!
//! - [`SessionId`] — A newtype wrapper around `String` serving as the primary
//!   key for session storage and telemetry. The wrapper ensures type safety and
//!   prevents confusion with arbitrary strings throughout the codebase.
//!
//! - [`TokenUsage`] — Tracks token consumption for an individual LLM call.
//!   Contains prompt, completion, and total token counts as reported by the
//!   LLM provider. These values are estimates and may vary slightly between
//!   providers or model versions.
//!
//! - [`SearchHit`] — Represents a single result from a session content search.
//!   Each hit includes the associated session ID, a text snippet of matched
//!   content, and a relevance score for ranking results.
//!
//! ## Design Rationale
//!
//! Keeping these types in a dedicated module avoids circular dependencies and
//! provides a stable public API surface. Other modules import only what they
//! need from here, and any type with cross-cutting usage belongs in this module.
//!

use std::fmt::Display;
use std::hash::Hash;
use serde::{Deserialize, Serialize};

/// A newtype wrapper around `String` representing a unique session identifier.
///
/// `SessionId` is the primary key used throughout the crate for session storage,
/// retrieval, and telemetry. The wrapper provides type safety by distinguishing
/// session identifiers from arbitrary strings, preventing accidental misuse such
/// as passing a raw ID where a session reference is expected.
///
/// # (De)serialization
///
/// The type is marked `#[repr(transparent)]` and derives `Serialize`/`Deserialize`
/// from `serde`, making it transparent in JSON and other formats. The inner
/// `String` is directly exposed during serialization, so `"session-abc123"` in
/// JSON corresponds to `SessionId("session-abc123".to_string())` in Rust.
///
/// # Example
///
/// ```
/// use agent_types::SessionId;
///
/// let id = SessionId("sess-001".to_string());
/// assert_eq!(id.0, "sess-001");
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

/// Tracks token usage for a single LLM API call.
///
/// Token counts are reported by the LLM provider and represent the provider's
/// own accounting. Values may differ slightly between providers or between
/// model versions from the same provider. Applications should treat these as
/// estimates suitable for cost tracking and context window management, not as
/// exact authoritative counts.
///
/// # Fields
///
/// - `prompt_tokens` — Number of tokens in the input prompt. For models that
///   support caching, this may exclude tokens served from a cached context.
/// - `completion_tokens` — Number of tokens in the model's generated response.
/// - `total_tokens` — Sum of `prompt_tokens` and `completion_tokens`. This
///   field provides the simplest summary for logging and billing purposes.
///
/// # Usage in Sessions
///
/// When a [`Message`] carries token usage data, it reflects the cost and context
/// consumed by that individual turn. Aggregating `total_tokens` across all
/// messages in a session yields the session's total token footprint.
///
/// [`Message`]: crate::agent_session::Message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
    /// Total tokens used (prompt + completion).
    pub total_tokens: u32,
}

/// A single result from a session content search.
///
/// `SearchHit` represents one match found when searching the content of stored
/// sessions. The hit contains enough context to display a result preview and
/// to retrieve the full session if the user selects this result.
///
/// # Fields
///
/// - `session_id` — Identifies which session this hit belongs to. Use this to
///   load the full session via [`SessionStore::load`].
/// - `snippet` — A short excerpt of the matched text. The excerpt is typically
///   bounded to a maximum length and may include ellipsis if the match spans
///   a large region of content.
/// - `relevance_score` — A floating-point score indicating how closely the hit
///   matches the query. Higher scores indicate stronger relevance. Score ranges
///   are backend-specific; do not assume a particular scale.
///
/// [`SessionStore::load`]: crate::agent_session::SessionStore::load
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Session ID associated with this hit.
    pub session_id: SessionId,
    /// Snippet of matched content.
    pub snippet: String,
    /// Relevance score (higher = more relevant).
    pub relevance_score: f32,
}
