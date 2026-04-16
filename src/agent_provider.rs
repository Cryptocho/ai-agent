//! Agent provider module.
//!
//! LLM client abstraction layer. This module defines the `LlmClient` trait,
//! which provides a unified interface for talking to any LLM provider
//! (Anthropic, OpenAI-compatible, xAI, or custom). Providers are exposed as
//! trait objects, allowing binary authors to swap implementations without
//! recompiling the crate.
//!
//! The module also includes `ProviderFallbacks`, a struct that implements a
//! configurable fallback chain across multiple providers with retry logic.
//!

use async_trait::async_trait;

// ============================================================================
// ProviderKind
// ============================================================================

/// Closed enum of supported LLM providers.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    Xai,
    Custom(String),
}

// ============================================================================
// ChatRequest (open enum)
// ============================================================================

/// Open enum of provider-specific chat requests.
///
/// Binary authors can add variants via the `Custom` variant.
#[derive(Debug, Clone)]
pub enum ChatRequest {
    Anthropic(AnthropicRequest),
    OpenAi(OpenAiRequest),
    Xai(XaiRequest),
    Custom(String, Box<serde_json::Value>),
}

/// Placeholder request type for Anthropic.
#[derive(Debug, Clone)]
pub struct AnthropicRequest(pub serde_json::Value);

/// Placeholder request type for OpenAI.
#[derive(Debug, Clone)]
pub struct OpenAiRequest(pub serde_json::Value);

/// Placeholder request type for xAI.
#[derive(Debug, Clone)]
pub struct XaiRequest(pub serde_json::Value);

// ============================================================================
// ChatResponse (closed enum)
// ============================================================================

/// Closed enum of provider-specific chat responses.
#[derive(Debug, Clone)]
pub enum ChatResponse {
    Anthropic(Box<AnthropicResponse>),
    OpenAi(Box<OpenAiResponse>),
    Xai(Box<XaiResponse>),
    Custom(String, Box<serde_json::Value>),
}

/// Placeholder response type for Anthropic.
#[derive(Debug, Clone)]
pub struct AnthropicResponse(pub serde_json::Value);

/// Placeholder response type for OpenAI.
#[derive(Debug, Clone)]
pub struct OpenAiResponse(pub serde_json::Value);

/// Placeholder response type for xAI.
#[derive(Debug, Clone)]
pub struct XaiResponse(pub serde_json::Value);

// ============================================================================
// ModelEvent (open enum for streaming)
// ============================================================================

/// Open enum of streaming model events.
#[derive(Debug, Clone)]
pub enum ModelEvent {
    TextDelta(String),
    ToolCallStart { name: String },
    ToolCallDelta { name: String, delta: String },
    ToolCallComplete { name: String },
    Usage {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    },
    MessageStop,
}

// ============================================================================
// LlmClient trait
// ============================================================================

/// Core trait for LLM API calls (streaming and non-streaming).
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Returns the provider kind for this client.
    fn provider(&self) -> ProviderKind;

    /// Send a chat request and receive a non-streaming response.
    async fn chat(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, Box<dyn std::error::Error + Send + Sync>>;

    /// Send a chat request and receive a streaming response.
    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<Box<dyn futures::Stream<Item = ModelEvent> + Send + Sync>, Box<dyn std::error::Error + Send + Sync>>;

    /// Count the number of tokens in a request.
    async fn count_tokens(
        &self,
        request: &ChatRequest,
    ) -> Result<u32, Box<dyn std::error::Error + Send + Sync>>;

    /// Check if a request fits within a given context window.
    fn fits_in_context(&self, request: &ChatRequest, max_tokens: u32) -> bool;
}

// ============================================================================
// ProviderFallbacks
// ============================================================================

/// Struct that manages a fallback chain of providers.
#[derive(Debug, Clone)]
pub struct ProviderFallbacks {
    pub primary: ProviderKind,
    pub fallbacks: Vec<ProviderKind>,
}

impl ProviderFallbacks {
    /// Creates a new `ProviderFallbacks` with a primary provider and an optional fallback chain.
    pub fn new(primary: ProviderKind, fallbacks: Vec<ProviderKind>) -> Self {
        Self { primary, fallbacks }
    }

    /// Returns an iterator over the full chain (primary followed by fallbacks).
    pub fn chain(&self) -> impl Iterator<Item = &ProviderKind> {
        std::iter::once(&self.primary).chain(self.fallbacks.iter())
    }
}

// ============================================================================
// CredentialProvider trait
// ============================================================================

/// Trait for fetching and refreshing API credentials.
pub trait CredentialProvider: Send + Sync {
    /// Get a credential for the given provider.
    fn get_credential(
        &self,
        provider: ProviderKind,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
}

// ============================================================================
// ContextOverage
// ============================================================================

/// Error struct indicating a request exceeds the model's context window.
#[derive(Debug, Clone)]
pub struct ContextOverage {
    pub overage_tokens: u32,
    pub estimated_cost: f64,
}

impl std::fmt::Display for ContextOverage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "context overage: {} tokens over, estimated cost ${:.4}",
            self.overage_tokens, self.estimated_cost
        )
    }
}

impl std::error::Error for ContextOverage {}
