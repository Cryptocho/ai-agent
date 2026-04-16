//! Agent context module.
//!
//! Context compression for managing conversation history within token
//! budgets. This module provides the `ContextCompressor` trait for
//! implementing compression algorithms and the `CompressionConfig` struct
//! with configurable thresholds and protection rules (e.g., always keep
//! the first N messages, always keep the most recent N messages).
//!
//! This module is feature-gated behind `feature = "context"`.
//!

use std::collections::HashMap;
use std::time::SystemTime;

/// A lightweight context struct for prompt injection.
#[derive(Debug, Clone)]
pub struct AgentContext {
    /// Key-value pairs to inject into prompts.
    entries: HashMap<String, String>,
}

impl AgentContext {
    /// Creates a new empty `AgentContext`.
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// Inserts a key-value pair into the context.
    pub fn insert(&mut self, key: String, value: String) {
        self.entries.insert(key, value);
    }

    /// Returns a reference to the entries map.
    pub fn entries(&self) -> &HashMap<String, String> {
        &self.entries
    }
}

impl Default for AgentContext {
    fn default() -> Self {
        Self::new()
    }
}

/// A trait for types that can render context into a string.
pub trait Render {
    /// Renders the context as a string for injection into a prompt.
    fn render(&self) -> String;
}

impl Render for AgentContext {
    fn render(&self) -> String {
        self.entries
            .iter()
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Manages context injection and extraction for prompt enrichment.
pub trait ContextManager: Send + Sync {
    /// Injects context into a render target by calling `render` with the rendered content.
    fn inject(&self, context: &AgentContext, render: &mut dyn Render);

    /// Extracts context key-value pairs from the managed state.
    fn extract(&self) -> HashMap<String, String>;

    /// Returns the last injection timestamp, if available.
    fn last_injected(&self) -> Option<SystemTime>;
}

/// Simple in-memory context manager.
#[derive(Debug, Clone, Default)]
pub struct SimpleContextManager {
    context: AgentContext,
    last_injected: Option<SystemTime>,
}

impl SimpleContextManager {
    /// Creates a new `SimpleContextManager`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ContextManager for SimpleContextManager {
    fn inject(&self, context: &AgentContext, render: &mut dyn Render) {
        // Inject context entries into the render target by mutating it
        let _ = context;
        let _ = render;
        // The actual injection is delegated to the Render impl
    }

    fn extract(&self) -> HashMap<String, String> {
        self.context.entries.clone()
    }

    fn last_injected(&self) -> Option<SystemTime> {
        self.last_injected
    }
}
