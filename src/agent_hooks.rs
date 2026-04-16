//! Agent hooks module.
//!
//! Extensibility layer for intercepting and reacting to agent events.
//! The hook system allows binary authors to register callbacks that fire
//! before and after tool use, and on side effects (network access, env
//! var changes, filesystem scope, etc.). All hooks are async and fire
//! during the agent loop via the `HookRunner` trait.
//!

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::agent_types::SessionId;

/// Enum of hookable events throughout the agent loop.
#[derive(Debug, Clone)]
pub enum HookEvent {
    /// A tool is being called with its name and serialized input.
    ToolCall { tool_name: String, input: String },
    /// A tool has returned with its name and serialized output.
    ToolResult { tool_name: String, output: String },
    /// An LLM request is being sent.
    LlmRequest { request: Arc<dyn Any + Send + Sync> },
    /// An LLM response has been received.
    LlmResponse { response: Arc<dyn Any + Send + Sync> },
    /// Start of a turn.
    TurnStart { turn: u32 },
    /// End of a turn.
    TurnEnd { turn: u32 },
    /// A fork has occurred.
    Fork { parent_id: String, child_id: String },
    /// A merge has occurred.
    Merge { parent_id: String, child_id: String },
}

/// Context object passed to every hook invocation.
#[derive(Debug, Clone)]
pub struct HookContext {
    /// The event that triggered the hook.
    pub event: HookEvent,
    /// Active session ID, if any.
    pub session_id: Option<SessionId>,
    /// Current turn number, if known.
    pub turn: Option<u32>,
    /// Arbitrary key-value metadata attached by prior hooks or the system.
    pub metadata: HashMap<String, String>,
}

/// Result returned by a hook runner.
#[derive(Debug, Clone)]
pub enum HookResult {
    /// Continue normal execution.
    Continue,
    /// Skip the current operation (e.g., skip a tool call).
    Skip,
    /// Abort the agent loop with an error message.
    Abort(String),
    /// Override the operation's input or output with the given value.
    Override { value: String },
}

/// Permission decision made by a hook or permission checker.
#[derive(Debug, Clone)]
pub enum PermissionDecision {
    /// Operation is allowed.
    Allow,
    /// Operation is denied with a reason.
    Deny(String),
    /// Operation requires user confirmation.
    Ask,
}

/// Core trait for async hook execution. Implementors define what happens
/// when a hook event fires. `HookRunner` is composable via `then()`.
#[async_trait]
pub trait HookRunner: Send + Sync {
    /// Run the hook with the given context. Returns a `HookResult` that
    /// determines whether execution continues, skips, aborts, or is overridden.
    async fn run(&self, context: HookContext) -> HookResult;
}

/// Chains multiple `HookRunner`s together, running each in sequence until
/// one returns `Abort` (short-circuit) or all complete.
pub struct CompositeHookRunner {
    runners: Vec<Arc<dyn HookRunner>>,
}

impl CompositeHookRunner {
    /// Create a new empty composite runner.
    pub fn new() -> Self {
        Self { runners: Vec::new() }
    }

    /// Add a runner to the chain.
    pub fn push(mut self, runner: Arc<dyn HookRunner>) -> Self {
        self.runners.push(runner);
        self
    }
}

impl Default for CompositeHookRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl HookRunner for CompositeHookRunner {
    async fn run(&self, context: HookContext) -> HookResult {
        for runner in &self.runners {
            let result = runner.run(context.clone()).await;
            if matches!(result, HookResult::Abort(_)) {
                return result;
            }
        }
        HookResult::Continue
    }
}

impl Clone for CompositeHookRunner {
    fn clone(&self) -> Self {
        Self {
            runners: self.runners.clone(),
        }
    }
}

#[async_trait]
impl HookRunner for Arc<dyn HookRunner + 'static> {
    async fn run(&self, context: HookContext) -> HookResult {
        (**self).run(context).await
    }
}

/// Wrapper that allows a synchronous function to be used as an async `HookRunner`.
pub struct SyncHookAdapter<F> {
    f: F,
}

impl<F> SyncHookAdapter<F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

#[async_trait]
impl<F: Fn(HookContext) -> HookResult + Send + Sync + 'static> HookRunner for SyncHookAdapter<F> {
    async fn run(&self, context: HookContext) -> HookResult {
        (self.f)(context)
    }
}

/// Compose two hook runners together, running `self` first then `other`.
/// Short-circuits on `Abort`.
pub struct ChainedHookRunner {
    first: Arc<dyn HookRunner>,
    second: Arc<dyn HookRunner>,
}

impl ChainedHookRunner {
    pub fn new(first: Arc<dyn HookRunner>, second: Arc<dyn HookRunner>) -> Self {
        Self { first, second }
    }
}

#[async_trait]
impl HookRunner for ChainedHookRunner {
    async fn run(&self, context: HookContext) -> HookResult {
        let result = self.first.run(context.clone()).await;
        if matches!(result, HookResult::Abort(_)) {
            return result;
        }
        self.second.run(context).await
    }
}

#[async_trait]
impl HookRunner for Box<dyn HookRunner + 'static> {
    async fn run(&self, context: HookContext) -> HookResult {
        (**self).run(context).await
    }
}
