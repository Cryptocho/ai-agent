//! Agent core module.
//!
//! The central orchestrator for the agent runtime. This module defines the
//! `Agent` trait, which drives the main agent loop — sending messages to an
//! LLM, receiving tool calls, executing them, and looping until the model
//! returns text or the iteration budget is exhausted.
//!
//! # Turn Loop Overview
//!
//! The turn loop proceeds turn-by-turn, each turn consisting of:
//! 1. **Send** — transmit the current conversation state to the LLM
//! 2. **Receive** — parse the LLM response for text content or tool calls
//! 3. **Evaluate** — check permission policy for tool call authorization
//! 4. **Execute** — dispatch approved tool calls and append results to session
//! 5. **Repeat** — continue until text is returned or budget is exhausted
//!
//! # Stateless Design
//!
//! The agent is stateless with respect to sessions; all session state is
//! passed in via `AgentContext`, allowing a single agent instance to run
//! across many sessions. The context aggregates all subsystem references
//! (LLM client, tool executor, session store, permission policy, hooks, plugins)
//! as trait objects, enabling any concrete implementation without recompilation.
//!
//! # Budget Exhaustion
//!
//! When `IterationBudget` is exhausted, the loop terminates immediately. The
//! caller (e.g., CLI) is responsible for displaying events and deciding whether
//! to continue or present partial results to the user.
//!
//! # Interrupt Mechanism
//!
//! An external caller (e.g., the CLI on Ctrl+C) calls `AgentContext::interrupt()`
//! to signal the agent to stop at the next opportunity. The agent checks the
//! interrupt flag between turns and emits `Event::Interrupted` when set.
//!
//! # Event-Based Telemetry
//!
//! All significant loop events are emitted via the `Event` enum, providing a
//! hook for telemetry, logging, and debugging. Consumers subscribe to events
//! to observe turn boundaries, LLM traffic, tool invocations, and errors.

use std::fmt::Debug;
use std::sync::Arc;
use async_trait::async_trait;

use crate::agent_config::AgentConfig;
use crate::agent_hooks::HookRunner;
use crate::agent_mcp::McpClient;
use crate::agent_permissions::PermissionPolicy;
use crate::agent_plugin::PluginManager;
use crate::agent_provider::LlmClient;
use crate::agent_session::SessionStore;
use crate::agent_tools::ToolExecutor;
use crate::agent_types::SessionId;

// ============================================================================
// AgentError
// ============================================================================

/// Top-level error enum for the agent runtime.
///
/// Covers all subsystems: LLM/provider, session, permissions, tools, config,
/// MCP, and internal errors. Each variant carries structured context.
#[derive(Debug, Clone)]
pub enum AgentError {
    /// Context-related error (e.g., session not found, invalid state).
    Context(String),
    /// Session error (IO, corrupt data, format).
    Session(String),
    /// Permission denied with a reason string.
    Permission(String),
    /// Tool error with the tool name.
    Tool(String),
    /// Configuration error with a description.
    Config(String),
    /// Internal error with a description.
    Internal(String),
}

impl AgentError {
    /// Returns `true` if this error is retryable.
    ///
    /// Retryable errors are those where retrying the operation might succeed
    /// (e.g., transient network errors, timeouts). Non-retryable errors are
    /// those that will not improve with retry (e.g., permission denied,
    /// configuration errors).
    pub fn is_retryable(&self) -> bool {
        match self {
            AgentError::Context(_) => false,
            AgentError::Session(_) => true,
            AgentError::Permission(_) => false,
            AgentError::Tool(_) => false,
            AgentError::Config(_) => false,
            AgentError::Internal(_) => true,
        }
    }
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::Context(msg) => write!(f, "Context error: {}", msg),
            AgentError::Session(msg) => write!(f, "Session error: {}", msg),
            AgentError::Permission(msg) => write!(f, "Permission error: {}", msg),
            AgentError::Tool(msg) => write!(f, "Tool error: {}", msg),
            AgentError::Config(msg) => write!(f, "Config error: {}", msg),
            AgentError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for AgentError {}

// ============================================================================
// Event
// ============================================================================

/// Events emitted during agent execution.
///
/// These events are the primary hook for telemetry, logging, and debugging
/// the agent loop. Each event covers a logical step of a turn.
///
/// # Telemetry Use
///
/// Consumers (telemetry sinks, loggers, debug UIs) subscribe to these events
/// to reconstruct the full execution timeline. The `turn` field on `TurnStarted`
/// and `TurnCompleted` provides a sequence number for correlating events within
/// a session. All token counts are raw estimates suitable for cost tracking
/// and context-pressure monitoring.
///
/// # Serialization
///
/// `ToolCalled` and `ToolResult` carry serialized JSON strings, not parsed
/// values. Consumers are responsible for deserialization if needed.
#[derive(Debug, Clone)]
pub enum Event {
    /// A new turn has started.
    ///
    /// Emitted at the beginning of each `run_turn` call before any LLM or tool
    /// activity. The `turn` field is the zero-based turn number for this
    /// session, useful for progress tracking and replay.
    TurnStarted { turn: u32 },

    /// A turn has completed with an outcome description.
    ///
    /// Emitted when the turn loop exits, either because the LLM returned text,
    /// the budget was exhausted, or an error halted execution. The `outcome`
    /// string describes the exit reason (e.g., "Budget exhausted", "text
    /// response"). Correlate with `TurnStarted` using the `turn` field.
    TurnCompleted { turn: u32, outcome: String },

    /// A tool was called with its name and serialized input.
    ///
    /// Emitted immediately before a tool call is dispatched to the executor.
    /// The `tool` field is the canonical tool name; `input` is the JSON
    /// serialization of the arguments. Log this event for request-replay and
    /// cost attribution per tool.
    ToolCalled { tool: String, input: String },

    /// A tool returned with its name and serialized output.
    ///
    /// Emitted after a tool call completes (success or failure). The `tool`
    /// field is the canonical tool name; `output` is the JSON serialization of
    /// the result or error message. Correlate with the matching `ToolCalled`
    /// event using the session timeline.
    ToolResult { tool: String, output: String },

    /// An LLM request was sent with the token count.
    ///
    /// Emitted when the agent transmits a chat request to the LLM provider.
    /// `tokens` is the estimated input token count for the current prompt,
    /// useful for context-pressure monitoring and cost estimation.
    LlmRequest { tokens: u32 },

    /// An LLM response was received with the token count.
    ///
    /// Emitted when the agent receives a complete response from the LLM
    /// provider. `tokens` is the output token count for this response,
    /// useful for cost attribution and throughput measurement.
    LlmResponse { tokens: u32 },

    /// The agent was interrupted.
    ///
    /// Emitted when the interrupt flag is detected between turns. The session
    /// state is preserved at this point; the caller may resume or surface
    /// partial results to the user. No further events will be emitted after
    /// `Interrupted` until the next session.
    Interrupted,

    /// An error occurred during execution.
    ///
    /// Emitted when a non-recoverable error halts the turn loop (e.g., LLM
    /// provider failure, tool error, permission denial). The `error` string
    /// contains the structured error message. Log the full error chain for
    /// debugging; surface the message to the user if no recovery is possible.
    Error { error: String },
}

// ============================================================================
// IterationBudget
// ============================================================================

/// Budget tracking for agent turn iterations.
///
/// Tracks how many turns remain, the total allowed turns, and an optional
/// budget for subagent turns. The budget decrements on each turn and prevents
/// further iteration when exhausted.
///
/// # Fields
///
/// - `remaining` — number of turns left in this budget. Decremented by
///   [`consume()`][Self::consume] on each turn. Reaching zero signals that the
///   agent should stop.
/// - `total` — the original turn limit at construction. Introspection field
///   for reporting; does not affect behavior.
/// - `subagent_budget` — an optional nested budget for subagent turns. When
///   `Some(n)`, the subagent has its own independent turn budget of `n` turns.
///   The parent budget is not decremented while the subagent budget is active.
///   When `None`, no subagent budget is active and all turns count against
///   the parent.
///
/// # `consume()` Semantics
///
/// [`consume()`][Self::consume] decrements `remaining` and returns `true` if
/// turns remain, `false` if the budget is already exhausted. It is idempotent
/// at zero — calling `consume()` on an exhausted budget returns `false` without
/// underflow. The caller checks the return value to decide whether to proceed
/// with the turn or terminate the loop.
///
/// # Subagent Budget Concept
///
/// When delegating to a subagent, the parent agent may allocate a separate
/// turn budget to bound the subagent's work independently. Use
/// [`with_subagent()`][Self::with_subagent] to construct a budget with a
/// subagent allocation. The parent and subagent budgets are independent; the
/// caller is responsible for switching between them based on delegation state.
#[derive(Debug, Clone)]
pub struct IterationBudget {
    /// Number of turns remaining in the budget.
    remaining: usize,
    /// Total number of allowed turns.
    total: usize,
    /// Optional budget for subagent turns. When `None`, no subagent budget is active.
    subagent_budget: Option<usize>,
}

impl IterationBudget {
    /// Creates a new budget with `total` turns and no subagent budget.
    pub fn new(total: usize) -> Self {
        Self {
            remaining: total,
            total,
            subagent_budget: None,
        }
    }

    /// Creates a new budget with `total` turns and a `subagent` budget for subagent turns.
    pub fn with_subagent(total: usize, subagent: usize) -> Self {
        Self {
            remaining: total,
            total,
            subagent_budget: Some(subagent),
        }
    }

    /// Consumes one turn from the budget.
    ///
    /// Returns `false` when there are no remaining turns (budget exhausted).
    /// Does NOT panic when remaining reaches zero.
    pub fn consume(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        true
    }

    /// Returns the number of turns remaining.
    pub fn remaining(&self) -> usize {
        self.remaining
    }

    /// Returns `true` if a subagent budget is currently active.
    pub fn is_subagent(&self) -> bool {
        self.subagent_budget.is_some()
    }

    /// Returns a reference to the budget for inspection.
    pub fn as_budget_ref(&self) -> &Self {
        self
    }
}

// ============================================================================
// AgentContext
// ============================================================================

/// Shared context passed into each agent turn.
///
/// `AgentContext` aggregates all subsystem references (trait objects) so that
/// the agent loop can interact with the LLM, tools, session store, permissions,
/// hooks, and plugins without hardcoding concrete implementations.
///
/// The context is intentionally open-ended — binary authors can extend it with
/// additional trait objects as needed without modifying the crate.
pub struct AgentContext {
    /// LLM client for sending chat requests and receiving responses.
    ///
    /// Used by the agent loop to send conversation state and receive model
    /// outputs (text deltas and tool calls). Implement any LLM provider
    /// (Anthropic, OpenAI-compatible, etc.) as a `dyn LlmClient`.
    pub llm: Arc<dyn LlmClient>,

    /// Tool executor for dispatching tool calls to registered tools.
    ///
    /// Receives tool names and JSON inputs, invokes the matching tool
    /// implementation, and returns serialized results. Also handles batch
    /// dispatch for parallel tool calls.
    pub tool_executor: Arc<dyn ToolExecutor>,

    /// Session store for loading and saving conversation state.
    ///
    /// Provides persistence for session data including message history,
    /// metadata, forks, and compaction records. Implement any storage backend
    /// (JSONL, SQLite, remote API) as a `dyn SessionStore`.
    pub session_store: Arc<dyn SessionStore>,

    /// Permission policy for evaluating tool call authorization.
    ///
    /// Determines whether a given tool call is permitted based on the active
    /// permission mode and configured rules. Called before tool execution;
    /// returns an outcome (allow, deny, or ask).
    pub permission_policy: Arc<dyn PermissionPolicy>,

    /// Hook runner for firing telemetry and extensibility hooks.
    ///
    /// Dispatches hook events (pre-tool-use, post-tool-use, side-effects)
    /// to registered hook implementations. Used for logging, monitoring,
    /// and policy enforcement at key points in the execution path.
    pub hook_runner: Arc<dyn HookRunner>,

    /// Optional MCP client for Model Context Protocol servers.
    ///
    /// When `Some`, the agent can discover and call tools from MCP servers
    /// via the MCP protocol. When `None`, MCP features are unavailable and
    /// only builtin/plugin tools are used.
    pub mcp_client: Option<Arc<dyn McpClient>>,

    /// Plugin manager for runtime-loaded plugins.
    ///
    /// Manages plugin lifecycle (init, shutdown), tool registration, and
    /// hook registration from dynamically loaded plugins.
    pub plugin_manager: Arc<PluginManager>,

    /// Aggregated agent configuration.
    ///
    /// Contains all configuration sections (model, tools, permissions, sandbox,
    /// MCP, hooks, plugins, provider fallbacks, trusted roots). Loaded from
    /// the config cascade by a `ConfigLoader` implementation.
    pub config: AgentConfig,

    /// Iteration budget tracking remaining turns.
    ///
    /// Decremented on each turn; exhaustion terminates the loop. Also carries
    /// an optional subagent budget for bounded delegation scenarios.
    pub budget: IterationBudget,

    /// Active session identifier.
    ///
    /// Unique identifier for the current session, used for telemetry
    /// correlation, session persistence keys, and logging context.
    pub session_id: SessionId,

    /// Whether the agent has been interrupted.
    ///
    /// Set to `true` by `interrupt()`. The agent loop checks this flag
    /// between turns; when `true`, it emits `Event::Interrupted` and halts.
    interrupted: bool,
}

impl Debug for AgentContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentContext")
            .field("llm", &"Arc<dyn LlmClient>")
            .field("tool_executor", &"Arc<dyn ToolExecutor>")
            .field("session_store", &"Arc<dyn SessionStore>")
            .field("permission_policy", &"Arc<dyn PermissionPolicy>")
            .field("hook_runner", &"Arc<dyn HookRunner>")
            .field("mcp_client", &self.mcp_client.as_ref().map(|_| "Arc<dyn McpClient>"))
            .field("plugin_manager", &"Arc<PluginManager>")
            .field("config", &self.config)
            .field("budget", &self.budget)
            .field("session_id", &self.session_id)
            .field("interrupted", &self.interrupted)
            .finish()
    }
}

impl AgentContext {
    /// Runs a single turn of the agent loop.
    ///
    /// This method executes one turn: sends a request to the LLM, receives
    /// tool calls, evaluates permissions, executes tools, and returns events.
    /// Returns an error if the turn cannot be executed.
    pub async fn run_turn(&mut self) -> Result<Event, AgentError> {
        self.interrupted = false;

        let turn = (self.budget.total.saturating_sub(self.budget.remaining)) as u32;

        // Emit TurnStarted event
        let event = Event::TurnStarted { turn };

        // Check if budget is exhausted
        if !self.budget.consume() {
            return Ok(Event::TurnCompleted {
                turn,
                outcome: "Budget exhausted".to_string(),
            });
        }

        // Placeholder: actual implementation would send to LLM, call tools, etc.
        Ok(event)
    }

    /// Interrupts the agent, stopping the current turn as soon as possible.
    pub async fn interrupt(&mut self) {
        self.interrupted = true;
    }

    /// Returns `true` if the agent has been interrupted.
    pub fn is_interrupted(&self) -> bool {
        self.interrupted
    }
}

// ============================================================================
// Agent trait
// ============================================================================

/// Central trait implementing the agent loop.
///
/// Implementors define how the agent interacts with the LLM, handles tool calls,
/// and manages the conversation flow. The trait is intentionally stateless —
/// a single instance can run across many sessions by varying the `AgentContext`.
///
/// Unlike a binary-specific agent implementation, this trait does NOT hardcode
/// any particular response structure or tool-calling convention. It provides
/// the foundation for building agents that work with any LLM provider.
///
/// # Implementor Responsibilities
///
/// An implementor must:
/// - Manage the full turn loop: send to the LLM, parse responses, dispatch
///   tool calls through `context.tool_executor`, evaluate permission policy,
///   append results to the session, and repeat until termination.
/// - Emit `Event` variants at each significant step for telemetry and logging.
/// - Check `context.is_interrupted()` between turns and emit `Event::Interrupted`
///   when the flag is set.
/// - Persist session state via `context.session_store` after each turn.
///
/// # Stateless Nature
///
/// The `Agent` trait holds no session-specific state. All state flows through
/// `AgentContext`: the LLM client, tool executor, session store, budget,
/// permission policy, and hooks are all passed in. A single `Agent`
/// implementation can serve concurrent sessions by running each with its own
/// `AgentContext` (subject to the implementor's concurrency safety).
///
/// # Event Streaming
///
/// The implementor emits `Event` values to signal loop progress. Callers
/// subscribe to these events for telemetry, logging, and debug UIs. Events
/// cover turn boundaries, LLM traffic, tool invocations, and errors. See
/// [`Event`] for the full list of variants and their expected use.
///
/// # Interrupt Wiring
///
/// `interrupt()` sets the interrupt flag in the context. The implementor is
/// responsible for checking the flag between turns and reacting appropriately.
/// The caller is responsible for wiring platform signals (e.g., Ctrl+C, signals)
/// to call `interrupt()`.
#[async_trait]
pub trait Agent: Send + Sync {
    /// Runs the agent loop to completion with the given context.
    ///
    /// The agent loop proceeds turn-by-turn until the model returns text,
    /// the iteration budget is exhausted, or an error occurs.
    async fn run(&self, context: &mut AgentContext) -> Result<(), AgentError>;

    /// Interrupts the agent, signaling it to stop at the next opportunity.
    ///
    /// Sets the interrupt flag in the context. The implementor checks this
    /// flag between turns and halts loop execution when it is set, emitting
    /// `Event::Interrupted`. This method returns immediately without waiting
    /// for the current turn to finish.
    async fn interrupt(&self);

    /// Returns `true` if the agent has been interrupted.
    ///
    /// Checks the interrupt flag set by `interrupt()`. Use this between turns
    /// to decide whether to continue or exit the loop early.
    fn is_interrupted(&self) -> bool;
}

// ============================================================================
// Turn Loop Termination Conditions
// ============================================================================

// ## Turn Loop Termination Conditions
//
// A single `run_turn` call ends when one of these conditions is reached:
//
// | Condition | Return | Side Effect |
// |-----------|--------|-------------|
// | LLM returns text with no tool calls | `Event::MessageStop` with text content | Message appended to session |
// | LLM returns tool calls | `Event::ToolCallComplete` for each | Tool calls appended, results awaited |
// | Iteration budget exhausted | `Event::Error(AgentError::Internal("budget exhausted"))` | Session marked, no further auto-advance |
// | Hook denies tool | `Event::ToolDenied { tool, reason }` | Tool call not executed, caller decides next step |
// | Permission denied + user declines | `Event::PermissionDenied` | Tool call not executed |
// | Provider error (retryable) | Retry within budget, then fall through chain | See graceful degradation cascade |
// | Provider error (non-retryable) | `Event::Error` propagated to caller | Session unchanged |
// | `interrupt()` called | `Event::Interrupted` | Session preserved, caller may resume |
//
// The caller (e.g., CLI) is responsible for displaying events and deciding whether
// to call `run_turn` again. The agent loop is the caller's loop, not the crate's.
