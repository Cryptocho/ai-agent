# AI Agent Crate

A general-purpose toolkit for building AI agents in Rust. This crate provides the foundational traits and composition APIs that binary authors implement to create customized agent runtimes.

## Overview

The crate is structured around a small number of core traits that encapsulate the fundamental operations any agent needs: running a turn with an LLM, dispatching tools, persisting sessions, and evaluating permission requests. Every major component is a trait object, enabling you to substitute custom implementations for any layer without recompiling the crate itself.

**The crate makes no assumptions about:**
- Which LLM providers are available — the `LlmClient` trait is open to any provider
- Which permission levels a tool requires — those are per-tool, not baked in
- How sessions are stored — any `SessionStore` implementation works
- Which compression algorithm is used — any `ContextCompressor` implementation works
- How hooks evaluate decisions — any `HookRunner` implementation works
- What the default iteration budget is — set by the binary

This is a library crate intended for binary authors who want to build tailored agent runtimes. It provides the trait definitions and composition primitives; concrete implementations for specific providers, storage backends, and policy defaults belong in your binary.

## Quickstart

The minimal wiring to run an agent turn requires four components: an `Agent`, an `LlmClient`, a `ToolExecutor`, and a `SessionStore`. Below is the skeletal structure showing how these pieces connect.

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

// 1. Implement the LlmClient trait for your provider
struct MyLlmClient { /* ... */ }
impl agent_provider::LlmClient for MyLlmClient {
    type Stream = /* your stream type */;
    fn model(&self) -> &str { "claude-sonnet-4-20250514" }
    fn provider(&self) -> agent_provider::ProviderKind { /* ... */ }
    async fn chat(&self, req: &agent_provider::ChatRequest)
        -> Result<agent_provider::ChatResponse, Self::Error> { /* ... */ }
    fn chat_stream(&self, req: &agent_provider::ChatRequest)
        -> Result<Self::Stream, Self::Error> { /* ... */ }
    async fn count_tokens(&self, req: &agent_provider::ChatRequest)
        -> Result<u32, Self::Error> { /* ... */ }
    async fn fits_in_context(&self, req: &agent_provider::ChatRequest)
        -> Result<(), agent_provider::ContextOverage> { /* ... */ }
}

// 2. Implement the ToolExecutor trait with your tool registry
struct MyToolExecutor { /* ... */ }
impl agent_tools::ToolExecutor for MyToolExecutor {
    async fn execute(&self, name: &str, input: &serde_json::Value)
        -> Result<String, agent_tools::ToolError> { /* ... */ }
    fn list_tools(&self) -> Vec<agent_tools::ToolDefinition> { /* ... */ }
    fn has_tool(&self, name: &str) -> bool { /* ... */ }
    fn cancel(&self, name: &str) { /* ... */ }
}

// 3. Implement the SessionStore trait for your persistence backend
struct MySessionStore { /* ... */ }
impl agent_session::SessionStore for MySessionStore {
    async fn save(&self, session: &agent_session::Session) -> Result<(), agent_session::SessionError> { /* ... */ }
    async fn load(&self, id: &agent_session::SessionId) -> Result<Option<agent_session::Session>, agent_session::SessionError> { /* ... */ }
    async fn fork(&self, parent: &agent_session::SessionId, name: &str) -> Result<agent_session::Session, agent_session::SessionError> { /* ... */ }
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<agent_session::SearchHit>, agent_session::SessionError> { /* ... */ }
}

// 4. Build the AgentContext and run a turn
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm: Arc<dyn agent_provider::LlmClient> = Arc::new(MyLlmClient::new()?);
    let tool_executor: Arc<dyn agent_tools::ToolExecutor> = Arc::new(MyToolExecutor::new());
    let session_store: Arc<dyn agent_session::SessionStore> = Arc::new(MySessionStore::new()?);

    let session_id = agent_session::SessionId::new_v4();
    let session = session_store.load(&session_id).await?.unwrap_or_else(|| {
        agent_session::Session::new(session_id.clone())
    });

    let context = agent_core::AgentContext {
        session_id,
        session: Arc::new(RwLock::new(session)),
        tool_executor,
        permission_policy: Arc::new(agent_permissions::PermissionPolicy::new(
            agent_permissions::PermissionMode::WorkspaceWrite,
        )),
        compression_llm: Some(Arc::clone(&llm)),
        budget: agent_core::IterationBudget::new(90),
        metadata: Default::default(),
    };

    let mut agent: impl agent_core::Agent = /* your agent implementation */;

    let events = agent.run_turn("Hello, agent.", &context, None).await;
    tokio::pin!(events);

    while let Some(event) = events.next().await {
        println!("{event:?}");
    }

    Ok(())
}
```

This example omits error handling, streaming rendering, and prompter wiring for brevity. Real integrations add permission prompts, hook registration, and event stream consumption appropriate to your use case.

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `mcp` | enabled | MCP client lifecycle management and transport abstraction |
| `memory` | disabled | Memory provider system with fenced context blocks |
| `skills` | disabled | Skills hub and skill source trait |
| `prompt` | enabled | System prompt builder and context file loader |
| `cli` | disabled | CLI/REPL rendering and slash command handling |

Enable or disable features in `Cargo.toml`:

```toml
[dependencies]
ai-agent = { version = "0.1", default-features = false, features = ["mcp", "prompt"] }
```

Disabling unused features reduces compile times and binary size.

## Module Map

### Core Modules (always compiled)

| Module | Description |
|--------|-------------|
| `agent-core` | Agent trait, AgentContext, and IterationBudget |
| `agent-provider` | LlmClient trait, ChatRequest/ModelEvent enums, ProviderFallbacks |
| `agent-tools` | Tool and ToolExecutor traits, PermissionLevel enum |
| `agent-session` | Session, Message, SessionStore trait |
| `agent-mcp` | McpClient trait, McpTransport, lifecycle state machine |
| `agent-hooks` | HookRunner trait, HookContext, HookEvent types |
| `agent-permissions` | PermissionPolicy engine, PermissionMode, rule types |
| `agent-plugin` | Plugin trait, PluginManager, manifest types |
| `agent-config` | AgentConfig struct, ConfigLoader trait |

### Feature-Gated Modules

| Module | Feature | Description |
|--------|---------|-------------|
| `agent-prompt` | `prompt` | SystemPromptBuilder, ContextFileLoader, injection scanning |
| `agent-context` | (always) | ContextCompressor trait, CompressionConfig, Hermes 5-phase algorithm |
| `agent-telemetry` | (always) | TelemetrySink trait, SessionTracer |

### Additional Modules

| Module | Feature | Description |
|--------|---------|-------------|
| `agent-memory` | `memory` | MemoryProvider trait, MemoryManager |
| `agent-skills` | `skills` | SkillSource trait, SkillsManager |
| `agent-gateway` | (always) | PlatformAdapter trait, SessionSource routing |
| `agent-cli` | `cli` | CliRenderer, SlashCommandHandler traits |

## Design Principles

1. **Trait-based polymorphism** — All major components (`Tool`, `LlmClient`, `MemoryProvider`, `PlatformAdapter`) are traits, enabling plugin systems and provider swapping without recompilation.

2. **Permission levels as an ordering** — `ReadOnly < WorkspaceWrite < DangerFullAccess`. The ordering is crate-defined; the priority semantics for Deny/Ask/Allow rules belong to your binary's policy configuration.

3. **Async everywhere in the execution path** — Tools, LLM calls, MCP operations, hooks, memory providers, and session persistence are all async. Blocking operations must use `tokio::task::spawn_blocking`.

4. **Tool name normalization** — The normalization algorithm (lowercase, strip underscores/dashes, alias expansion) is defined here. The concrete alias registry belongs to your binary.

5. **Incremental persistence** — The `SessionStore` trait supports append-only logs. Whether you use JSONL, SQLite, or another backend is your choice; rotation thresholds and atomic write strategy are binary-level decisions.

6. **Graceful degradation cascade** — When failures occur, the agent degrades in priority order: MCP degraded continues with available tools; permission denied prompts or errors; rate-limited providers retry with backoff then fall through the chain; exhausted providers surface a final error.

7. **Fenced context** — Memory providers produce content wrapped in `<memory-context>` tags. The format is defined here; the provider implementation is your choice.

8. **Iteration budgets** — The `IterationBudget` struct tracks remaining turns. Default values for parent and subagent budgets belong to your binary configuration.

## Turn Loop Termination

A single `run_turn` call ends when one of these conditions is reached:

| Condition | Result |
|-----------|--------|
| LLM returns text with no tool calls | `Event::MessageStop` with content; message appended to session |
| LLM returns tool calls | `Event::ToolCallComplete` for each; tool calls appended, results awaited |
| Iteration budget exhausted | `Event::Error(AgentError::Internal("budget exhausted"))` |
| Hook denies tool | `Event::ToolDenied`; tool not executed |
| Permission denied + user declines | `Event::PermissionDenied`; tool not executed |
| Retryable provider error | Retry within budget, then fall through provider chain |
| Non-retryable provider error | `Event::Error` propagated to caller |
| `interrupt()` called | `Event::Interrupted`; session preserved |

The caller is responsible for displaying events and deciding whether to call `run_turn` again. The agent loop is the caller's loop, not the crate's.

## Async Runtime

This crate requires **tokio** as the async runtime. The requirements are:

- **Persistent event loops** — Use a single `tokio::runtime::Runtime` per thread, not per-task.
- **Thread pool for sync tools** — Use `tokio::task::spawn_blocking` for CPU-bound or blocking sync tools.
- **Async tool dispatch** — All tool execution goes through `async fn`.
- **No `!Send` futures** — The registry handles both `Send` and `!Send` tools via spawn patterns.

## Error Types

The crate defines a hierarchy of error types:

- `ApiError` — LLM provider errors (missing credentials, context exceeded, rate limiting, etc.)
- `ToolError` — Tool not found, execution failure, timeout, result size exceeded
- `SessionError` — IO failure, corrupt data, format errors
- `McpError` — Server exit, transport errors, protocol errors, timeout, auth
- `AgentError` — Encompasses all of the above via variants

`ApiError::is_retryable()` identifies errors suitable for retry: HTTP 429/500-503, rate limiting, IO, and retries-exhausted. Non-retryable errors (auth failures, context length exceeded) terminate the fallback chain immediately.

## License

MIT
