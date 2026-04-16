//! # ai-agent
//!
//! A general-purpose, trait-based toolkit for building AI agents in Rust. The
//! crate provides the foundational traits and runtime infrastructure that binary
//! authors implement to create customized agent applications. Every major
//! component is expressed as a trait object, allowing LLM providers, tool
//! registries, session stores, permission policies, and memory systems to be
//! swapped without modifying the crate itself. The crate makes no assumptions
//! about which providers are available, how sessions are persisted, or what
//! default iteration budgets apply -- all of those decisions belong to the
//! binary layer. Hooks, plugins, and MCP clients integrate through well-defined
//! extension points. The architecture is async-first throughout the execution
//! path, requiring a `tokio` runtime in any binary that uses this crate. All
//! composition points are trait objects, enabling testability through mock
//! implementations and flexibility for any deployment context.
//!
//! ## Design Philosophy
//!
//! - **Trait-based polymorphism over inheritance** -- `Tool`, `LlmClient`,
//!   `MemoryProvider`, and `PlatformAdapter` are all trait objects, enabling
//!   plugin systems and provider swapping without recompilation.
//!
//! - **Permission levels as an ordering** (`ReadOnly < WorkspaceWrite <
//!   DangerFullAccess`) -- the ordering is crate-defined; specific priority
//!   semantics (Deny > Ask > Allow) belong to the binary layer.
//!
//! - **Async everywhere in the execution path** -- tools, LLM calls, MCP,
//!   hooks, memory, and persistence are all non-blocking.
//!
//! - **Graceful degradation cascade** -- MCP degraded continues with available
//!   tools; permission denied prompts or errors; rate-limited providers retry
//!   then fall through the chain; exhausted providers surface a final error.
//!
//! - **Incremental persistence** -- the `SessionStore` trait supports append-only
//!   logs; the binary chooses JSONL vs SQLite and configures rotation thresholds.
//!
//! - **Fenced context** -- memory providers produce content wrapped in
//!   `<memory-context>` tags; the format is crate-defined; the provider
//!   implementation is binary-chosen.
//!
//! - **Iteration budgets** -- the `IterationBudget` struct is crate-defined;
//!   default values (parent 90, subagent 50) belong to the binary layer.
//!
//! ## Module Hierarchy
//!
//! The crate is organized into three tiers, radiating outward from `agent-core`:
//!
//! ```text
//! agent-core (required, center)
//! ├── agent-provider (required)       -- LLM client abstraction
//! ├── agent-tools (required)          -- Tool registry and execution
//! ├── agent-session (required)         -- Session management
//! ├── agent-mcp (required)             -- MCP client support
//! ├── agent-hooks (required)           -- Hook system
//! ├── agent-permissions (required)     -- Permission evaluation
//! ├── agent-plugin (required)         -- Plugin system
//! ├── agent-config (required)          -- Configuration loading
//! ├── agent-prompt (should)            -- System prompt builder
//! ├── agent-context (should)          -- Context compression
//! ├── agent-telemetry (should)         -- Telemetry sink
//! ├── agent-memory (nice-to-have)      -- Memory providers
//! └── agent-skills (nice-to-have)      -- Skills management
//! ```
//!
//! ## Feature Flags
//!
//! | Flag | Module | Description |
//! |------|--------|-------------|
//! | `prompt` | `agent_prompt` | System prompt building and context file loading |
//! | `context` | `agent_context` | Context compression with `ContextCompressor` trait |
//! | `telemetry` | `agent_telemetry` | Event recording via `TelemetrySink` trait |
//! | `memory` | `agent_memory` | Memory provider system for persistent context |
//! | `skills` | `agent_skills` | Skills hub with `SkillSource` trait |
//!
//! ## Configuration Cascade
//!
//! Configuration is loaded in priority order (highest to lowest):
//!
//! 1. **Environment variables** -- `AGENT_*` prefixed vars override all files
//! 2. **`{cwd}/.agent/settings.local.json`** -- project-local, never committed
//! 3. **`{cwd}/.agent/settings.json`** -- project-configured
//! 4. **`~/.agent/settings.json`** -- user defaults (or `$AGENT_HOME`)
//! 5. **Compiled-in defaults** -- used when no config file exists
//!
//! MCP server definitions are deep-merged across files.
//!
//! ## Async Runtime
//!
//! This crate requires `tokio` and uses it internally. Binary authors must:
//!
//! - Use a single `tokio::runtime::Runtime` per thread, not per-task
//! - Use `tokio::task::spawn_blocking` for CPU-bound or blocking sync tools
//! - Ensure no `!Send` futures in the execution path
//!
//! ## Module Reference
//!
//! ### Core Modules (always compiled)
//!
//! - `agent_core` -- Agent trait and context
//! - `agent_provider` -- LLM client abstraction
//! - `agent_tools` -- Tool registry and execution
//! - `agent_session` -- Session management
//! - `agent_permissions` -- Permission evaluation
//! - `agent_hooks` -- Hook system
//! - `agent_mcp` -- MCP client support
//! - `agent_plugin` -- Plugin system
//! - `agent_config` -- Configuration loading
//!
//! ### Feature-Gated Modules
//!
//! - `agent_prompt` -- System prompt building (`feature = "prompt"`)
//! - `agent_context` -- Context compression (`feature = "context"`)
//! - `agent_telemetry` -- Telemetry (`feature = "telemetry"`)
//! - `agent_memory` -- Memory providers (`feature = "memory"`)
//! - `agent_skills` -- Skills management (`feature = "skills"`)
//!
//! ### Supporting Modules
//!
//! - `agent_error` -- Error types
//! - `agent_types` -- Shared types
//!
//! ## Essential vs Nice-to-Have
//!
//! Essential (required): Agent trait, LLM provider abstraction, tool registry,
//! session management, MCP client, hook system, permission policy, plugin
//! system, and config loading.
//!
//! Should have: system prompt builder, telemetry sink, and context compression.
//!
//! Nice-to-have: memory providers, skills hub, messaging gateway, and CLI/REPL
//! surface traits.

#![allow(dead_code)]
//!
//! All public items in this library crate are intentionally exposed for consumer use.
//! dead_code warnings are suppressed since this is a trait library — types are defined
//! here but constructed and used by downstream binary crates.

pub mod prelude;

pub mod agent_core;
pub mod agent_provider;
pub mod agent_tools;
pub mod agent_session;
pub mod agent_permissions;
pub mod agent_hooks;
pub mod agent_mcp;
pub mod agent_plugin;
pub mod agent_config;

#[cfg(feature = "prompt")]
pub mod agent_prompt;

#[cfg(feature = "context")]
pub mod agent_context;

#[cfg(feature = "telemetry")]
pub mod agent_telemetry;

#[cfg(feature = "memory")]
pub mod agent_memory;

#[cfg(feature = "skills")]
pub mod agent_skills;

pub mod agent_error;
pub mod agent_types;

pub mod test_utils;
