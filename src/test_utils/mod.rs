//! Internal test utilities and mock implementations.
//!
//! This module provides mock implementations of core traits for testing
//! purposes. Previously this would have been a separate test-utils crate.
//!
//! # Example
//!
//! ```
//! use ai_agent::test_utils::MockLlmClient;
//! ```

use std::sync::RwLock;
use std::collections::HashMap;

// ============================================================================
// Mock LLM Client
// ============================================================================

/// Mock LLM client for testing.
pub struct MockLlmClient {
    pub model_name: String,
    pub responses: RwLock<Vec<String>>,
    pub call_count: RwLock<usize>,
}

impl MockLlmClient {
    pub fn new(model: &str) -> Self {
        Self {
            model_name: model.to_string(),
            responses: RwLock::new(Vec::new()),
            call_count: RwLock::new(0),
        }
    }

    pub fn add_response(&self, response: String) {
        self.responses.write().unwrap().push(response);
    }
}

// ============================================================================
// Mock Tool Executor
// ============================================================================

/// Mock tool executor for testing.
pub struct MockToolExecutor {
    pub tools: RwLock<HashMap<String, String>>,
    pub calls: RwLock<Vec<(String, String)>>,
}

impl Default for MockToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl MockToolExecutor {
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
            calls: RwLock::new(Vec::new()),
        }
    }

    pub fn register_tool(&self, name: &str, response: &str) {
        self.tools.write().unwrap().insert(name.to_string(), response.to_string());
    }
}

// ============================================================================
// Mock Session Store
// ============================================================================

/// Mock session store for testing.
pub struct MockSessionStore {
    pub sessions: RwLock<HashMap<String, String>>,
    pub saves: RwLock<Vec<String>>,
}

impl Default for MockSessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MockSessionStore {
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            saves: RwLock::new(Vec::new()),
        }
    }
}

// ============================================================================
// Mock MCP Client
// ============================================================================

/// Mock MCP client for testing.
pub struct MockMcpClient {
    pub tools: RwLock<Vec<String>>,
    pub connected: RwLock<bool>,
}

impl Default for MockMcpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl MockMcpClient {
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(Vec::new()),
            connected: RwLock::new(false),
        }
    }
}

// ============================================================================
// Mock Hook Runner
// ============================================================================

/// Mock hook runner for testing.
pub struct MockHookRunner {
    pub hooks_called: RwLock<Vec<String>>,
}

impl Default for MockHookRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl MockHookRunner {
    pub fn new() -> Self {
        Self {
            hooks_called: RwLock::new(Vec::new()),
        }
    }
}

// ============================================================================
// Mock Permission Policy
// ============================================================================

/// Mock permission policy for testing.
pub struct MockPermissionPolicy {
    pub allow_all: bool,
}

impl MockPermissionPolicy {
    pub fn allow_all() -> Self {
        Self { allow_all: true }
    }

    pub fn deny_all() -> Self {
        Self { allow_all: false }
    }
}

// ============================================================================
// Mock Plugin
// ============================================================================

/// Mock plugin for testing.
pub struct MockPlugin {
    pub id: String,
    pub initialized: RwLock<bool>,
}

impl MockPlugin {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            initialized: RwLock::new(false),
        }
    }
}
