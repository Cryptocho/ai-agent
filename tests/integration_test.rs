//! Integration tests for ai-agent.
//!
//! These tests verify the integration between core components.

use ai_agent::test_utils::{MockLlmClient, MockToolExecutor};

/// Basic smoke test to ensure the crate integrates correctly.
#[test]
fn test_smoke() {
    // Placeholder: real tests will be added in future phases
    std::hint::black_box(());
}

/// Test that mock implementations can be created.
#[test]
fn test_mock_llm_client() {
    let client = MockLlmClient::new("claude-3-opus");
    assert_eq!(client.model_name, "claude-3-opus");
}

/// Test that mock tool executor can be created and tools registered.
#[test]
fn test_mock_tool_executor() {
    let executor = MockToolExecutor::new();
    executor.register_tool("test_tool", "test_response");
    let tools = executor.tools.read().unwrap();
    assert!(tools.contains_key("test_tool"));
}