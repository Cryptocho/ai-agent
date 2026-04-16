//! Agent permissions module.
//!
//! Permission evaluation engine. This module provides `PermissionPolicy`,
//! which evaluates whether a given tool call should be allowed based on
//! the current permission mode and a set of configurable rules. The module
//! also defines the `PermissionMode` enum that controls how permission
//! checks behave at runtime (Allow, Deny, Ask).
//!
//! The priority ordering of rule matches (Deny > Ask > Allow) is a hard
//! security constraint defined here. The specific rules and their
//! configuration belong to the binary-level policy.
//!

use std::pin::Pin;
use futures::future::Future;

/// Permission mode controlling how tool calls are handled.
///
/// The priority ordering is hardcoded: Deny > Ask > Allow.
/// This is a crate-level security constraint, not user-configurable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PermissionMode {
    /// Explicitly deny the tool call.
    Deny,
    /// Prompt the user for explicit allow/deny.
    Ask,
    /// Allow the tool call without prompting.
    Allow,
}

/// Hardcoded priority for permission modes.
///
/// Deny blocks everything, Ask requires explicit allow, Allow passes everything.
/// This constant is crate-defined and NOT configurable.
const PERMISSION_PRIORITY_DENY: i32 = 2;
const PERMISSION_PRIORITY_ASK: i32 = 1;
const PERMISSION_PRIORITY_ALLOW: i32 = 0;

/// Check if a permission mode allows an operation based on hardcoded priority.
///
/// Deny > Ask > Allow: if either mode is Deny, returns false.
/// If neither is Deny and at least one is Ask, requires explicit allow.
/// If both are Allow, returns true.
pub fn permission_mode_allows(mode: PermissionMode, required: PermissionMode) -> bool {
    let mode_rank = match mode {
        PermissionMode::Deny => PERMISSION_PRIORITY_DENY,
        PermissionMode::Ask => PERMISSION_PRIORITY_ASK,
        PermissionMode::Allow => PERMISSION_PRIORITY_ALLOW,
    };
    let required_rank = match required {
        PermissionMode::Deny => PERMISSION_PRIORITY_DENY,
        PermissionMode::Ask => PERMISSION_PRIORITY_ASK,
        PermissionMode::Allow => PERMISSION_PRIORITY_ALLOW,
    };
    mode_rank <= required_rank
}

/// Result of a permission policy evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionOutcome {
    /// The tool call is allowed to proceed.
    Allowed,
    /// The tool call was denied with a reason.
    Denied(String),
    /// The tool call requires user input to decide.
    Ask(String),
}

/// A pattern for matching tool names using glob matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionPattern {
    /// Glob pattern to match against tool names.
    pub tool_name_pattern: String,
}

impl PermissionPattern {
    /// Create a new permission pattern.
    pub fn new(tool_name_pattern: impl Into<String>) -> Self {
        Self {
            tool_name_pattern: tool_name_pattern.into(),
        }
    }

    /// Check if this pattern matches a given tool name.
    pub fn matches(&self, tool_name: &str) -> bool {
        glob_match(&self.tool_name_pattern, tool_name)
    }
}

/// Simple glob matching implementation.
/// Supports `*` (match any characters) and `?` (match single character).
fn glob_match(pattern: &str, name: &str) -> bool {
    let mut pattern_chars = pattern.chars().peekable();
    let mut name_chars = name.chars().peekable();

    while pattern_chars.peek().is_some() || name_chars.peek().is_some() {
        match pattern_chars.next() {
            None => return name_chars.next().is_none(),
            Some('*') => {
                // `*` matches zero or more characters
                // Collect rest of pattern after this `*`
                let rest_pattern: Vec<char> = pattern_chars.clone().collect();

                // Try matching rest of pattern at each position in remaining name
                let name_remaining: Vec<char> = name_chars.clone().collect();
                for i in 0..=name_remaining.len() {
                    let candidate: Vec<char> = name_remaining[i..].to_vec();
                    if glob_match_chars(&rest_pattern, &candidate) {
                        return true;
                    }
                }
                return false;
            }
            Some('?') => {
                if name_chars.next().is_none() {
                    return false;
                }
            }
            Some(c) => {
                if name_chars.next() != Some(c) {
                    return false;
                }
            }
        }
    }
    true
}

fn glob_match_chars(pattern: &[char], name: &[char]) -> bool {
    let mut pattern_chars = pattern.iter().copied().peekable();
    let mut name_chars = name.iter().copied().peekable();

    while pattern_chars.peek().is_some() || name_chars.peek().is_some() {
        match pattern_chars.next() {
            None => return name_chars.next().is_none(),
            Some('*') => {
                // Collect remaining pattern after this `*`
                let remaining_pattern: Vec<char> = pattern_chars.clone().collect();
                if remaining_pattern.is_empty() {
                    return true;
                }
                // Try matching remaining pattern at each position in remaining name
                for i in 0..=name_chars.clone().count() {
                    let rest: Vec<char> = name.iter().copied().skip(i).collect();
                    if glob_match_chars(&remaining_pattern, &rest) {
                        return true;
                    }
                }
                return false;
            }
            Some('?') => {
                if name_chars.next().is_none() {
                    return false;
                }
            }
            Some(c) => {
                if name_chars.next() != Some(c) {
                    return false;
                }
            }
        }
    }
    name_chars.next().is_none()
}

/// Policy trait for evaluating tool permission requests.
pub trait PermissionPolicy: Send + Sync {
    /// Evaluate whether a tool call should be allowed.
    ///
    /// `tool_name` is the name of the tool being invoked.
    /// `input` is the input to the tool.
    fn evaluate(&self, tool_name: &str, input: &str) -> PermissionOutcome;
}

/// Trait for prompting the user for permission decisions.
pub trait PermissionPrompter: Send + Sync {
    /// Prompt the user for a permission decision.
    ///
    /// `tool_name` is the name of the tool being invoked.
    /// `input` is the input to the tool.
    fn prompt(
        &self,
        tool_name: &str,
        input: &str,
    ) -> Pin<Box<dyn Future<Output = PermissionOutcome> + Send + '_>>;
}
