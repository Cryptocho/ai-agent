//! Agent prompt module.
//!
//! System prompt building and context file injection. This module provides
//! `SystemPromptBuilder` for composing prompts from named sections
//! (static and dynamic), and the `ContextFileLoader` trait for discovering
//! and scanning context files for injection risk.
//!
//! This module is feature-gated behind `feature = "prompt"`.
//!

use std::collections::HashMap;

/// A prompt template with a template string and its variable names.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// The template string with variable placeholders (e.g. `"Hello, {name}!"`).
    pub template: String,
    /// Ordered list of variable names referenced in the template.
    pub variables: Vec<String>,
}

impl PromptTemplate {
    /// Creates a new `PromptTemplate` with the given template string and variables.
    pub fn new(template: String, variables: Vec<String>) -> Self {
        Self { template, variables }
    }
}

/// Renders prompt templates by substituting variables.
pub trait PromptRenderer: Send + Sync {
    /// Renders the given template by substituting each variable with the corresponding value.
    fn render(&self, template: &PromptTemplate, vars: HashMap<String, String>) -> String;
}

/// Renders templates by performing placeholder substitution on a template string.
#[derive(Debug, Clone, Default)]
pub struct SimpleRenderer;

impl PromptRenderer for SimpleRenderer {
    fn render(&self, template: &PromptTemplate, vars: HashMap<String, String>) -> String {
        let mut result = template.template.clone();
        for var in &template.variables {
            let placeholder = format!("{{{var}}}");
            if let Some(value) = vars.get(var) {
                result = result.replace(&placeholder, value);
            }
        }
        result
    }
}
