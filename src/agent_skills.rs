//! Agent skills module.
//!
//! Skills management for discovering and loading agent capabilities from
//! external skill sources. The `SkillSource` trait provides search and
//! fetch operations, while `SkillsManager` coordinates across multiple sources.
//!
//! This module is feature-gated behind `feature = "skills"`.
//!

use std::fmt;
use std::sync::Arc;

/// A callable skill with a name, description, and handler function.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Unique name identifying the skill.
    pub name: String,
    /// Human-readable description of what the skill does.
    pub description: String,
    /// The callable handler for this skill.
    pub handler: Arc<dyn SkillHandler>,
}

/// Trait for skill handler functions.
///
///Handlers receive a JSON value as input and return a JSON value.
pub trait SkillHandler: Send + Sync + fmt::Debug {
    /// Invokes the skill with the given JSON input.
    fn invoke(&self, input: serde_json::Value) -> Result<serde_json::Value, String>;
}

impl<F> SkillHandler for F
where
    F: Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync + fmt::Debug,
{
    fn invoke(&self, input: serde_json::Value) -> Result<serde_json::Value, String> {
        self(input)
    }
}

/// Error returned when a skill cannot be found.
#[derive(Debug, Clone)]
pub struct SkillNotFoundError(pub String);

impl fmt::Display for SkillNotFoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "skill not found: {}", self.0)
    }
}

impl std::error::Error for SkillNotFoundError {}

/// Manages registration and invocation of skills.
pub trait SkillRegistry: Send + Sync {
    /// Registers a new skill, replacing any existing skill with the same name.
    fn register(&self, skill: Skill) -> Result<(), String>;

    /// Invokes the named skill with the given input.
    fn invoke(&self, name: &str, input: serde_json::Value) -> Result<serde_json::Value, String>;

    /// Lists the names of all registered skills.
    fn list_skills(&self) -> Vec<String>;
}

/// In-memory implementation of `SkillRegistry`.
#[derive(Debug, Default)]
pub struct InMemorySkillRegistry {
    skills: std::sync::RwLock<std::collections::HashMap<String, Skill>>,
}

impl InMemorySkillRegistry {
    /// Creates a new empty in-memory skill registry.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SkillRegistry for InMemorySkillRegistry {
    fn register(&self, skill: Skill) -> Result<(), String> {
        let mut skills = self.skills.write().map_err(|e| e.to_string())?;
        skills.insert(skill.name.clone(), skill);
        Ok(())
    }

    fn invoke(&self, name: &str, input: serde_json::Value) -> Result<serde_json::Value, String> {
        let skills = self.skills.read().map_err(|e| e.to_string())?;
        skills
            .get(name)
            .ok_or_else(|| format!("skill not found: {}", name))
            .map(|skill| skill.handler.invoke(input))?
    }

    fn list_skills(&self) -> Vec<String> {
        let skills = self.skills.read().map_err(|e| e.to_string()).unwrap();
        skills.keys().cloned().collect()
    }
}
