//! Agent plugin module.
//!
//! Plugin system for extending the agent at runtime. Plugins declare their
//! capabilities via a `PluginManifest` (tools, hooks, lifecycle handlers,
//! permissions) and implement the `Plugin` trait for initialization and
//! shutdown. The `PluginManager` ties together a tool registry and hook
//! runner with a collection of loaded plugins.
//!

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

/// Error types for plugin operations.
#[derive(Debug, Clone)]
pub enum PluginError {
    /// Plugin initialization failed.
    Initialization(String),
    /// Plugin shutdown failed.
    Shutdown(String),
    /// Hook execution failed.
    Hook(String),
    /// Plugin configuration error.
    Config(String),
    /// Internal plugin error.
    Internal(String),
}

impl PluginError {
    /// Returns true if the error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            PluginError::Initialization(_)
                | PluginError::Hook(_)
                | PluginError::Config(_)
        )
    }
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginError::Initialization(msg) => write!(f, "initialization error: {}", msg),
            PluginError::Shutdown(msg) => write!(f, "shutdown error: {}", msg),
            PluginError::Hook(msg) => write!(f, "hook error: {}", msg),
            PluginError::Config(msg) => write!(f, "config error: {}", msg),
            PluginError::Internal(msg) => write!(f, "internal error: {}", msg),
        }
    }
}

impl std::error::Error for PluginError {}

/// Manifest describing a plugin's identity and capabilities.
#[derive(Debug, Clone)]
pub struct PluginManifest {
    /// Human-readable name.
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// Short description of the plugin.
    pub description: String,
    /// List of hook names the plugin registers.
    pub hooks: Vec<String>,
    /// Plugin dependencies by name.
    pub dependencies: Vec<String>,
}

/// Core trait for plugin implementations.
#[async_trait::async_trait]
pub trait Plugin: Send + Sync {
    /// Initialize the plugin with the given manifest.
    async fn initialize(&self, manifest: &PluginManifest) -> Result<(), PluginError>;

    /// Shut down the plugin gracefully.
    async fn shutdown(&self) -> Result<(), PluginError>;

    /// Return the plugin's name.
    fn name(&self) -> String;

    /// Return the plugin's version string.
    fn version(&self) -> String;
}

/// Manages a collection of loaded plugins.
#[derive(Default)]
pub struct PluginManager {
    plugins: HashMap<String, Arc<dyn Plugin>>,
}

impl Debug for PluginManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginManager")
            .field("plugins", &self.plugins.len())
            .finish()
    }
}

impl PluginManager {
    /// Create a new empty plugin manager.
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register a plugin. The plugin's name (from `Plugin::name()`) is used as the key.
    pub fn register(&mut self, plugin: Arc<dyn Plugin>) -> Result<(), PluginError> {
        let name = plugin.name();
        if self.plugins.contains_key(&name) {
            return Err(PluginError::Initialization(format!(
                "plugin '{}' is already registered",
                name
            )));
        }
        self.plugins.insert(name, plugin);
        Ok(())
    }

    /// Unregister a plugin by name.
    pub fn unregister(&mut self, name: &str) -> Result<(), PluginError> {
        self.plugins
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| PluginError::Internal(format!("plugin '{}' not found", name)))
    }

    /// List all registered plugins as (name, version) pairs.
    pub fn list_plugins(&self) -> Vec<(String, String)> {
        self.plugins
            .values()
            .map(|p| (p.name(), p.version()))
            .collect()
    }
}
