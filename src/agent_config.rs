//! Agent config module.
//!
//! Configuration loading for the agent. This module provides the `AgentConfig`
//! struct, which aggregates all runtime configuration (model, tools,
//! permissions, sandbox, MCP servers, hooks, plugins, provider fallbacks,
//! trusted roots), and the `ConfigLoader` trait for loading this
//! configuration from a directory. The cascade precedence (env vars over
//! file-based settings) is binary-level policy.
//!

use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::agent_permissions::PermissionMode;
use crate::agent_provider::ProviderKind;

// ============================================================================
// ConfigError
// ============================================================================

/// Errors that can occur when loading or parsing configuration.
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Failed to parse the configuration file.
    Parse(String),
    /// The file format is not supported by any available loader.
    FormatNotSupported(String),
    /// The configuration file or directory was not found.
    NotFound(String),
    /// An I/O error occurred while reading the file.
    IO(String),
    /// The configuration failed validation.
    Validation(String),
    /// An internal error occurred.
    Internal(String),
}

impl ConfigError {
    /// Returns `true` if this error is retryable.
    ///
    /// Only I/O and internal errors are considered retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(self, ConfigError::IO(_) | ConfigError::Internal(_))
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Parse(msg) => write!(f, "parse error: {}", msg),
            ConfigError::FormatNotSupported(ext) => write!(f, "unsupported format: {}", ext),
            ConfigError::NotFound(msg) => write!(f, "not found: {}", msg),
            ConfigError::IO(msg) => write!(f, "I/O error: {}", msg),
            ConfigError::Validation(msg) => write!(f, "validation error: {}", msg),
            ConfigError::Internal(msg) => write!(f, "internal error: {}", msg),
        }
    }
}

impl std::error::Error for ConfigError {}

// ============================================================================
// ProviderConfig
// ============================================================================

/// Configuration for the LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// The kind of provider (Anthropic, OpenAI, etc.).
    pub kind: ProviderKind,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
    /// Optional base URL for custom endpoints.
    pub base_url: Option<String>,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

// ============================================================================
// PermissionsConfig
// ============================================================================

/// Configuration for permission policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionPolicyConfig {
    /// Glob pattern for matching tool names.
    pub pattern: String,
    /// Permission mode for matching tools.
    pub mode: PermissionMode,
}

/// Configuration for permissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionsConfig {
    /// Default permission mode when no policy matches.
    pub default_mode: PermissionMode,
    /// Ordered list of permission policies (first match wins).
    pub policies: Vec<PermissionPolicyConfig>,
}

// ============================================================================
// HooksConfig
// ============================================================================

/// Configuration for hooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HooksConfig {
    /// Whether hooks are enabled.
    pub enabled: bool,
    /// List of hook names to invoke.
    pub hook_names: Vec<String>,
}

// ============================================================================
// SessionConfig
// ============================================================================

/// Configuration for session handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Storage format for session data (opaque string, format-specific).
    pub storage_format: String,
    /// Maximum number of history entries to retain.
    pub max_history: usize,
}

// ============================================================================
// McpConfig
// ============================================================================

/// Configuration for an MCP (Model Context Protocol) server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// Transport type (e.g., "stdio", "http").
    pub transport: String,
    /// Optional command to launch the MCP server.
    pub command: Option<String>,
    /// Environment variables to pass to the MCP server.
    pub env: HashMap<String, String>,
}

// ============================================================================
// PluginConfig
// ============================================================================

/// Configuration for a plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin name.
    pub name: String,
    /// Path to the plugin binary or library.
    pub path: String,
    /// Plugin-specific configuration key-value pairs.
    pub config: HashMap<String, String>,
}

// ============================================================================
// AgentConfig
// ============================================================================

/// Aggregated runtime configuration for the agent.
///
/// This struct collects all subsystem configurations:
/// provider, permissions, hooks, session, MCP, plugins, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// LLM provider configuration.
    pub provider: ProviderConfig,
    /// Permissions configuration.
    pub permissions: PermissionsConfig,
    /// Hooks configuration.
    pub hooks: HooksConfig,
    /// Session configuration.
    pub session: SessionConfig,
    /// MCP server configurations.
    pub mcp: McpConfig,
    /// Plugin configurations.
    pub plugins: Vec<PluginConfig>,
    /// Arbitrary metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

// ============================================================================
// ConfigLoader
// ============================================================================

/// Trait for loading configuration from a working directory.
///
/// Implementors can parse configuration from a specific format (JSON, TOML,
/// YAML, etc.). Use [`ConfigLoaderBuilder`] to compose multiple loaders into
/// a registry that picks the first supported format.
#[async_trait::async_trait]
pub trait ConfigLoader: Send + Sync {
    /// Load configuration from the given working directory.
    ///
    /// The loader should look for a config file (e.g., `agent.json`,
    /// `agent.toml`) in `cwd` and parse it.
    async fn load(&self, cwd: &Path) -> Result<AgentConfig, ConfigError>;

    /// Returns `true` if this loader supports the given file extension.
    fn supports_format(&self, ext: &str) -> bool;
}

// ============================================================================
// JsonConfigLoader
// ============================================================================

/// Loader for JSON configuration files.
#[derive(Debug, Clone)]
pub struct JsonConfigLoader {
    filename: String,
}

impl JsonConfigLoader {
    /// Creates a new `JsonConfigLoader` that looks for `filename` in config dirs.
    pub fn new(filename: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
        }
    }
}

#[async_trait::async_trait]
impl ConfigLoader for JsonConfigLoader {
    async fn load(&self, cwd: &Path) -> Result<AgentConfig, ConfigError> {
        let path = cwd.join(&self.filename);
        let content =
            tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| match e.kind() {
                    std::io::ErrorKind::NotFound => ConfigError::NotFound(path.display().to_string()),
                    _ => ConfigError::IO(e.to_string()),
                })?;

        serde_json::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))
    }

    fn supports_format(&self, ext: &str) -> bool {
        ext.eq_ignore_ascii_case("json")
    }
}

// ============================================================================
// TomlConfigLoader
// ============================================================================

/// Loader for TOML configuration files.
#[derive(Debug, Clone)]
pub struct TomlConfigLoader {
    filename: String,
}

impl TomlConfigLoader {
    /// Creates a new `TomlConfigLoader` that looks for `filename` in config dirs.
    pub fn new(filename: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
        }
    }
}

#[async_trait::async_trait]
impl ConfigLoader for TomlConfigLoader {
    async fn load(&self, cwd: &Path) -> Result<AgentConfig, ConfigError> {
        let path = cwd.join(&self.filename);
        let content =
            tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| match e.kind() {
                    std::io::ErrorKind::NotFound => ConfigError::NotFound(path.display().to_string()),
                    _ => ConfigError::IO(e.to_string()),
                })?;

        toml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))
    }

    fn supports_format(&self, ext: &str) -> bool {
        ext.eq_ignore_ascii_case("toml")
    }
}

// ============================================================================
// YamlConfigLoader
// ============================================================================

/// Loader for YAML configuration files.
#[derive(Debug, Clone)]
pub struct YamlConfigLoader {
    filename: String,
}

impl YamlConfigLoader {
    /// Creates a new `YamlConfigLoader` that looks for `filename` in config dirs.
    pub fn new(filename: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
        }
    }
}

#[async_trait::async_trait]
impl ConfigLoader for YamlConfigLoader {
    async fn load(&self, cwd: &Path) -> Result<AgentConfig, ConfigError> {
        let path = cwd.join(&self.filename);
        let content =
            tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| match e.kind() {
                    std::io::ErrorKind::NotFound => ConfigError::NotFound(path.display().to_string()),
                    _ => ConfigError::IO(e.to_string()),
                })?;

        serde_yaml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))
    }

    fn supports_format(&self, ext: &str) -> bool {
        ext.eq_ignore_ascii_case("yaml") || ext.eq_ignore_ascii_case("yml")
    }
}

// ============================================================================
// ConfigLoaderBuilder
// ============================================================================

/// Builder for composing multiple [`ConfigLoader`] implementations.
///
/// # Example
///
/// ```
/// use ai_agent::{ConfigLoaderBuilder, JsonConfigLoader, TomlConfigLoader, YamlConfigLoader};
///
/// let loader = ConfigLoaderBuilder::new()
///     .with_loader(JsonConfigLoader::new("agent.json"))
///     .with_loader(TomlConfigLoader::new("agent.toml"))
///     .with_loader(YamlConfigLoader::new("agent.yaml"))
///     .build();
/// ```
pub struct ConfigLoaderBuilder {
    loaders: Vec<Box<dyn ConfigLoader>>,
}

impl Debug for ConfigLoaderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigLoaderBuilder")
            .field("loaders", &self.loaders.len())
            .finish()
    }
}

impl ConfigLoaderBuilder {
    /// Creates a new empty builder.
    pub fn new() -> Self {
        Self { loaders: Vec::new() }
    }

    /// Adds a loader to the registry.
    pub fn with_loader(mut self, loader: impl ConfigLoader + 'static) -> Self {
        self.loaders.push(Box::new(loader));
        self
    }

    /// Builds a [`ConfigLoaderRegistry`] from the registered loaders.
    pub fn build(self) -> ConfigLoaderRegistry {
        ConfigLoaderRegistry {
            loaders: self.loaders,
        }
    }
}

impl Default for ConfigLoaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ConfigLoaderRegistry
// ============================================================================

/// A registry of [`ConfigLoader`] implementations that tries each loader
/// in order until one succeeds.
///
/// The registry selects a loader based on file extension. If no loader
/// supports the requested format, [`ConfigError::FormatNotSupported`] is
/// returned.
pub struct ConfigLoaderRegistry {
    loaders: Vec<Box<dyn ConfigLoader>>,
}

impl Debug for ConfigLoaderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigLoaderRegistry")
            .field("loaders", &self.loaders.len())
            .finish()
    }
}


impl ConfigLoaderRegistry {
    /// Attempts to load configuration from `cwd` using the first available
    /// loader that supports a file found in `cwd`.
    ///
    /// It scans for files matching the registered loaders' filenames and
    /// attempts to load with the matching loader.
    pub async fn load(&self, cwd: &Path) -> Result<AgentConfig, ConfigError> {
        if self.loaders.is_empty() {
            return Err(ConfigError::Internal("no loaders registered".into()));
        }

        // Try each loader in order; first successful parse wins.
        for loader in &self.loaders {
            // We need to try loading; if the file doesn't exist, skip to next.
            // A more sophisticated implementation could check file existence first.
            match loader.load(cwd).await {
                Ok(config) => return Ok(config),
                Err(ConfigError::NotFound(_)) => continue,
                Err(e) => return Err(e),
            }
        }

        Err(ConfigError::NotFound(
            "no configuration file found for any registered loader".into(),
        ))
    }

    /// Returns `true` if any registered loader supports the given extension.
    pub fn supports_format(&self, ext: &str) -> bool {
        self.loaders.iter().any(|l| l.supports_format(ext))
    }
}

#[async_trait::async_trait]
impl ConfigLoader for ConfigLoaderRegistry {
    async fn load(&self, cwd: &Path) -> Result<AgentConfig, ConfigError> {
        ConfigLoaderRegistry::load(self, cwd).await
    }

    fn supports_format(&self, ext: &str) -> bool {
        ConfigLoaderRegistry::supports_format(self, ext)
    }
}
