//! Agent memory module.
//!
//! Memory provider system for persisting and retrieving agent memory across
//! sessions. The `MemoryProvider` trait defines the interface for memory
//! backends, including lifecycle hooks (turn start, session end, pre-compress)
//! and tool exposure. The `MemoryManager` coordinates multiple providers.
//!
//! This module is feature-gated behind `feature = "memory"`.
//!

use std::error::Error as StdError;
use std::fmt;

/// Errors that can occur during memory operations.
#[derive(Debug, Clone)]
pub enum MemoryError {
    /// The requested key was not found in the store.
    NotFound(String),
    /// An I/O error occurred while accessing the store.
    Io(String),
    /// The store is not currently available.
    Unavailable(String),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::NotFound(key) => write!(f, "key not found: {}", key),
            MemoryError::Io(msg) => write!(f, "memory I/O error: {}", msg),
            MemoryError::Unavailable(msg) => write!(f, "memory unavailable: {}", msg),
        }
    }
}

impl StdError for MemoryError {}

/// Stores and retrieves binary memory blobs by key.
pub trait MemoryStore: Send + Sync {
    /// Stores a value under the given key.
    fn store(&self, key: &str, value: &[u8]) -> Result<(), MemoryError>;

    /// Retrieves the value stored under the given key.
    fn retrieve(&self, key: &str) -> Result<Vec<u8>, MemoryError>;

    /// Deletes the value stored under the given key.
    fn delete(&self, key: &str) -> Result<(), MemoryError>;

    /// Returns true if the store contains a value for the given key.
    fn exists(&self, key: &str) -> bool {
        self.retrieve(key).is_ok()
    }
}

/// In-memory implementation of `MemoryStore`.
#[derive(Debug, Clone, Default)]
pub struct InMemoryStore {
    data: std::sync::Arc<std::sync::RwLock<std::collections::HashMap<String, Vec<u8>>>>,
}

impl InMemoryStore {
    /// Creates a new empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl MemoryStore for InMemoryStore {
    fn store(&self, key: &str, value: &[u8]) -> Result<(), MemoryError> {
        let mut map = self.data.write().map_err(|e| MemoryError::Io(e.to_string()))?;
        map.insert(key.to_string(), value.to_vec());
        Ok(())
    }

    fn retrieve(&self, key: &str) -> Result<Vec<u8>, MemoryError> {
        let map = self.data.read().map_err(|e| MemoryError::Io(e.to_string()))?;
        map.get(key)
            .cloned()
            .ok_or_else(|| MemoryError::NotFound(key.to_string()))
    }

    fn delete(&self, key: &str) -> Result<(), MemoryError> {
        let mut map = self.data.write().map_err(|e| MemoryError::Io(e.to_string()))?;
        map.remove(key);
        Ok(())
    }
}
