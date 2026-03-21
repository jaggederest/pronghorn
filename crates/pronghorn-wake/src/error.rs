use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WakeError {
    #[error("wake word backend initialization failed: {0}")]
    BackendInit(String),

    #[error("failed to load wake word model from {path}: {reason}")]
    ModelLoad { path: PathBuf, reason: String },

    #[error("audio format mismatch: expected {expected}, got {actual}")]
    FormatMismatch { expected: String, actual: String },

    #[error("no wake words configured")]
    NoWakeWords,

    #[error("backend '{0}' is not available (feature not enabled)")]
    BackendNotAvailable(String),
}
