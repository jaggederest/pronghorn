use thiserror::Error;

#[derive(Debug, Error)]
pub enum SatelliteError {
    #[error(transparent)]
    Wire(#[from] pronghorn_wire::WireError),

    #[error(transparent)]
    Wake(#[from] pronghorn_wake::WakeError),

    #[error(transparent)]
    Config(#[from] crate::config::ConfigError),

    #[error("server handshake failed: expected Welcome packet")]
    HandshakeFailed,

    #[error("server did not respond within timeout")]
    ServerTimeout,

    #[error("failed to spawn wake word thread: {0}")]
    WakeThread(#[from] std::io::Error),
}
