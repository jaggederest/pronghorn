use std::net::SocketAddr;
use std::path::Path;

use pronghorn_audio::AudioConfig;
use pronghorn_wake::WakeConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SatelliteConfig {
    pub audio: AudioConfig,
    pub wake: WakeConfig,
    pub transport: SatelliteTransportConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SatelliteTransportConfig {
    /// Address of the Pronghorn server to connect to.
    pub server_address: SocketAddr,
    /// Keepalive interval in milliseconds.
    pub keepalive_interval_ms: u64,
}

impl Default for SatelliteTransportConfig {
    fn default() -> Self {
        Self {
            server_address: ([127, 0, 0, 1], 9999).into(),
            keepalive_interval_ms: 5_000,
        }
    }
}

impl SatelliteConfig {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        Ok(config)
    }

    pub fn load_or_default(path: &Path) -> Result<Self, ConfigError> {
        if path.exists() {
            Self::load(path)
        } else {
            Ok(Self::default())
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config: {0}")]
    Parse(#[from] toml::de::Error),
}
