use std::path::Path;

use pronghorn_audio::AudioConfig;
use pronghorn_pipeline::PipelineConfig;
use pronghorn_wire::TransportConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub audio: AudioConfig,
    pub transport: TransportConfig,
    pub pipeline: PipelineConfig,
}

impl ServerConfig {
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
