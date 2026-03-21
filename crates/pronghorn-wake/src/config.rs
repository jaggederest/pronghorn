use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Top-level wake word configuration — lives under `[wake]` in pronghorn.toml.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WakeConfig {
    /// Master switch. Defaults to false so the system works without model files.
    pub enabled: bool,
    /// Which backend to use.
    pub backend: WakeBackend,
    /// Rustpotter-specific settings.
    pub rustpotter: RustpotterBackendConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WakeBackend {
    #[default]
    Rustpotter,
}

/// Configuration for the Rustpotter wake word backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RustpotterBackendConfig {
    /// Map of wake word name → path to .rpw file (reference or model).
    pub wake_words: BTreeMap<String, PathBuf>,
    /// Score threshold against individual template frames (0.0–1.0).
    pub threshold: f32,
    /// Score threshold against averaged template (0.0–1.0).
    pub avg_threshold: f32,
    /// Emit detection as soon as min_scores partial detections are reached.
    pub eager: bool,
    /// Minimum positive scores before detection fires.
    pub min_scores: usize,
}

impl Default for RustpotterBackendConfig {
    fn default() -> Self {
        Self {
            wake_words: BTreeMap::new(),
            threshold: 0.5,
            avg_threshold: 0.2,
            eager: false,
            min_scores: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips_through_toml() {
        let config = WakeConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: WakeConfig = toml::from_str(&toml_str).unwrap();
        assert!(!parsed.enabled);
        assert_eq!(parsed.rustpotter.threshold, 0.5);
    }

    #[test]
    fn partial_toml_uses_defaults() {
        let toml_str = r#"
enabled = true
backend = "rustpotter"

[rustpotter]
threshold = 0.7

[rustpotter.wake_words]
hey_pronghorn = "models/hey_pronghorn.rpw"
"#;
        let parsed: WakeConfig = toml::from_str(toml_str).unwrap();
        assert!(parsed.enabled);
        assert_eq!(parsed.rustpotter.threshold, 0.7);
        assert_eq!(parsed.rustpotter.avg_threshold, 0.2); // defaulted
        assert_eq!(parsed.rustpotter.min_scores, 5); // defaulted
        assert!(parsed.rustpotter.wake_words.contains_key("hey_pronghorn"));
    }

    #[test]
    fn disabled_by_default() {
        let parsed: WakeConfig = toml::from_str("").unwrap();
        assert!(!parsed.enabled);
    }
}
