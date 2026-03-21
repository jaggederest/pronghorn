use serde::{Deserialize, Serialize};

/// Aggregated pipeline configuration — lives under `[pipeline]` in server.toml.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelineConfig {
    pub stt: SttConfig,
    pub tts: TtsConfig,
    pub intent: IntentConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    pub backend: SttBackend,
    /// URL for the STT service (e.g., "ws://localhost:9090").
    pub url: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SttBackend {
    #[default]
    Echo,
    FasterWhisper,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    pub backend: TtsBackend,
    /// URL for the TTS service (e.g., "http://localhost:5500").
    pub url: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TtsBackend {
    #[default]
    Echo,
    Piper,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct IntentConfig {
    pub backend: IntentBackend,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IntentBackend {
    #[default]
    Echo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips_through_toml() {
        let config = PipelineConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: PipelineConfig = toml::from_str(&toml_str).unwrap();
        assert!(matches!(parsed.stt.backend, SttBackend::Echo));
        assert!(matches!(parsed.tts.backend, TtsBackend::Echo));
        assert!(matches!(parsed.intent.backend, IntentBackend::Echo));
    }

    #[test]
    fn partial_toml_uses_defaults() {
        let toml_str = r#"
[stt]
backend = "faster-whisper"
url = "ws://localhost:9090"
"#;
        let parsed: PipelineConfig = toml::from_str(toml_str).unwrap();
        assert!(matches!(parsed.stt.backend, SttBackend::FasterWhisper));
        assert_eq!(parsed.stt.url, "ws://localhost:9090");
        // TTS and intent defaulted
        assert!(matches!(parsed.tts.backend, TtsBackend::Echo));
        assert!(matches!(parsed.intent.backend, IntentBackend::Echo));
    }
}
