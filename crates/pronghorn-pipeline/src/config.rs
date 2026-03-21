use std::path::PathBuf;

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
    /// Whisper-specific configuration.
    pub whisper: WhisperConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SttBackend {
    #[default]
    Echo,
    Whisper,
}

/// Configuration for the Whisper ONNX STT backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WhisperConfig {
    /// Directory containing Whisper model files:
    /// encoder_model.onnx, decoder_model_merged.onnx,
    /// multilingual.tiktoken, positional_embedding.npz, mel_filters.npz
    pub model_dir: PathBuf,
    /// Language code (e.g., "en").
    pub language: String,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("models/whisper-base"),
            language: "en".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    pub backend: TtsBackend,
    /// Kokoro-specific configuration.
    pub kokoro: KokoroConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TtsBackend {
    #[default]
    Echo,
    Kokoro,
}

/// Configuration for the Kokoro ONNX TTS backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KokoroConfig {
    /// Path to the Kokoro ONNX model file.
    pub model_path: PathBuf,
    /// Path to the tokenizer JSON file.
    pub tokenizer_path: PathBuf,
    /// Path to voice .bin files directory.
    pub voices_path: PathBuf,
    /// Voice ID (e.g., "af_heart").
    pub voice: String,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/kokoro/model.onnx"),
            tokenizer_path: PathBuf::from("models/kokoro/tokenizer.json"),
            voices_path: PathBuf::from("models/voices"),
            voice: "af_heart".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct IntentConfig {
    pub backend: IntentBackend,
    pub ollama: OllamaConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IntentBackend {
    #[default]
    Echo,
    Ollama,
}

/// Configuration for the Ollama LLM intent backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OllamaConfig {
    /// Ollama API base URL.
    pub url: String,
    /// Model name.
    pub model: String,
    /// System prompt for the assistant persona.
    pub system_prompt: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:11434".into(),
            model: "gemma3:1b".into(),
            system_prompt: "You are Jarvis, a helpful voice assistant. Keep responses brief and conversational (1-2 sentences).".into(),
        }
    }
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
    fn whisper_config_from_toml() {
        let toml_str = r#"
[stt]
backend = "whisper"

[stt.whisper]
model_dir = "models/whisper-small"
language = "en"
"#;
        let parsed: PipelineConfig = toml::from_str(toml_str).unwrap();
        assert!(matches!(parsed.stt.backend, SttBackend::Whisper));
        assert_eq!(
            parsed.stt.whisper.model_dir,
            PathBuf::from("models/whisper-small")
        );
    }

    #[test]
    fn kokoro_config_from_toml() {
        let toml_str = r#"
[tts]
backend = "kokoro"

[tts.kokoro]
model_path = "models/kokoro.onnx"
voice = "bf_emma"
"#;
        let parsed: PipelineConfig = toml::from_str(toml_str).unwrap();
        assert!(matches!(parsed.tts.backend, TtsBackend::Kokoro));
        assert_eq!(parsed.tts.kokoro.voice, "bf_emma");
    }

    #[test]
    fn partial_toml_uses_defaults() {
        let parsed: PipelineConfig = toml::from_str("").unwrap();
        assert!(matches!(parsed.stt.backend, SttBackend::Echo));
        assert!(matches!(parsed.tts.backend, TtsBackend::Echo));
        assert_eq!(parsed.stt.whisper.language, "en");
        assert_eq!(parsed.tts.kokoro.voice, "af_heart");
    }
}
