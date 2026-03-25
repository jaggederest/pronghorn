use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Aggregated pipeline configuration — lives under `[pipeline]` in server.toml.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelineConfig {
    pub stt: SttConfig,
    pub tts: TtsConfig,
    pub intent: IntentConfig,
    pub vad: VadConfig,
    pub ha: HaConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    pub backend: SttBackend,
    /// Whisper-specific configuration.
    pub whisper: WhisperConfig,
    /// Sherpa-ONNX streaming configuration.
    pub sherpa: SherpaConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SttBackend {
    #[default]
    Echo,
    Whisper,
    Sherpa,
}

/// Configuration for the Sherpa-ONNX streaming STT backend.
///
/// Uses the Online (streaming) Recognizer for frame-by-frame transcription
/// with partial results. Requires a streaming-capable model (e.g. Zipformer
/// transducer trained with causal attention).
///
/// The directory must contain: encoder, decoder, joiner ONNX files and tokens.txt.
/// File discovery is automatic — looks for `*encoder*.onnx`, `*decoder*.onnx`, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SherpaConfig {
    /// Directory containing the streaming model files.
    pub model_dir: PathBuf,
}

impl Default for SherpaConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"),
        }
    }
}

impl SherpaConfig {
    /// Find a model file by glob pattern within model_dir.
    /// Prefers int8 quantized models if available, falls back to fp32.
    fn find_model_file(&self, pattern: &str) -> Option<PathBuf> {
        let full_pattern = self.model_dir.join(pattern);
        let mut matches: Vec<PathBuf> = glob::glob(&full_pattern.to_string_lossy())
            .ok()?
            .filter_map(|r| r.ok())
            .collect();
        // Sort so int8 comes first (alphabetically "int8" < "onnx" without int8)
        matches.sort();
        matches.into_iter().next()
    }

    /// Resolve the encoder, decoder, joiner, and tokens paths from model_dir.
    /// Returns (encoder, decoder, joiner, tokens) or an error describing what's missing.
    pub fn resolve_model_files(&self) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf), String> {
        let encoder = self
            .find_model_file("*encoder*.onnx")
            .ok_or_else(|| format!("no *encoder*.onnx in {}", self.model_dir.display()))?;
        let decoder = self
            .find_model_file("*decoder*.onnx")
            .ok_or_else(|| format!("no *decoder*.onnx in {}", self.model_dir.display()))?;
        let joiner = self
            .find_model_file("*joiner*.onnx")
            .ok_or_else(|| format!("no *joiner*.onnx in {}", self.model_dir.display()))?;
        let tokens = self.model_dir.join("tokens.txt");
        if !tokens.exists() {
            return Err(format!("no tokens.txt in {}", self.model_dir.display()));
        }
        Ok((encoder, decoder, joiner, tokens))
    }
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
    /// Kokoro via kokoroxide (standalone ort).
    pub kokoro: KokoroConfig,
    /// Kokoro via sherpa-rs (shared ort with STT).
    pub sherpa_kokoro: SherpaKokoroConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TtsBackend {
    #[default]
    Echo,
    Kokoro,
    SherpaKokoro,
}

/// Configuration for Kokoro TTS via sherpa-rs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SherpaKokoroConfig {
    /// Directory containing model.onnx, voices.bin, tokens.txt, espeak-ng-data/
    pub model_dir: PathBuf,
    /// Speaker ID (0-10 for kokoro-en-v0_19).
    pub speaker_id: i32,
    /// Speech speed multiplier.
    pub speed: f32,
}

impl Default for SherpaKokoroConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("models/sherpa-kokoro"),
            speaker_id: 0,
            speed: 1.0,
        }
    }
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

/// Configuration for the Home Assistant WebSocket client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HaConfig {
    /// WebSocket URL of the HA instance, e.g. `ws://homeassistant.local:8123/api/websocket`.
    pub url: String,
    /// Long-lived access token from HA → Profile → Long-Lived Access Tokens.
    pub token: String,
}

impl Default for HaConfig {
    fn default() -> Self {
        Self {
            url: "ws://homeassistant.local:8123/api/websocket".into(),
            token: String::new(),
        }
    }
}

/// Configuration for server-side Silero VAD endpoint detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VadConfig {
    /// Whether to enable server-side Silero VAD for endpoint detection.
    pub enabled: bool,
    /// Path to silero_vad.onnx model file.
    pub model_path: PathBuf,
    /// Seconds of silence after speech to trigger endpoint.
    pub min_silence_duration: f32,
    /// Minimum speech duration before endpoint detection activates.
    pub min_speech_duration: f32,
    /// VAD detection threshold (0.0–1.0).
    pub threshold: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_path: PathBuf::from("models/silero_vad.onnx"),
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            threshold: 0.5,
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
