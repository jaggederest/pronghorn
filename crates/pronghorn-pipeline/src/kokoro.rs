use bytes::Bytes;
use pronghorn_audio::{AudioFormat, AudioFrame};
use tokio::sync::mpsc;
use tracing::info;

use crate::config::KokoroConfig;
use crate::resample::Resampler;
use crate::tts::{TextToSpeech, TtsError};

/// Kokoro TTS backend using ONNX Runtime.
///
/// Loads a Kokoro ONNX model (82M params) and runs synthesis in-process.
/// Output is 24kHz f32 audio, resampled to 16kHz i16 for our wire format.
pub struct KokoroTts {
    config: KokoroConfig,
    // TODO: ort::Session once model loading is implemented
}

impl KokoroTts {
    pub fn new(config: &KokoroConfig) -> Result<Self, TtsError> {
        if !config.model_path.exists() {
            return Err(TtsError::Connection(format!(
                "model file not found: {}",
                config.model_path.display()
            )));
        }
        info!(
            model = %config.model_path.display(),
            voice = %config.voice,
            "kokoro TTS initialized"
        );
        Ok(Self {
            config: config.clone(),
        })
    }
}

impl TextToSpeech for KokoroTts {
    async fn synthesize(
        &self,
        text: &str,
        audio_tx: mpsc::Sender<AudioFrame>,
    ) -> Result<(), TtsError> {
        info!(
            text_len = text.len(),
            voice = %self.config.voice,
            model = %self.config.model_path.display(),
            "kokoro synthesizing"
        );

        // TODO: Step 1 — Phonemize text → phoneme tokens
        // Kokoro expects phoneme token IDs, not raw text.
        // This may require espeak-ng bindings or a pure-Rust phonemizer.

        // TODO: Step 2 — Load voice embedding
        // Voice files are .npy or .bin files in the voices directory.

        // TODO: Step 3 — Run ONNX model
        // Input: token IDs tensor + voice embedding tensor
        // Output: f32 audio samples at 24kHz

        // TODO: Step 4 — Resample 24kHz → 16kHz
        // TODO: Step 5 — Convert f32 → i16 PCM

        // STUB: generate silence frames proportional to text length,
        // but go through the resample path to verify the pipeline works.
        let stub_duration_samples = text.len() * 480; // ~20ms per char at 24kHz
        let samples_24k: Vec<f32> = vec![0.0; stub_duration_samples];

        // Resample 24kHz → 16kHz
        let mut resampler = Resampler::new_24k_to_16k();
        let samples_16k = resampler.process(&samples_24k);

        // Convert f32 → i16 PCM and chunk into 20ms frames (640 bytes = 320 i16 samples)
        let samples_i16: Vec<i16> = samples_16k
            .iter()
            .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        let frame_samples = 320; // 20ms at 16kHz mono
        for chunk in samples_i16.chunks(frame_samples) {
            let pcm_bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
            let frame = AudioFrame::new(AudioFormat::SPEECH, Bytes::from(pcm_bytes));
            if audio_tx.send(frame).await.is_err() {
                break; // receiver dropped
            }
        }

        Ok(())
    }
}
