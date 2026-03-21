use pronghorn_audio::AudioFrame;
use tokio::sync::mpsc;
use tracing::info;

use crate::config::WhisperConfig;
use crate::stt::{SpeechToText, SttError, Transcript};

/// Whisper STT backend using ONNX Runtime.
///
/// Loads a Whisper ONNX model and runs inference in-process.
/// Audio frames are collected, converted to f32, mel spectrogram is computed,
/// then the encoder+decoder produce text.
pub struct WhisperStt {
    config: WhisperConfig,
    // TODO: ort::Session once model loading is implemented
}

impl WhisperStt {
    pub fn new(config: &WhisperConfig) -> Result<Self, SttError> {
        if !config.model_path.exists() {
            return Err(SttError::Connection(format!(
                "model file not found: {}",
                config.model_path.display()
            )));
        }
        info!(model = %config.model_path.display(), language = %config.language, "whisper STT initialized");
        Ok(Self {
            config: config.clone(),
        })
    }
}

impl SpeechToText for WhisperStt {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioFrame>,
        transcript_tx: mpsc::Sender<Transcript>,
    ) -> Result<(), SttError> {
        // Step 1: Collect all audio frames into a single f32 buffer
        let mut samples_i16: Vec<i16> = Vec::new();
        while let Some(frame) = audio_rx.recv().await {
            // AudioFrame.samples is Bytes containing little-endian i16 PCM
            for chunk in frame.samples.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                samples_i16.push(sample);
            }
        }

        let total_samples = samples_i16.len();
        let duration_ms = total_samples as u64 * 1000 / 16_000;
        info!(
            samples = total_samples,
            duration_ms,
            model = %self.config.model_path.display(),
            "collected audio for whisper transcription"
        );

        // Step 2: Convert i16 → f32
        let _samples_f32: Vec<f32> = samples_i16.iter().map(|&s| s as f32 / 32768.0).collect();

        // TODO: Step 3 — Compute 80-bin log-mel spectrogram
        // This requires FFT (rustfft) + mel filterbank + log scaling.
        // Whisper uses 80 mel bins, 25ms window, 10ms hop, 16kHz input.

        // TODO: Step 4 — Run encoder ONNX model
        // Input: mel spectrogram tensor [1, 80, T]
        // Output: encoder hidden states

        // TODO: Step 5 — Run decoder ONNX model (autoregressive)
        // Input: encoder output + token IDs
        // Output: next token logits
        // Repeat until <|endoftext|> token

        // TODO: Step 6 — Decode token IDs → text

        // STUB: return a placeholder indicating model path and audio duration
        let text = format!(
            "[whisper stub: {}ms audio, model={}]",
            duration_ms,
            self.config.model_path.display()
        );

        let _ = transcript_tx
            .send(Transcript {
                text,
                is_final: true,
            })
            .await;

        Ok(())
    }
}
