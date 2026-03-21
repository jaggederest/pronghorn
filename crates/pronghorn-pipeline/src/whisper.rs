use pronghorn_audio::AudioFrame;
use tokio::sync::mpsc;
use tracing::info;

use crate::config::WhisperConfig;
use crate::stt::{SpeechToText, SttError, Transcript};

/// Whisper STT backend using rusty-whisper (tract ONNX inference).
///
/// Loads encoder + decoder ONNX models, mel filterbank, tokenizer, and
/// positional embeddings from a model directory. Audio frames are collected,
/// written as a temp WAV file, and transcribed.
pub struct WhisperStt {
    #[cfg(feature = "whisper")]
    config: WhisperConfig,
    #[cfg(feature = "whisper")]
    whisper: rusty_whisper::Whisper,
    #[cfg(not(feature = "whisper"))]
    _config: WhisperConfig,
}

impl WhisperStt {
    pub fn new(config: &WhisperConfig) -> Result<Self, SttError> {
        #[cfg(feature = "whisper")]
        {
            let dir = &config.model_dir;
            if !dir.exists() {
                return Err(SttError::Connection(format!(
                    "model directory not found: {}",
                    dir.display()
                )));
            }

            let encoder = dir.join("encoder_model.onnx");
            let decoder = dir.join("decoder_model_merged.onnx");
            let tokenizer = dir.join("multilingual.tiktoken");
            let pos_emb = dir.join("positional_embedding.npz");
            let mel_filters = dir.join("mel_filters.npz");

            for (name, path) in [
                ("encoder", &encoder),
                ("decoder", &decoder),
                ("tokenizer", &tokenizer),
                ("positional embedding", &pos_emb),
                ("mel filters", &mel_filters),
            ] {
                if !path.exists() {
                    return Err(SttError::Connection(format!(
                        "{name} not found: {}",
                        path.display()
                    )));
                }
            }

            info!(model_dir = %dir.display(), language = %config.language, "loading whisper model");

            // rusty-whisper panics on errors internally — catch panics
            let whisper = std::panic::catch_unwind(|| {
                rusty_whisper::Whisper::new(
                    encoder.to_str().unwrap(),
                    decoder.to_str().unwrap(),
                    tokenizer.to_str().unwrap(),
                    pos_emb.to_str().unwrap(),
                    mel_filters.to_str().unwrap(),
                )
            })
            .map_err(|_| SttError::Connection("whisper model loading panicked".into()))?;

            info!("whisper model loaded");

            Ok(Self {
                config: config.clone(),
                whisper,
            })
        }

        #[cfg(not(feature = "whisper"))]
        {
            Ok(Self {
                _config: config.clone(),
            })
        }
    }
}

impl SpeechToText for WhisperStt {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioFrame>,
        transcript_tx: mpsc::Sender<Transcript>,
    ) -> Result<(), SttError> {
        // Step 1: Collect all audio frames
        let mut samples_i16: Vec<i16> = Vec::new();
        while let Some(frame) = audio_rx.recv().await {
            for chunk in frame.samples.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                samples_i16.push(sample);
            }
        }

        let total_samples = samples_i16.len();
        let duration_ms = total_samples as u64 * 1000 / 16_000;
        info!(samples = total_samples, duration_ms, "transcribing audio");

        if samples_i16.is_empty() {
            let _ = transcript_tx
                .send(Transcript {
                    text: String::new(),
                    is_final: true,
                })
                .await;
            return Ok(());
        }

        #[cfg(feature = "whisper")]
        {
            // Step 2: Write temp WAV file (rusty-whisper only accepts file paths)
            let wav_path = write_temp_wav(&samples_i16)
                .map_err(|e| SttError::Stream(format!("failed to write temp WAV: {e}")))?;

            let wav_path_str = wav_path.to_string_lossy().to_string();
            let language = self.config.language.clone();

            // Step 3: Run inference on a blocking thread (tract is CPU-bound)
            // We need to move the whisper ref into the blocking task.
            // Since Whisper may not be Send, we use the path approach.
            let whisper_ref = &self.whisper;
            let text = tokio::task::spawn_blocking({
                // SAFETY: whisper is borrowed for the duration of the blocking task.
                // The task completes before this function returns.
                let whisper_ptr = whisper_ref as *const rusty_whisper::Whisper;
                move || {
                    let whisper = unsafe { &*whisper_ptr };
                    whisper.recognize_from_audio(&wav_path_str, &language)
                }
            })
            .await
            .map_err(|e| SttError::Stream(format!("whisper inference task failed: {e}")))?;

            // Clean up temp file
            let _ = std::fs::remove_file(&wav_path);

            let text = text.trim().to_string();
            info!(text = %text, duration_ms, "transcription complete");

            let _ = transcript_tx
                .send(Transcript {
                    text,
                    is_final: true,
                })
                .await;
        }

        #[cfg(not(feature = "whisper"))]
        {
            let _ = transcript_tx
                .send(Transcript {
                    text: format!("[whisper stub: {duration_ms}ms audio]"),
                    is_final: true,
                })
                .await;
        }

        Ok(())
    }
}

#[cfg(feature = "whisper")]
fn write_temp_wav(samples: &[i16]) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    use hound::{SampleFormat, WavSpec, WavWriter};

    let temp_dir = tempfile::tempdir()?;
    let path = temp_dir.into_path().join("audio.wav");

    let spec = WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(&path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    Ok(path)
}
