use pronghorn_audio::AudioFrame;
use tokio::sync::mpsc;
#[cfg(feature = "whisper")]
use tokio::sync::oneshot;
use tracing::info;

use crate::config::WhisperConfig;
use crate::stt::{SpeechToText, SttError, Transcript};

/// Request sent to the Whisper inference thread.
#[cfg(feature = "whisper")]
struct InferenceRequest {
    samples_i16: Vec<i16>,
    language: String,
    reply: oneshot::Sender<Result<String, String>>,
}

/// Whisper STT backend using rusty-whisper (tract ONNX inference).
///
/// The Whisper model (from rusty-whisper) isn't Send, so it lives on a
/// dedicated background thread. Inference requests are sent via channel.
pub struct WhisperStt {
    #[cfg(feature = "whisper")]
    request_tx: std::sync::mpsc::Sender<InferenceRequest>,
    #[cfg(feature = "whisper")]
    language: String,
    #[cfg(not(feature = "whisper"))]
    _config: WhisperConfig,
}

// SAFETY: The Whisper model lives on its own thread. WhisperStt only holds
// a channel sender (which is Send+Sync). No direct access to the model.
unsafe impl Send for WhisperStt {}
unsafe impl Sync for WhisperStt {}

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

            info!(model_dir = %dir.display(), "loading whisper model");

            // Spawn a dedicated thread that owns the Whisper model
            let (request_tx, request_rx) = std::sync::mpsc::channel::<InferenceRequest>();

            let encoder_str = encoder.to_string_lossy().to_string();
            let decoder_str = decoder.to_string_lossy().to_string();
            let tokenizer_str = tokenizer.to_string_lossy().to_string();
            let pos_emb_str = pos_emb.to_string_lossy().to_string();
            let mel_filters_str = mel_filters.to_string_lossy().to_string();

            std::thread::Builder::new()
                .name("whisper-inference".into())
                .spawn(move || {
                    let whisper = rusty_whisper::Whisper::new(
                        &encoder_str,
                        &decoder_str,
                        &tokenizer_str,
                        &pos_emb_str,
                        &mel_filters_str,
                    );
                    tracing::info!("whisper model loaded on inference thread");

                    while let Ok(req) = request_rx.recv() {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            // Write temp WAV
                            let wav_path = match write_temp_wav(&req.samples_i16) {
                                Ok(p) => p,
                                Err(e) => return Err(format!("failed to write temp WAV: {e}")),
                            };
                            let wav_str = wav_path.to_string_lossy().to_string();
                            let text = whisper.recognize_from_audio(&wav_str, &req.language);
                            let _ = std::fs::remove_file(&wav_path);
                            Ok(text)
                        }));

                        let result = match result {
                            Ok(r) => r,
                            Err(_) => Err("whisper inference panicked".into()),
                        };
                        let _ = req.reply.send(result);
                    }
                })
                .map_err(|e| {
                    SttError::Connection(format!("failed to spawn whisper thread: {e}"))
                })?;

            info!("whisper inference thread started");
            Ok(Self {
                request_tx,
                language: config.language.clone(),
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
            let (reply_tx, reply_rx) = oneshot::channel();
            self.request_tx
                .send(InferenceRequest {
                    samples_i16,
                    language: self.language.clone(),
                    reply: reply_tx,
                })
                .map_err(|_| SttError::Stream("whisper inference thread gone".into()))?;

            let text = reply_rx
                .await
                .map_err(|_| SttError::Stream("whisper inference thread dropped reply".into()))?
                .map_err(|e| SttError::Stream(e))?;

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
fn write_temp_wav(
    samples: &[i16],
) -> Result<std::path::PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    use hound::{SampleFormat, WavSpec, WavWriter};

    let dir = std::env::temp_dir().join("pronghorn-whisper");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.wav", std::process::id()));

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
