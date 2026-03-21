use pronghorn_audio::AudioFrame;
use tokio::sync::mpsc;
#[cfg(feature = "sherpa")]
use tokio::sync::oneshot;
use tracing::info;

use crate::config::SherpaConfig;
use crate::stt::{SpeechToText, SttError, Transcript};

/// Request sent to the Sherpa inference thread.
#[cfg(feature = "sherpa")]
struct SherpaRequest {
    samples: Vec<f32>,
    reply: oneshot::Sender<Result<String, String>>,
}

/// STT backend using sherpa-onnx's Zipformer transducer.
///
/// Currently uses the offline (batch) recognizer, which is still dramatically
/// faster than Whisper (~10-50x realtime vs ~0.85x). The sherpa-rs crate
/// doesn't wrap the online/streaming C API yet. When it does (or when we
/// add raw FFI), this will switch to true frame-by-frame streaming.
///
/// The model lives on a dedicated thread (same pattern as Whisper/Kokoro).
pub struct SherpaStt {
    #[cfg(feature = "sherpa")]
    request_tx: std::sync::mpsc::Sender<SherpaRequest>,
    #[cfg(not(feature = "sherpa"))]
    _config: SherpaConfig,
}

unsafe impl Send for SherpaStt {}
unsafe impl Sync for SherpaStt {}

impl SherpaStt {
    pub fn new(config: &SherpaConfig) -> Result<Self, SttError> {
        #[cfg(feature = "sherpa")]
        {
            let dir = &config.model_dir;
            if !dir.exists() {
                return Err(SttError::Connection(format!(
                    "model directory not found: {}",
                    dir.display()
                )));
            }

            let encoder = dir.join("encoder.onnx");
            let decoder = dir.join("decoder.onnx");
            let joiner = dir.join("joiner.onnx");
            let tokens = dir.join("tokens.txt");

            for (name, path) in [
                ("encoder", &encoder),
                ("decoder", &decoder),
                ("joiner", &joiner),
                ("tokens", &tokens),
            ] {
                if !path.exists() {
                    return Err(SttError::Connection(format!(
                        "{name} not found: {}",
                        path.display()
                    )));
                }
            }

            info!(model_dir = %dir.display(), "loading sherpa STT");

            let (request_tx, request_rx) = std::sync::mpsc::channel::<SherpaRequest>();

            let encoder_str = encoder.to_string_lossy().to_string();
            let decoder_str = decoder.to_string_lossy().to_string();
            let joiner_str = joiner.to_string_lossy().to_string();
            let tokens_str = tokens.to_string_lossy().to_string();

            std::thread::Builder::new()
                .name("sherpa-stt".into())
                .spawn(move || {
                    use sherpa_rs::zipformer::{ZipFormer, ZipFormerConfig};

                    let config = ZipFormerConfig {
                        encoder: encoder_str,
                        decoder: decoder_str,
                        joiner: joiner_str,
                        tokens: tokens_str,
                        num_threads: Some(2),
                        debug: false,
                        ..Default::default()
                    };

                    let mut recognizer = match ZipFormer::new(config) {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::error!("failed to create sherpa recognizer: {e}");
                            return;
                        }
                    };

                    tracing::info!("sherpa recognizer loaded on inference thread");

                    while let Ok(req) = request_rx.recv() {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            recognizer.decode(16000, req.samples)
                        }));

                        let result = match result {
                            Ok(text) => Ok(text),
                            Err(_) => Err("sherpa inference panicked".into()),
                        };
                        let _ = req.reply.send(result);
                    }
                })
                .map_err(|e| SttError::Connection(format!("failed to spawn sherpa thread: {e}")))?;

            info!("sherpa STT thread started");
            Ok(Self { request_tx })
        }

        #[cfg(not(feature = "sherpa"))]
        {
            Ok(Self {
                _config: config.clone(),
            })
        }
    }
}

impl SpeechToText for SherpaStt {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioFrame>,
        transcript_tx: mpsc::Sender<Transcript>,
    ) -> Result<(), SttError> {
        // Collect all audio frames (batch mode for now)
        let mut all_samples: Vec<f32> = Vec::new();
        while let Some(frame) = audio_rx.recv().await {
            for chunk in frame.samples.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                all_samples.push(sample);
            }
        }

        let duration_ms = all_samples.len() as u64 * 1000 / 16_000;
        info!(
            samples = all_samples.len(),
            duration_ms, "sherpa transcribing"
        );

        if all_samples.is_empty() {
            let _ = transcript_tx
                .send(Transcript {
                    text: String::new(),
                    is_final: true,
                })
                .await;
            return Ok(());
        }

        #[cfg(feature = "sherpa")]
        {
            let (reply_tx, reply_rx) = oneshot::channel();
            self.request_tx
                .send(SherpaRequest {
                    samples: all_samples,
                    reply: reply_tx,
                })
                .map_err(|_| SttError::Stream("sherpa thread gone".into()))?;

            let text = reply_rx
                .await
                .map_err(|_| SttError::Stream("sherpa thread dropped reply".into()))?
                .map_err(SttError::Stream)?;

            let text = text.trim().to_string();
            info!(text = %text, duration_ms, "sherpa transcription complete");

            let _ = transcript_tx
                .send(Transcript {
                    text,
                    is_final: true,
                })
                .await;
        }

        #[cfg(not(feature = "sherpa"))]
        {
            let _ = transcript_tx
                .send(Transcript {
                    text: format!("[sherpa stub: {duration_ms}ms audio]"),
                    is_final: true,
                })
                .await;
        }

        Ok(())
    }
}
