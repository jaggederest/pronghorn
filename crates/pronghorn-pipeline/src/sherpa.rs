use pronghorn_audio::AudioFrame;
use tokio::sync::mpsc;
#[cfg(feature = "sherpa")]
use tracing::{debug, info, trace};

use crate::config::SherpaConfig;
use crate::stt::{SpeechToText, SttError, Transcript};

/// Message sent to the Sherpa inference thread.
#[cfg(feature = "sherpa")]
enum SherpaMsg {
    /// Start a new transcription session. The thread creates a fresh OnlineStream
    /// and sends transcripts back through the provided sender.
    NewSession(mpsc::Sender<Transcript>),
    /// Feed audio samples to the current session's stream.
    AudioChunk(Vec<f32>),
    /// Signal end of audio for the current session.
    InputFinished,
}

/// STT backend using sherpa-onnx's streaming Online Recognizer.
///
/// Processes audio frame-by-frame via the Zipformer transducer, emitting
/// partial transcripts as words are recognized. The final transcript is
/// available almost instantly after the last audio chunk.
///
/// The recognizer lives on a dedicated thread. Each transcribe() call sends
/// a NewSession message followed by audio chunks and InputFinished.
pub struct SherpaStt {
    #[cfg(feature = "sherpa")]
    msg_tx: std::sync::mpsc::Sender<SherpaMsg>,
    #[cfg(not(feature = "sherpa"))]
    _config: SherpaConfig,
}

unsafe impl Send for SherpaStt {}
unsafe impl Sync for SherpaStt {}

impl SherpaStt {
    pub fn new(config: &SherpaConfig) -> Result<Self, SttError> {
        #[cfg(feature = "sherpa")]
        {
            let (encoder, decoder, joiner, tokens) =
                config.resolve_model_files().map_err(SttError::Connection)?;

            info!(
                model_dir = %config.model_dir.display(),
                encoder = %encoder.display(),
                "loading sherpa streaming STT"
            );

            let (msg_tx, msg_rx) = std::sync::mpsc::channel::<SherpaMsg>();

            let encoder_str = encoder.to_string_lossy().to_string();
            let decoder_str = decoder.to_string_lossy().to_string();
            let joiner_str = joiner.to_string_lossy().to_string();
            let tokens_str = tokens.to_string_lossy().to_string();

            std::thread::Builder::new()
                .name("sherpa-stt".into())
                .spawn(move || {
                    run_inference_thread(
                        msg_rx,
                        &encoder_str,
                        &decoder_str,
                        &joiner_str,
                        &tokens_str,
                    );
                })
                .map_err(|e| SttError::Connection(format!("failed to spawn sherpa thread: {e}")))?;

            info!("sherpa streaming STT thread started");
            Ok(Self { msg_tx })
        }

        #[cfg(not(feature = "sherpa"))]
        {
            Ok(Self {
                _config: config.clone(),
            })
        }
    }
}

/// Inference thread: creates the OnlineRecognizer once, handles sessions sequentially.
#[cfg(feature = "sherpa")]
fn run_inference_thread(
    msg_rx: std::sync::mpsc::Receiver<SherpaMsg>,
    encoder: &str,
    decoder: &str,
    joiner: &str,
    tokens: &str,
) {
    use sherpa_rs::online_recognizer::{OnlineRecognizer, OnlineRecognizerConfig};

    let config = OnlineRecognizerConfig {
        encoder: encoder.into(),
        decoder: decoder.into(),
        joiner: joiner.into(),
        tokens: tokens.into(),
        num_threads: Some(2),
        decoding_method: "greedy_search".into(),
        debug: false,
        ..Default::default()
    };

    let recognizer = match OnlineRecognizer::new(config) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("failed to create online recognizer: {e}");
            return;
        }
    };

    tracing::info!("sherpa online recognizer loaded on inference thread");

    // Wait for session messages
    let mut transcript_tx: Option<mpsc::Sender<Transcript>> = None;
    let mut stream = None;
    let mut last_text = String::new();
    let mut chunk_count = 0u32;
    let mut total_fed = 0usize;

    while let Ok(msg) = msg_rx.recv() {
        match msg {
            SherpaMsg::NewSession(tx) => {
                // Create a fresh stream for this session
                match recognizer.create_stream() {
                    Ok(s) => {
                        stream = Some(s);
                        transcript_tx = Some(tx);
                        last_text.clear();
                        chunk_count = 0;
                        total_fed = 0;
                        tracing::debug!("new streaming session started");
                    }
                    Err(e) => {
                        tracing::error!("failed to create online stream: {e}");
                        let _ = tx.blocking_send(Transcript {
                            text: String::new(),
                            is_final: true,
                        });
                    }
                }
            }
            SherpaMsg::AudioChunk(samples) => {
                let Some(ref s) = stream else { continue };
                let Some(ref tx) = transcript_tx else {
                    continue;
                };

                chunk_count += 1;
                total_fed += samples.len();
                if chunk_count <= 3 || chunk_count % 25 == 0 {
                    let rms: f32 =
                        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
                    tracing::debug!(
                        chunk = chunk_count,
                        samples = samples.len(),
                        total_fed,
                        rms,
                        "feeding audio to recognizer"
                    );
                }

                s.accept_waveform(16000, &samples);

                let mut decoded = 0;
                while recognizer.is_ready(s) {
                    recognizer.decode(s);
                    decoded += 1;
                }
                if decoded > 0 {
                    tracing::debug!(decoded, "decoded chunks");
                }

                let text = recognizer.get_result(s).trim().to_string();
                trace!(text = %text, chunk = chunk_count, "recognizer state");
                if !text.is_empty() && text != last_text {
                    debug!(text = %text, "partial transcript");
                    last_text.clone_from(&text);
                    let _ = tx.blocking_send(Transcript {
                        text,
                        is_final: false,
                    });
                }
            }
            SherpaMsg::InputFinished => {
                let Some(ref s) = stream else { continue };
                let Some(ref tx) = transcript_tx else {
                    continue;
                };

                debug!(chunk_count, total_fed, "InputFinished, flushing recognizer");
                s.input_finished();

                let mut final_decoded = 0;
                while recognizer.is_ready(s) {
                    recognizer.decode(s);
                    final_decoded += 1;
                }
                debug!(final_decoded, "final decode cycles after input_finished");

                let text = recognizer.get_result(s).trim().to_string();
                tracing::info!(text = %text, "final transcript");
                let _ = tx.blocking_send(Transcript {
                    text,
                    is_final: true,
                });

                // Clean up session state — stream will be dropped
                stream = None;
                transcript_tx = None;
                last_text.clear();
            }
        }
    }
}

impl SpeechToText for SherpaStt {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioFrame>,
        transcript_tx: mpsc::Sender<Transcript>,
    ) -> Result<(), SttError> {
        #[cfg(feature = "sherpa")]
        {
            // Create a channel for this session's transcripts
            let (session_tx, mut session_rx) = mpsc::channel::<Transcript>(16);

            // Tell the inference thread to start a new session
            self.msg_tx
                .send(SherpaMsg::NewSession(session_tx))
                .map_err(|_| SttError::Stream("sherpa thread gone".into()))?;

            let mut total_samples = 0usize;

            // Stream audio chunks to the inference thread
            while let Some(frame) = audio_rx.recv().await {
                let samples: Vec<f32> = frame
                    .samples
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                total_samples += samples.len();
                self.msg_tx
                    .send(SherpaMsg::AudioChunk(samples))
                    .map_err(|_| SttError::Stream("sherpa thread gone".into()))?;
            }

            // Audio channel closed (VAD endpoint or satellite timeout)
            let duration_ms = total_samples as u64 * 1000 / 16_000;
            info!(
                total_samples,
                duration_ms, "audio stream ended, finalizing STT"
            );

            self.msg_tx
                .send(SherpaMsg::InputFinished)
                .map_err(|_| SttError::Stream("sherpa thread gone".into()))?;

            // Forward transcripts from inference thread to the pipeline
            while let Some(transcript) = session_rx.recv().await {
                let is_final = transcript.is_final;
                let _ = transcript_tx.send(transcript).await;
                if is_final {
                    break;
                }
            }
        }

        #[cfg(not(feature = "sherpa"))]
        {
            let mut frame_count = 0u32;
            while let Some(_frame) = audio_rx.recv().await {
                frame_count += 1;
            }
            let _ = transcript_tx
                .send(Transcript {
                    text: format!("[sherpa stub: {frame_count} frames]"),
                    is_final: true,
                })
                .await;
        }

        Ok(())
    }
}

#[cfg(all(test, feature = "sherpa"))]
mod tests {
    use super::*;
    use bytes::Bytes;
    use pronghorn_audio::{AudioFormat, AudioFrame};
    use std::path::PathBuf;

    fn silence_frame() -> AudioFrame {
        AudioFrame::new(AudioFormat::SPEECH, Bytes::from(vec![0u8; 640]))
    }

    fn test_model_config() -> Option<crate::config::SherpaConfig> {
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let config = crate::config::SherpaConfig {
            model_dir: workspace_root
                .join("models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"),
        };
        if config.model_dir.exists() {
            Some(config)
        } else {
            eprintln!(
                "skipping: model dir not found at {}",
                config.model_dir.display()
            );
            None
        }
    }

    /// Test that the OnlineRecognizer can be created with the streaming model.
    /// This catches FFI struct layout mismatches that cause segfaults.
    #[test]
    fn online_recognizer_creates_without_segfault() {
        use sherpa_rs::online_recognizer::{OnlineRecognizer, OnlineRecognizerConfig};

        let Some(config) = test_model_config() else {
            return;
        };
        let (encoder, decoder, joiner, tokens) = config.resolve_model_files().unwrap();

        let recognizer_config = OnlineRecognizerConfig {
            encoder: encoder.to_string_lossy().into(),
            decoder: decoder.to_string_lossy().into(),
            joiner: joiner.to_string_lossy().into(),
            tokens: tokens.to_string_lossy().into(),
            num_threads: Some(1),
            debug: true,
            ..Default::default()
        };

        let recognizer =
            OnlineRecognizer::new(recognizer_config).expect("failed to create recognizer");
        let stream = recognizer.create_stream().expect("failed to create stream");

        // Feed 1 second of silence
        let silence: Vec<f32> = vec![0.0; 16000];
        stream.accept_waveform(16000, &silence);

        while recognizer.is_ready(&stream) {
            recognizer.decode(&stream);
        }

        let result = recognizer.get_result(&stream);
        eprintln!("recognizer result for silence: {:?}", result);
    }

    /// Test SherpaStt streaming transcription end-to-end via the SpeechToText trait.
    #[tokio::test]
    async fn streaming_stt_produces_final_transcript() {
        let Some(config) = test_model_config() else {
            return;
        };
        let stt = SherpaStt::new(&config).expect("failed to create SherpaStt");

        let (audio_tx, audio_rx) = mpsc::channel::<AudioFrame>(64);
        let (transcript_tx, mut transcript_rx) = mpsc::channel::<Transcript>(16);

        let stt_handle = tokio::spawn(async move { stt.transcribe(audio_rx, transcript_tx).await });

        // Send 50 frames of silence (~1 second) then close
        for _ in 0..50 {
            audio_tx.send(silence_frame()).await.unwrap();
        }
        drop(audio_tx);

        stt_handle.await.unwrap().unwrap();

        // Should get at least a final transcript
        let mut got_final = false;
        while let Some(transcript) = transcript_rx.recv().await {
            if transcript.is_final {
                got_final = true;
                eprintln!("final transcript: {:?}", transcript.text);
                break;
            }
        }
        assert!(got_final, "expected a final transcript");
    }
}
