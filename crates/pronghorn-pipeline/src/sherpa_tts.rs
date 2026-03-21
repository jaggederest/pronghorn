use pronghorn_audio::AudioFrame;
use tokio::sync::mpsc;
#[cfg(feature = "sherpa")]
use tokio::sync::oneshot;
use tracing::info;

use crate::config::SherpaKokoroConfig;
use crate::tts::{TextToSpeech, TtsError};

#[cfg(feature = "sherpa")]
struct TtsRequest {
    text: String,
    reply: oneshot::Sender<Result<(Vec<f32>, u32), String>>,
}

/// Kokoro TTS backend via sherpa-rs.
///
/// Shares the sherpa-onnx runtime with SherpaStt. Phonemization is handled
/// internally by sherpa-onnx (espeak-ng data bundled in the model archive).
pub struct SherpaTts {
    #[cfg(feature = "sherpa")]
    request_tx: std::sync::mpsc::Sender<TtsRequest>,
    #[cfg(not(feature = "sherpa"))]
    _config: SherpaKokoroConfig,
}

unsafe impl Send for SherpaTts {}
unsafe impl Sync for SherpaTts {}

impl SherpaTts {
    pub fn new(config: &SherpaKokoroConfig) -> Result<Self, TtsError> {
        #[cfg(feature = "sherpa")]
        {
            let dir = &config.model_dir;
            if !dir.exists() {
                return Err(TtsError::Connection(format!(
                    "model directory not found: {}",
                    dir.display()
                )));
            }

            info!(
                model_dir = %dir.display(),
                speaker_id = config.speaker_id,
                speed = config.speed,
                "loading sherpa kokoro TTS"
            );

            let (request_tx, request_rx) = std::sync::mpsc::channel::<TtsRequest>();

            let model = dir.join("model.onnx").to_string_lossy().to_string();
            let voices = dir.join("voices.bin").to_string_lossy().to_string();
            let tokens = dir.join("tokens.txt").to_string_lossy().to_string();
            let data_dir = dir.join("espeak-ng-data").to_string_lossy().to_string();
            let speaker_id = config.speaker_id;
            let speed = config.speed;

            std::thread::Builder::new()
                .name("sherpa-tts".into())
                .spawn(move || {
                    use sherpa_rs::tts::{KokoroTts, KokoroTtsConfig};

                    let tts_config = KokoroTtsConfig {
                        model,
                        voices,
                        tokens,
                        data_dir,
                        length_scale: speed,
                        ..Default::default()
                    };

                    let mut tts = KokoroTts::new(tts_config);
                    tracing::info!("sherpa kokoro TTS loaded on inference thread");

                    while let Ok(req) = request_rx.recv() {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            match tts.create(&req.text, speaker_id, speed) {
                                Ok(audio) => Ok((audio.samples, audio.sample_rate)),
                                Err(e) => Err(format!("TTS synthesis failed: {e}")),
                            }
                        }));

                        let result = match result {
                            Ok(r) => r,
                            Err(_) => Err("sherpa TTS panicked".into()),
                        };
                        let _ = req.reply.send(result);
                    }
                })
                .map_err(|e| {
                    TtsError::Connection(format!("failed to spawn sherpa TTS thread: {e}"))
                })?;

            info!("sherpa kokoro TTS thread started");
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

impl TextToSpeech for SherpaTts {
    async fn synthesize(
        &self,
        text: &str,
        audio_tx: mpsc::Sender<AudioFrame>,
    ) -> Result<(), TtsError> {
        info!(text_len = text.len(), "sherpa kokoro synthesizing");

        #[cfg(feature = "sherpa")]
        {
            let (reply_tx, reply_rx) = oneshot::channel();
            self.request_tx
                .send(TtsRequest {
                    text: text.to_string(),
                    reply: reply_tx,
                })
                .map_err(|_| TtsError::Synthesis("sherpa TTS thread gone".into()))?;

            let (samples, sample_rate) = reply_rx
                .await
                .map_err(|_| TtsError::Synthesis("sherpa TTS thread dropped reply".into()))?
                .map_err(TtsError::Synthesis)?;

            info!(
                samples = samples.len(),
                sample_rate,
                duration_ms = samples.len() as u64 * 1000 / sample_rate as u64,
                "sherpa kokoro synthesis complete, resampling"
            );

            // Resample to 16kHz if needed (Kokoro outputs at 24kHz)
            let samples_16k = if sample_rate != 16_000 {
                let mut resampler = crate::resample::Resampler::new_24k_to_16k();
                resampler.process(&samples)
            } else {
                samples
            };

            // Convert f32 → i16 PCM and chunk into 20ms frames
            let samples_i16: Vec<i16> = samples_16k
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();

            let frame_samples = 320;
            for chunk in samples_i16.chunks(frame_samples) {
                let pcm_bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
                let frame = AudioFrame::new(
                    pronghorn_audio::AudioFormat::SPEECH,
                    bytes::Bytes::from(pcm_bytes),
                );
                if audio_tx.send(frame).await.is_err() {
                    break;
                }
            }
        }

        #[cfg(not(feature = "sherpa"))]
        {
            let _ = text;
            let _ = audio_tx;
        }

        Ok(())
    }
}
