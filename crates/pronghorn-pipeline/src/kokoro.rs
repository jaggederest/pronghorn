use bytes::Bytes;
use pronghorn_audio::{AudioFormat, AudioFrame};
use tokio::sync::mpsc;
#[cfg(feature = "kokoro")]
use tokio::sync::oneshot;
use tracing::info;

use crate::config::KokoroConfig;
use crate::resample::Resampler;
use crate::tts::{TextToSpeech, TtsError};

/// Request sent to the Kokoro inference thread.
#[cfg(feature = "kokoro")]
struct SynthesisRequest {
    text: String,
    reply: oneshot::Sender<Result<Vec<f32>, String>>,
}

/// Kokoro TTS backend using kokoroxide (ONNX inference + espeak-ng phonemization).
///
/// The kokoroxide model isn't Send, so it lives on a dedicated background thread.
/// Synthesis requests are sent via channel. Output is 24kHz f32 audio, resampled
/// to 16kHz i16 for our wire format.
pub struct KokoroTts {
    #[cfg(feature = "kokoro")]
    request_tx: std::sync::mpsc::Sender<SynthesisRequest>,
    #[cfg(not(feature = "kokoro"))]
    _config: KokoroConfig,
}

// SAFETY: The kokoroxide model lives on its own thread. KokoroTts only holds
// a channel sender (which is Send+Sync). No direct access to the model.
unsafe impl Send for KokoroTts {}
unsafe impl Sync for KokoroTts {}

impl KokoroTts {
    pub fn new(config: &KokoroConfig) -> Result<Self, TtsError> {
        #[cfg(feature = "kokoro")]
        {
            if !config.model_path.exists() {
                return Err(TtsError::Connection(format!(
                    "model file not found: {}",
                    config.model_path.display()
                )));
            }
            if !config.tokenizer_path.exists() {
                return Err(TtsError::Connection(format!(
                    "tokenizer not found: {}",
                    config.tokenizer_path.display()
                )));
            }

            let voice_file = config.voices_path.join(format!("{}.bin", config.voice));
            if !voice_file.exists() {
                return Err(TtsError::Connection(format!(
                    "voice file not found: {}",
                    voice_file.display()
                )));
            }

            info!(
                model = %config.model_path.display(),
                voice = %config.voice,
                "loading kokoro model"
            );

            let (request_tx, request_rx) = std::sync::mpsc::channel::<SynthesisRequest>();

            let model_path = config.model_path.to_string_lossy().to_string();
            let tokenizer_path = config.tokenizer_path.to_string_lossy().to_string();
            let voice_file_str = voice_file.to_string_lossy().to_string();

            std::thread::Builder::new()
                .name("kokoro-inference".into())
                .spawn(move || {
                    // Load model
                    let tts_config = kokoroxide::TTSConfig::new(&model_path, &tokenizer_path);
                    let mut tts = match kokoroxide::KokoroTTS::with_config(tts_config) {
                        Ok(t) => t,
                        Err(e) => {
                            tracing::error!("failed to load kokoro model: {e}");
                            return;
                        }
                    };

                    // Load voice
                    let voice = match kokoroxide::load_voice_style(&voice_file_str) {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::error!("failed to load voice: {e}");
                            return;
                        }
                    };

                    tracing::info!("kokoro model loaded on inference thread");

                    while let Ok(req) = request_rx.recv() {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            match tts.speak(&req.text, &voice) {
                                Ok(audio) => Ok(audio.samples),
                                Err(e) => Err(format!("synthesis failed: {e}")),
                            }
                        }));

                        let result = match result {
                            Ok(r) => r,
                            Err(_) => Err("kokoro inference panicked".into()),
                        };
                        let _ = req.reply.send(result);
                    }
                })
                .map_err(|e| TtsError::Connection(format!("failed to spawn kokoro thread: {e}")))?;

            info!("kokoro inference thread started");
            Ok(Self { request_tx })
        }

        #[cfg(not(feature = "kokoro"))]
        {
            Ok(Self {
                _config: config.clone(),
            })
        }
    }
}

impl TextToSpeech for KokoroTts {
    async fn synthesize(
        &self,
        text: &str,
        audio_tx: mpsc::Sender<AudioFrame>,
    ) -> Result<(), TtsError> {
        let sentences = crate::sherpa_tts::split_sentences(text);
        info!(
            text_len = text.len(),
            sentences = sentences.len(),
            "kokoro synthesizing (sentence-streaming)"
        );

        #[cfg(feature = "kokoro")]
        {
            for (i, sentence) in sentences.iter().enumerate() {
                let (reply_tx, reply_rx) = oneshot::channel();
                self.request_tx
                    .send(SynthesisRequest {
                        text: sentence.to_string(),
                        reply: reply_tx,
                    })
                    .map_err(|_| TtsError::Synthesis("kokoro inference thread gone".into()))?;

                let samples_24k = reply_rx
                    .await
                    .map_err(|_| {
                        TtsError::Synthesis("kokoro inference thread dropped reply".into())
                    })?
                    .map_err(TtsError::Synthesis)?;

                info!(
                    sentence = i + 1,
                    total = sentences.len(),
                    samples_24k = samples_24k.len(),
                    duration_ms = samples_24k.len() as u64 * 1000 / 24_000,
                    "sentence synthesized, streaming frames"
                );

                // Resample 24kHz → 16kHz
                let mut resampler = Resampler::new_24k_to_16k();
                let samples_16k = resampler.process(&samples_24k);

                // Convert f32 → i16 PCM and chunk into 20ms frames
                let samples_i16: Vec<i16> = samples_16k
                    .iter()
                    .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                    .collect();

                let frame_samples = 320;
                let mut aborted = false;
                for chunk in samples_i16.chunks(frame_samples) {
                    let pcm_bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
                    let frame = AudioFrame::new(AudioFormat::SPEECH, Bytes::from(pcm_bytes));
                    if audio_tx.send(frame).await.is_err() {
                        aborted = true;
                        break;
                    }
                }
                if aborted {
                    break;
                }
            }
        }

        #[cfg(not(feature = "kokoro"))]
        {
            // Stub: generate silence proportional to text length
            let stub_duration_samples = text.len() * 480;
            let samples_24k: Vec<f32> = vec![0.0; stub_duration_samples];

            let mut resampler = Resampler::new_24k_to_16k();
            let samples_16k = resampler.process(&samples_24k);

            let samples_i16: Vec<i16> = samples_16k
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();

            let frame_samples = 320;
            for chunk in samples_i16.chunks(frame_samples) {
                let pcm_bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
                let frame = AudioFrame::new(AudioFormat::SPEECH, Bytes::from(pcm_bytes));
                if audio_tx.send(frame).await.is_err() {
                    break;
                }
            }
        }

        Ok(())
    }
}
