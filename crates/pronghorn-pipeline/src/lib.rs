pub mod config;
pub mod error;
pub mod intent;
pub mod kokoro;
pub mod ollama;
pub mod resample;
pub mod stt;
pub mod tts;
pub mod whisper;

pub use config::{
    IntentBackend, IntentConfig, KokoroConfig, OllamaConfig, PipelineConfig, SttBackend, SttConfig,
    TtsBackend, TtsConfig, WhisperConfig,
};
pub use error::PipelineError;
pub use intent::{EchoIntent, IntentAction, IntentError, IntentProcessor, IntentResponse};
pub use kokoro::KokoroTts;
pub use ollama::OllamaIntent;
pub use resample::Resampler;
pub use stt::{EchoStt, SpeechToText, SttError, Transcript};
pub use tts::{EchoTts, TextToSpeech, TtsError};
pub use whisper::WhisperStt;

// ── Enum dispatch ───────────────────────────────────────────────────
//
// RPITIT traits aren't object-safe, so we can't use Box<dyn SpeechToText>.
// Instead, enum dispatch: one variant per backend, delegates to the inner impl.
// Zero-cost (no heap allocation), works with RPITIT.

use pronghorn_audio::AudioFrame;
use tokio::sync::mpsc;

/// Runtime-dispatched STT backend.
pub enum SttDispatch {
    Echo(EchoStt),
    Whisper(WhisperStt),
}

impl SpeechToText for SttDispatch {
    async fn transcribe(
        &self,
        audio_rx: mpsc::Receiver<AudioFrame>,
        transcript_tx: mpsc::Sender<Transcript>,
    ) -> Result<(), SttError> {
        match self {
            Self::Echo(s) => s.transcribe(audio_rx, transcript_tx).await,
            Self::Whisper(s) => s.transcribe(audio_rx, transcript_tx).await,
        }
    }
}

/// Runtime-dispatched TTS backend.
pub enum TtsDispatch {
    Echo(EchoTts),
    Kokoro(KokoroTts),
}

impl TextToSpeech for TtsDispatch {
    async fn synthesize(
        &self,
        text: &str,
        audio_tx: mpsc::Sender<AudioFrame>,
    ) -> Result<(), TtsError> {
        match self {
            Self::Echo(t) => t.synthesize(text, audio_tx).await,
            Self::Kokoro(t) => t.synthesize(text, audio_tx).await,
        }
    }
}

/// Runtime-dispatched intent backend.
pub enum IntentDispatch {
    Echo(EchoIntent),
    Ollama(OllamaIntent),
}

impl IntentProcessor for IntentDispatch {
    async fn process(&self, transcript: &str) -> Result<IntentResponse, IntentError> {
        match self {
            Self::Echo(i) => i.process(transcript).await,
            Self::Ollama(i) => i.process(transcript).await,
        }
    }
}

/// Create an STT backend from config.
pub fn create_stt(config: &SttConfig) -> Result<SttDispatch, SttError> {
    match config.backend {
        SttBackend::Echo => Ok(SttDispatch::Echo(EchoStt)),
        SttBackend::Whisper => {
            let stt = WhisperStt::new(&config.whisper)?;
            Ok(SttDispatch::Whisper(stt))
        }
    }
}

/// Create a TTS backend from config.
pub fn create_tts(config: &TtsConfig) -> Result<TtsDispatch, TtsError> {
    match config.backend {
        TtsBackend::Echo => Ok(TtsDispatch::Echo(EchoTts)),
        TtsBackend::Kokoro => {
            let tts = KokoroTts::new(&config.kokoro)?;
            Ok(TtsDispatch::Kokoro(tts))
        }
    }
}

/// Create an intent backend from config.
pub fn create_intent(config: &IntentConfig) -> Result<IntentDispatch, IntentError> {
    match config.backend {
        IntentBackend::Echo => Ok(IntentDispatch::Echo(EchoIntent)),
        IntentBackend::Ollama => {
            let intent = OllamaIntent::new(&config.ollama)?;
            Ok(IntentDispatch::Ollama(intent))
        }
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use pronghorn_audio::{AudioFormat, AudioFrame};
    use tokio::sync::mpsc;

    use super::*;

    fn silence_frame() -> AudioFrame {
        AudioFrame::new(AudioFormat::SPEECH, Bytes::from(vec![0u8; 640]))
    }

    /// End-to-end pipeline test: EchoStt → EchoIntent → EchoTts
    #[tokio::test]
    async fn echo_pipeline_end_to_end() {
        let (audio_tx, audio_rx) = mpsc::channel::<AudioFrame>(16);
        let (transcript_tx, mut transcript_rx) = mpsc::channel::<Transcript>(4);

        let stt = EchoStt;
        let stt_handle = tokio::spawn(async move { stt.transcribe(audio_rx, transcript_tx).await });

        for _ in 0..10 {
            audio_tx.send(silence_frame()).await.unwrap();
        }
        drop(audio_tx);

        stt_handle.await.unwrap().unwrap();

        let transcript = transcript_rx.recv().await.unwrap();
        assert!(transcript.is_final);
        assert_eq!(transcript.text, "received 10 frames");

        let intent = EchoIntent;
        let response = intent.process(&transcript.text).await.unwrap();
        assert_eq!(response.reply_text, "You said: received 10 frames");

        let (tts_audio_tx, mut tts_audio_rx) = mpsc::channel::<AudioFrame>(64);
        let tts = EchoTts;
        tts.synthesize(&response.reply_text, tts_audio_tx)
            .await
            .unwrap();

        let mut tts_frame_count = 0;
        while tts_audio_rx.try_recv().is_ok() {
            tts_frame_count += 1;
        }
        assert_eq!(tts_frame_count, response.reply_text.len());
    }

    /// Test enum dispatch with echo backends via factory functions.
    #[tokio::test]
    async fn dispatch_echo_round_trip() {
        let stt_config = SttConfig::default();
        let tts_config = TtsConfig::default();
        let intent_config = IntentConfig::default();

        let stt = create_stt(&stt_config).unwrap();
        let tts = create_tts(&tts_config).unwrap();
        let intent = create_intent(&intent_config).unwrap();

        // STT
        let (audio_tx, audio_rx) = mpsc::channel(16);
        let (transcript_tx, mut transcript_rx) = mpsc::channel(4);
        let stt_handle = tokio::spawn(async move { stt.transcribe(audio_rx, transcript_tx).await });

        for _ in 0..3 {
            audio_tx.send(silence_frame()).await.unwrap();
        }
        drop(audio_tx);
        stt_handle.await.unwrap().unwrap();

        let transcript = transcript_rx.recv().await.unwrap();
        assert!(transcript.is_final);

        // Intent
        let response = intent.process(&transcript.text).await.unwrap();

        // TTS
        let (tts_tx, mut tts_rx) = mpsc::channel(64);
        tts.synthesize(&response.reply_text, tts_tx).await.unwrap();

        let mut count = 0;
        while tts_rx.try_recv().is_ok() {
            count += 1;
        }
        assert!(count > 0);
    }
}
