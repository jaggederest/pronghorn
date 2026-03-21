pub mod config;
pub mod error;
pub mod intent;
pub mod stt;
pub mod tts;

pub use config::{
    IntentBackend, IntentConfig, PipelineConfig, SttBackend, SttConfig, TtsBackend, TtsConfig,
};
pub use error::PipelineError;
pub use intent::{EchoIntent, IntentAction, IntentError, IntentProcessor, IntentResponse};
pub use stt::{EchoStt, SpeechToText, SttError, Transcript};
pub use tts::{EchoTts, TextToSpeech, TtsError};

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
        // STT channels
        let (audio_tx, audio_rx) = mpsc::channel::<AudioFrame>(16);
        let (transcript_tx, mut transcript_rx) = mpsc::channel::<Transcript>(4);

        // Run STT
        let stt = EchoStt;
        let stt_handle = tokio::spawn(async move { stt.transcribe(audio_rx, transcript_tx).await });

        // Feed 10 frames then close
        for _ in 0..10 {
            audio_tx.send(silence_frame()).await.unwrap();
        }
        drop(audio_tx);

        stt_handle.await.unwrap().unwrap();

        // Get transcript
        let transcript = transcript_rx.recv().await.unwrap();
        assert!(transcript.is_final);
        assert_eq!(transcript.text, "received 10 frames");

        // Run intent
        let intent = EchoIntent;
        let response = intent.process(&transcript.text).await.unwrap();
        assert_eq!(response.reply_text, "You said: received 10 frames");

        // Run TTS
        let (tts_audio_tx, mut tts_audio_rx) = mpsc::channel::<AudioFrame>(64);
        let tts = EchoTts;
        tts.synthesize(&response.reply_text, tts_audio_tx)
            .await
            .unwrap();

        // Verify TTS produced frames
        let mut tts_frame_count = 0;
        while tts_audio_rx.try_recv().is_ok() {
            tts_frame_count += 1;
        }
        assert_eq!(tts_frame_count, response.reply_text.len());
    }
}
