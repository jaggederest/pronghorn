use std::future::Future;

use bytes::Bytes;
use pronghorn_audio::{AudioFormat, AudioFrame};
use thiserror::Error;
use tokio::sync::mpsc;

#[derive(Debug, Error)]
pub enum TtsError {
    #[error("TTS connection failed: {0}")]
    Connection(String),

    #[error("TTS synthesis error: {0}")]
    Synthesis(String),

    #[error("TTS backend not available: {0}")]
    BackendNotAvailable(String),
}

/// Text-to-speech pipeline stage.
///
/// Takes text and produces a stream of audio frames.
/// Streaming: sends frames as they're generated, doesn't buffer the whole response.
pub trait TextToSpeech: Send + Sync {
    /// Synthesize text into audio frames.
    ///
    /// Sends audio frames to `audio_tx` as they're generated.
    /// Returns when synthesis is complete.
    fn synthesize(
        &self,
        text: &str,
        audio_tx: mpsc::Sender<AudioFrame>,
    ) -> impl Future<Output = Result<(), TtsError>> + Send;
}

/// Echo TTS backend for development and testing.
///
/// Generates silence frames proportional to the text length.
/// Roughly 1 frame (20ms) per character — enough to verify the pipeline flows.
pub struct EchoTts;

impl TextToSpeech for EchoTts {
    async fn synthesize(
        &self,
        text: &str,
        audio_tx: mpsc::Sender<AudioFrame>,
    ) -> Result<(), TtsError> {
        // Generate ~1 frame per character (minimum 1 frame)
        let frame_count = text.len().max(1);
        for _ in 0..frame_count {
            let frame = AudioFrame::new(AudioFormat::SPEECH, Bytes::from(vec![0u8; 640]));
            if audio_tx.send(frame).await.is_err() {
                break; // receiver dropped
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echo_tts_generates_frames() {
        let (audio_tx, mut audio_rx) = mpsc::channel(64);

        let tts = EchoTts;
        tts.synthesize("hello", audio_tx).await.unwrap();

        let mut count = 0;
        while audio_rx.try_recv().is_ok() {
            count += 1;
        }
        assert_eq!(count, 5); // "hello" = 5 chars = 5 frames
    }

    #[tokio::test]
    async fn echo_tts_empty_text_produces_one_frame() {
        let (audio_tx, mut audio_rx) = mpsc::channel(64);

        let tts = EchoTts;
        tts.synthesize("", audio_tx).await.unwrap();

        assert!(audio_rx.try_recv().is_ok());
        assert!(audio_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn echo_tts_handles_dropped_receiver() {
        let (audio_tx, audio_rx) = mpsc::channel(1);
        drop(audio_rx);

        let tts = EchoTts;
        // Should not panic, just return Ok
        tts.synthesize("hello world", audio_tx).await.unwrap();
    }
}
