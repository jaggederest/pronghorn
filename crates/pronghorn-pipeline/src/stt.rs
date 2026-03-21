use std::future::Future;

use pronghorn_audio::AudioFrame;
use thiserror::Error;
use tokio::sync::mpsc;

/// A partial or final transcript from the STT stage.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub is_final: bool,
}

#[derive(Debug, Error)]
pub enum SttError {
    #[error("STT connection failed: {0}")]
    Connection(String),

    #[error("STT stream error: {0}")]
    Stream(String),

    #[error("STT backend not available: {0}")]
    BackendNotAvailable(String),
}

/// Speech-to-text pipeline stage.
///
/// Consumes a stream of audio frames and produces transcripts.
/// Streaming: processes frames as they arrive, doesn't wait for the full utterance.
pub trait SpeechToText: Send + Sync {
    /// Transcribe audio frames into text.
    ///
    /// Reads from `audio_rx` until the channel closes (sender dropped = end of speech).
    /// Sends partial and final transcripts to `transcript_tx`.
    fn transcribe(
        &self,
        audio_rx: mpsc::Receiver<AudioFrame>,
        transcript_tx: mpsc::Sender<Transcript>,
    ) -> impl Future<Output = Result<(), SttError>> + Send;
}

/// Echo STT backend for development and testing.
///
/// Drains all audio frames and emits a single final transcript
/// reporting how many frames were received.
pub struct EchoStt;

impl SpeechToText for EchoStt {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioFrame>,
        transcript_tx: mpsc::Sender<Transcript>,
    ) -> Result<(), SttError> {
        let mut frame_count = 0u32;
        while let Some(_frame) = audio_rx.recv().await {
            frame_count += 1;
        }
        let _ = transcript_tx
            .send(Transcript {
                text: format!("received {frame_count} frames"),
                is_final: true,
            })
            .await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use pronghorn_audio::AudioFormat;

    use super::*;

    fn silence_frame() -> AudioFrame {
        AudioFrame::new(AudioFormat::SPEECH, Bytes::from(vec![0u8; 640]))
    }

    #[tokio::test]
    async fn echo_stt_counts_frames() {
        let (audio_tx, audio_rx) = mpsc::channel(16);
        let (transcript_tx, mut transcript_rx) = mpsc::channel(4);

        let stt = EchoStt;
        let handle = tokio::spawn(async move { stt.transcribe(audio_rx, transcript_tx).await });

        // Send 5 frames then close
        for _ in 0..5 {
            audio_tx.send(silence_frame()).await.unwrap();
        }
        drop(audio_tx);

        handle.await.unwrap().unwrap();

        let transcript = transcript_rx.recv().await.unwrap();
        assert!(transcript.is_final);
        assert_eq!(transcript.text, "received 5 frames");
    }

    #[tokio::test]
    async fn echo_stt_handles_empty_stream() {
        let (_audio_tx, audio_rx) = mpsc::channel(16);
        let (transcript_tx, mut transcript_rx) = mpsc::channel(4);

        let stt = EchoStt;
        // Drop sender immediately
        drop(_audio_tx);
        stt.transcribe(audio_rx, transcript_tx).await.unwrap();

        let transcript = transcript_rx.recv().await.unwrap();
        assert_eq!(transcript.text, "received 0 frames");
    }
}
