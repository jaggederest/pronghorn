use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use pronghorn_audio::{AudioFormat, AudioFrame};
use pronghorn_pipeline::{
    IntentDispatch, IntentProcessor, SpeechToText, SttDispatch, TextToSpeech, Transcript,
    TtsDispatch,
};
use pronghorn_wire::{AudioData, Control, ControlType, JitterBuffer, Packet, Transport};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, error, info};

/// Handle to an active session's pipeline. Held by the event loop.
pub struct SessionHandle {
    /// Send audio packets from the satellite into the pipeline.
    /// Drop this to signal end-of-stream (StopListening).
    pub audio_tx: mpsc::Sender<AudioData>,
    /// The orchestrator task — join to detect completion.
    pub task: JoinHandle<()>,
}

/// Spawn a per-session pipeline orchestrator.
///
/// Wires: AudioData → JitterBuffer → STT → Intent → TTS → send back
pub fn spawn_orchestrator(
    session_id: u32,
    stt: Arc<SttDispatch>,
    tts: Arc<TtsDispatch>,
    intent: Arc<IntentDispatch>,
    transport: Arc<Transport>,
    remote_addr: SocketAddr,
    jitter_delay: u16,
) -> SessionHandle {
    let (audio_data_tx, audio_data_rx) = mpsc::channel::<AudioData>(64);

    let task = tokio::spawn(async move {
        if let Err(e) = run_pipeline(
            session_id,
            jitter_delay,
            audio_data_rx,
            stt,
            tts,
            intent,
            transport,
            remote_addr,
        )
        .await
        {
            error!(session_id, %e, "pipeline error");
        }
    });

    SessionHandle {
        audio_tx: audio_data_tx,
        task,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_pipeline(
    session_id: u32,
    jitter_delay: u16,
    audio_data_rx: mpsc::Receiver<AudioData>,
    stt: Arc<SttDispatch>,
    tts: Arc<TtsDispatch>,
    intent: Arc<IntentDispatch>,
    transport: Arc<Transport>,
    remote_addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Channel: jitter buffer → STT
    let (audio_frame_tx, audio_frame_rx) = mpsc::channel::<AudioFrame>(64);
    // Channel: STT → orchestrator
    let (transcript_tx, mut transcript_rx) = mpsc::channel::<Transcript>(8);

    // Task: JitterBuffer → AudioFrame
    let jitter_handle = tokio::spawn(jitter_to_frames(
        audio_data_rx,
        audio_frame_tx,
        jitter_delay,
    ));

    // Task: STT (shared backend via Arc)
    let stt_clone = Arc::clone(&stt);
    let stt_handle =
        tokio::spawn(async move { stt_clone.transcribe(audio_frame_rx, transcript_tx).await });

    // Wait for final transcript
    let mut final_text = String::new();
    while let Some(transcript) = transcript_rx.recv().await {
        debug!(session_id, text = %transcript.text, is_final = transcript.is_final, "transcript");
        if transcript.is_final {
            final_text = transcript.text;
            break;
        }
    }

    // Wait for STT and jitter tasks to finish
    let _ = stt_handle.await;
    let _ = jitter_handle.await;

    if final_text.is_empty() {
        debug!(session_id, "empty transcript, skipping TTS");
        return Ok(());
    }

    // Intent processing
    let response = intent.process(&final_text).await?;
    info!(session_id, reply = %response.reply_text, "intent response");

    // Send StartSpeaking
    transport
        .send_to(
            &Packet::Control(Control {
                session_id,
                control_type: ControlType::StartSpeaking,
                payload: Bytes::new(),
            }),
            remote_addr,
        )
        .await?;

    // TTS: synthesize and stream back (shared backend via Arc)
    let (tts_audio_tx, mut tts_audio_rx) = mpsc::channel::<AudioFrame>(64);
    let tts_clone = Arc::clone(&tts);
    let reply_text = response.reply_text.clone();
    let tts_handle =
        tokio::spawn(async move { tts_clone.synthesize(&reply_text, tts_audio_tx).await });

    let mut seq = 0u16;
    while let Some(frame) = tts_audio_rx.recv().await {
        let pkt = Packet::Audio(AudioData {
            session_id,
            sequence: seq,
            flags: 0,
            timestamp: seq as u32 * 320,
            payload: frame.samples,
        });
        transport.send_to(&pkt, remote_addr).await?;
        seq = seq.wrapping_add(1);
    }

    let _ = tts_handle.await;

    // Send StopSpeaking
    transport
        .send_to(
            &Packet::Control(Control {
                session_id,
                control_type: ControlType::StopSpeaking,
                payload: Bytes::new(),
            }),
            remote_addr,
        )
        .await?;

    info!(session_id, tts_frames = seq, "session complete");
    Ok(())
}

/// Adapter: receives AudioData packets, reorders via JitterBuffer, outputs AudioFrames.
async fn jitter_to_frames(
    mut audio_rx: mpsc::Receiver<AudioData>,
    frame_tx: mpsc::Sender<AudioFrame>,
    playout_delay: u16,
) {
    let mut jb = JitterBuffer::new(playout_delay);
    while let Some(data) = audio_rx.recv().await {
        jb.push(data);
        while let Some(d) = jb.pop() {
            if frame_tx
                .send(AudioFrame::new(AudioFormat::SPEECH, d.payload))
                .await
                .is_err()
            {
                return; // downstream closed
            }
        }
    }
    // audio_rx closed → drop frame_tx → STT sees channel close
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn jitter_to_frames_orders_packets() {
        let (audio_tx, audio_rx) = mpsc::channel(16);
        let (frame_tx, mut frame_rx) = mpsc::channel(16);

        let handle = tokio::spawn(jitter_to_frames(audio_rx, frame_tx, 2));

        // Send packets out of order
        audio_tx
            .send(AudioData {
                session_id: 1,
                sequence: 0,
                flags: 0,
                timestamp: 0,
                payload: Bytes::from(vec![0u8; 640]),
            })
            .await
            .unwrap();
        audio_tx
            .send(AudioData {
                session_id: 1,
                sequence: 2,
                flags: 0,
                timestamp: 640,
                payload: Bytes::from(vec![2u8; 640]),
            })
            .await
            .unwrap();
        audio_tx
            .send(AudioData {
                session_id: 1,
                sequence: 1,
                flags: 0,
                timestamp: 320,
                payload: Bytes::from(vec![1u8; 640]),
            })
            .await
            .unwrap();

        drop(audio_tx);
        handle.await.unwrap();

        // Should come out in order: 0, 1, 2
        let f0 = frame_rx.recv().await.unwrap();
        assert_eq!(f0.samples[0], 0);
        let f1 = frame_rx.recv().await.unwrap();
        assert_eq!(f1.samples[0], 1);
        let f2 = frame_rx.recv().await.unwrap();
        assert_eq!(f2.samples[0], 2);
        assert!(frame_rx.recv().await.is_none());
    }
}
