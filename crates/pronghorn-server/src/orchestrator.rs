use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use pronghorn_audio::{AudioFormat, AudioFrame};
use pronghorn_pipeline::{
    IntentDispatch, IntentProcessor, SpeechToText, SttDispatch, TextToSpeech, Transcript,
    TtsDispatch, VadConfig,
};
use pronghorn_wire::{AudioData, Control, ControlType, JitterBuffer, Packet, Transport};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Handle to an active session's pipeline. Held by the event loop.
pub struct SessionHandle {
    /// Send audio packets from the satellite into the pipeline.
    /// Drop this to signal end-of-stream (satellite timeout fallback).
    pub audio_tx: mpsc::Sender<AudioData>,
    /// The orchestrator task — join to detect completion.
    pub task: JoinHandle<()>,
}

/// Spawn a per-session pipeline orchestrator.
///
/// Wires: AudioData → JitterBuffer (+VAD) → STT → Intent → TTS → send back
#[allow(clippy::too_many_arguments)]
pub fn spawn_orchestrator(
    session_id: u32,
    stt: Arc<SttDispatch>,
    tts: Arc<TtsDispatch>,
    intent: Arc<IntentDispatch>,
    transport: Arc<Transport>,
    remote_addr: SocketAddr,
    jitter_delay: u16,
    vad_config: VadConfig,
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
            vad_config,
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
    vad_config: VadConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Channel: jitter buffer → STT
    let (audio_frame_tx, audio_frame_rx) = mpsc::channel::<AudioFrame>(64);
    // Channel: STT → orchestrator
    let (transcript_tx, mut transcript_rx) = mpsc::channel::<Transcript>(8);

    // Create Silero VAD for this session (if enabled and sherpa feature is on)
    let (vad_done_tx, vad_done_rx) = oneshot::channel::<()>();

    #[cfg(feature = "sherpa")]
    let vad = if vad_config.enabled {
        match pronghorn_pipeline::vad::create_vad(&vad_config) {
            Ok(v) => Some(v),
            Err(e) => {
                warn!(session_id, %e, "failed to create VAD, falling back to no-VAD mode");
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "sherpa"))]
    let vad: Option<()> = {
        let _ = &vad_config;
        None
    };

    // Task: JitterBuffer → AudioFrame (with optional VAD endpoint detection)
    #[cfg(feature = "sherpa")]
    let jitter_handle = if let Some(vad_instance) = vad {
        tokio::spawn(jitter_to_frames_with_vad(
            audio_data_rx,
            audio_frame_tx,
            jitter_delay,
            vad_instance,
            vad_done_tx,
        ))
    } else {
        // No VAD: behave as before (satellite timeout drives StopListening)
        drop(vad_done_tx);
        tokio::spawn(jitter_to_frames(
            audio_data_rx,
            audio_frame_tx,
            jitter_delay,
        ))
    };

    #[cfg(not(feature = "sherpa"))]
    let jitter_handle = {
        drop(vad_done_tx);
        let _ = vad;
        tokio::spawn(jitter_to_frames(
            audio_data_rx,
            audio_frame_tx,
            jitter_delay,
        ))
    };

    // Fire-and-forget task: send StopListening to satellite when VAD detects speech end
    let transport_vad = Arc::clone(&transport);
    tokio::spawn(async move {
        if vad_done_rx.await.is_ok() {
            info!(
                session_id,
                "VAD: speech ended, sending StopListening to satellite"
            );
            if let Err(e) = transport_vad
                .send_to(
                    &Packet::Control(Control {
                        session_id,
                        control_type: ControlType::StopListening,
                        payload: Bytes::new(),
                    }),
                    remote_addr,
                )
                .await
            {
                warn!(session_id, %e, "failed to send StopListening to satellite");
            }
        }
        // If vad_done_rx errors, VAD never fired (satellite timeout handled it) — no-op
    });

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

/// Adapter with Silero VAD: receives AudioData packets, reorders via JitterBuffer,
/// outputs AudioFrames, and detects end-of-speech via Silero VAD.
///
/// When VAD detects speech→silence, signals via `vad_done_tx` and stops forwarding.
/// The orchestrator sends StopListening to the satellite in response.
#[cfg(feature = "sherpa")]
async fn jitter_to_frames_with_vad(
    mut audio_rx: mpsc::Receiver<AudioData>,
    frame_tx: mpsc::Sender<AudioFrame>,
    playout_delay: u16,
    mut vad: pronghorn_pipeline::vad::SileroVad,
    vad_done_tx: oneshot::Sender<()>,
) {
    use pronghorn_pipeline::vad::i16_bytes_to_f32;

    let mut jb = JitterBuffer::new(playout_delay);
    let max_skip_wait = 5u32;
    let mut stall_count = 0u32;

    while let Some(data) = audio_rx.recv().await {
        jb.push(data);

        loop {
            match jb.pop() {
                Some(d) => {
                    stall_count = 0;

                    // Feed VAD with f32 samples
                    let samples = i16_bytes_to_f32(&d.payload);
                    vad.accept_waveform(samples);

                    // Forward frame to STT
                    if frame_tx
                        .send(AudioFrame::new(AudioFormat::SPEECH, d.payload))
                        .await
                        .is_err()
                    {
                        return; // downstream closed
                    }

                    // Check if VAD detected end of speech (completed speech segment)
                    if !vad.is_empty() {
                        info!("VAD: end of speech detected");
                        vad.pop(); // clear the segment buffer
                        let _ = vad_done_tx.send(());
                        // Drop frame_tx by returning — STT sees EOF
                        return;
                    }
                }
                None => {
                    if jb.is_playing() && jb.buffered() > 0 {
                        stall_count += 1;
                        if stall_count >= max_skip_wait {
                            debug!(
                                next_seq = ?jb.next_seq(),
                                buffered = jb.buffered(),
                                "skipping lost packet"
                            );
                            jb.skip();
                            stall_count = 0;
                            continue;
                        }
                    }
                    break;
                }
            }
        }
    }

    // Channel closed (satellite timeout case): flush remaining buffered packets
    flush_jitter_buffer(&mut jb, &frame_tx).await;
    // vad_done_tx dropped without sending — no StopListening needed (satellite already sent it)
}

/// Adapter without VAD: receives AudioData packets, reorders via JitterBuffer, outputs AudioFrames.
///
/// Used when VAD is disabled or sherpa feature is off. Satellite drives StopListening.
async fn jitter_to_frames(
    mut audio_rx: mpsc::Receiver<AudioData>,
    frame_tx: mpsc::Sender<AudioFrame>,
    playout_delay: u16,
) {
    let mut jb = JitterBuffer::new(playout_delay);
    let max_skip_wait = 5u32;
    let mut stall_count = 0u32;

    while let Some(data) = audio_rx.recv().await {
        jb.push(data);

        loop {
            match jb.pop() {
                Some(d) => {
                    stall_count = 0;
                    if frame_tx
                        .send(AudioFrame::new(AudioFormat::SPEECH, d.payload))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                None => {
                    if jb.is_playing() && jb.buffered() > 0 {
                        stall_count += 1;
                        if stall_count >= max_skip_wait {
                            debug!(
                                next_seq = ?jb.next_seq(),
                                buffered = jb.buffered(),
                                "skipping lost packet"
                            );
                            jb.skip();
                            stall_count = 0;
                            continue;
                        }
                    }
                    break;
                }
            }
        }
    }

    flush_jitter_buffer(&mut jb, &frame_tx).await;
}

/// Flush all remaining buffered packets from the jitter buffer, skipping gaps.
async fn flush_jitter_buffer(jb: &mut JitterBuffer, frame_tx: &mpsc::Sender<AudioFrame>) {
    loop {
        match jb.pop() {
            Some(d) => {
                if frame_tx
                    .send(AudioFrame::new(AudioFormat::SPEECH, d.payload))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            None => {
                if jb.buffered() > 0 {
                    jb.skip();
                    continue;
                }
                break;
            }
        }
    }
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
