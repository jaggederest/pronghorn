use std::time::Duration;

use bytes::Bytes;
use pronghorn_audio::{AudioFormat, AudioFrame, RingBuffer};
use pronghorn_wake::Detection;
use pronghorn_wire::{
    AudioData, Control, ControlType, Hello, Keepalive, Packet, Transport, audio_flags,
};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::config::SatelliteConfig;
use crate::error::SatelliteError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Idle,
    Streaming,
    Receiving,
}

pub async fn run_satellite(
    config: SatelliteConfig,
    mut audio_rx: mpsc::Receiver<AudioFrame>,
    speaker_tx: mpsc::Sender<AudioFrame>,
) -> Result<(), SatelliteError> {
    let server_addr = config.transport.server_address;
    let transport = Transport::bind(([0, 0, 0, 0], 0).into()).await?;
    info!(local = %transport.local_addr()?, server = %server_addr, "satellite transport bound");

    // Handshake
    transport
        .send_to(&Packet::Hello(Hello { client_version: 1 }), server_addr)
        .await?;
    let (pkt, _) = transport.recv_from().await?;
    let session_id = match pkt {
        Packet::Welcome(w) => {
            info!(session_id = w.session_id, "connected to server");
            w.session_id
        }
        _ => return Err(SatelliteError::HandshakeFailed),
    };

    // Pre-roll ring buffer
    let mut ring = RingBuffer::new(config.audio.preroll_frames);

    // Wake word detection thread (if enabled)
    let (wake_frame_tx, mut wake_detect_rx) = spawn_wake_thread(&config)?;

    let mut state = State::Idle;
    let mut sequence = 0u16;
    let mut timestamp = 0u32;
    let mut streaming_since: Option<tokio::time::Instant> = None;
    let streaming_timeout = Duration::from_secs(5); // auto-stop after 5s of streaming

    let keepalive_dur = Duration::from_millis(config.transport.keepalive_interval_ms);
    let mut keepalive_timer = tokio::time::interval(keepalive_dur);
    keepalive_timer.tick().await; // skip first immediate tick

    loop {
        tokio::select! {
            // Branch 1: mic frame from audio source
            frame = audio_rx.recv() => {
                let Some(frame) = frame else {
                    info!("audio source closed, shutting down");
                    break;
                };
                match state {
                    State::Idle => {
                        ring.push(frame.clone());
                        if let Some(ref tx) = wake_frame_tx {
                            match tx.try_send(frame) {
                                Ok(()) => {}
                                Err(e) => {
                                    debug!("wake frame send failed: {e}");
                                }
                            }
                        } else {
                            debug!("no wake frame tx");
                        }
                    }
                    State::Streaming => {
                        // Check streaming timeout (auto-stop, placeholder for VAD)
                        if let Some(since) = streaming_since
                            && since.elapsed() > streaming_timeout
                        {
                            info!("streaming timeout, sending StopListening");
                            transport
                                .send_to(
                                    &Packet::Control(Control {
                                        session_id,
                                        control_type: ControlType::StopListening,
                                        payload: Bytes::new(),
                                    }),
                                    server_addr,
                                )
                                .await?;
                            state = State::Receiving;
                            streaming_since = None;
                            continue;
                        }
                        let pkt = Packet::Audio(AudioData {
                            session_id,
                            sequence,
                            flags: 0,
                            timestamp,
                            payload: frame.samples,
                        });
                        transport.send_to(&pkt, server_addr).await?;
                        sequence = sequence.wrapping_add(1);
                        timestamp = timestamp.wrapping_add(320);
                    }
                    State::Receiving => {
                        // Discard mic frames during TTS playback
                        // (barge-in support is future work)
                    }
                }
            }
            // Branch 2: wake word detected
            Some(detection) = async {
                match &mut wake_detect_rx {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending::<Option<Detection>>().await,
                }
            } => {
                if state == State::Idle {
                    info!(
                        wake_word = %detection.wake_word,
                        score = detection.score,
                        "wake word detected"
                    );

                    // Send StartListening
                    transport.send_to(
                        &Packet::Control(Control {
                            session_id,
                            control_type: ControlType::StartListening,
                            payload: Bytes::new(),
                        }),
                        server_addr,
                    ).await?;

                    // Drain pre-roll
                    sequence = 0;
                    timestamp = 0;
                    let preroll = ring.drain();
                    debug!(preroll_frames = preroll.len(), "draining pre-roll");
                    for frame in preroll {
                        let pkt = Packet::Audio(AudioData {
                            session_id,
                            sequence,
                            flags: audio_flags::PRE_ROLL,
                            timestamp,
                            payload: frame.samples,
                        });
                        transport.send_to(&pkt, server_addr).await?;
                        sequence = sequence.wrapping_add(1);
                        timestamp = timestamp.wrapping_add(320);
                    }

                    state = State::Streaming;
                    streaming_since = Some(tokio::time::Instant::now());
                }
            }
            // Branch 3: packet from server
            result = transport.recv_from() => {
                let (pkt, _) = result?;
                match pkt {
                    Packet::Control(c) => match c.control_type {
                        ControlType::StopListening => {
                            debug!("server: stop listening");
                            // Send our own StopListening to confirm
                            transport.send_to(
                                &Packet::Control(Control {
                                    session_id,
                                    control_type: ControlType::StopListening,
                                    payload: Bytes::new(),
                                }),
                                server_addr,
                            ).await?;
                            state = State::Receiving;
                        }
                        ControlType::StartSpeaking => {
                            debug!("server: start speaking");
                            state = State::Receiving;
                        }
                        ControlType::StopSpeaking => {
                            info!("server: stop speaking, returning to idle");
                            state = State::Idle;
                        }
                        _ => {
                            debug!(control = ?c.control_type, "unhandled server control");
                        }
                    }
                    Packet::Audio(a) if state == State::Receiving => {
                        let frame = AudioFrame::new(AudioFormat::SPEECH, a.payload);
                        if speaker_tx.send(frame).await.is_err() {
                            warn!("speaker channel closed");
                        }
                    }
                    Packet::Audio(_) => {
                        debug!("ignoring audio in state {:?}", state);
                    }
                    Packet::Keepalive(k) => {
                        transport.send_to(
                            &Packet::Keepalive(Keepalive { session_id: k.session_id }),
                            server_addr,
                        ).await?;
                    }
                    _ => {}
                }
            }
            // Branch 4: keepalive timer
            _ = keepalive_timer.tick() => {
                transport.send_to(
                    &Packet::Keepalive(Keepalive { session_id }),
                    server_addr,
                ).await?;
            }
        }
    }

    Ok(())
}

type WakeChannels = (
    Option<mpsc::Sender<AudioFrame>>,
    Option<mpsc::Receiver<Detection>>,
);

/// Spawn the wake word detection thread if wake is enabled.
/// Returns the frame sender and detection receiver channels.
fn spawn_wake_thread(config: &SatelliteConfig) -> Result<WakeChannels, SatelliteError> {
    if !config.wake.enabled {
        info!("wake word detection disabled");
        return Ok((None, None));
    }

    let detector = pronghorn_wake::create_detector(&config.wake)?;

    // Use tokio mpsc for frame input too (try_send for non-blocking from async context)
    let (wake_frame_tx, mut wake_frame_rx) = mpsc::channel::<AudioFrame>(32);
    let (wake_detect_tx, wake_detect_rx) = mpsc::channel::<Detection>(1);

    std::thread::Builder::new()
        .name("wake-detector".into())
        .spawn(move || {
            let mut det = detector;
            let mut frame_count = 0u64;
            tracing::info!("wake detector thread running");
            while let Some(frame) = wake_frame_rx.blocking_recv() {
                frame_count += 1;
                if frame_count.is_multiple_of(500) {
                    tracing::debug!(frame_count, "wake detector processing frames");
                }
                match det.process_frame(&frame) {
                    Ok(Some(detection)) => {
                        tracing::info!(
                            wake_word = %detection.wake_word,
                            score = detection.score,
                            "WAKE WORD DETECTED on detector thread"
                        );
                        let _ = wake_detect_tx.blocking_send(detection);
                        det.reset();
                    }
                    Ok(None) => {}
                    Err(e) => {
                        tracing::error!("wake word error: {e}");
                    }
                }
            }
            tracing::info!("wake detector thread exiting");
        })?;

    Ok((Some(wake_frame_tx), Some(wake_detect_rx)))
}
