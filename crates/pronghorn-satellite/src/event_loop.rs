use std::time::Duration;

use bytes::Bytes;
use pronghorn_audio::{AudioFormat, AudioFrame, RingBuffer};
use pronghorn_wake::Detection;
use pronghorn_wire::{AudioData, Control, ControlType, Hello, Keepalive, Packet, Transport};
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

    // VAD state
    let speech_threshold = config.transport.speech_threshold as f64;
    let silence_frame_limit = (config.transport.silence_duration_ms / 20).max(1) as u32; // frames at 20ms each
    let min_speech_frames = (config.transport.min_speech_ms / 20).max(1) as u32;
    let max_stream_duration = Duration::from_millis(config.transport.max_stream_duration_ms);
    let mut speech_detected = false;
    let mut silence_frames = 0u32;
    let mut speech_frames = 0u32;

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
                        // VAD: compute RMS of this frame
                        let rms = compute_rms(&frame.samples);

                        // Log RMS every 50th frame (~1s) to help tune threshold
                        let total_frames = speech_frames + silence_frames;
                        if total_frames < 10 || total_frames.is_multiple_of(50) {
                            debug!(rms = rms as u32, threshold = speech_threshold as u32, speech_frames, silence_frames, "VAD");
                        }

                        // Hard timeout safety net
                        let timed_out = streaming_since
                            .is_some_and(|since| since.elapsed() > max_stream_duration);

                        // Track speech/silence
                        if rms > speech_threshold {
                            speech_detected = true;
                            speech_frames += 1;
                            silence_frames = 0;
                        } else if speech_detected && speech_frames >= min_speech_frames {
                            silence_frames += 1;
                        }

                        let vad_stop = speech_detected
                            && speech_frames >= min_speech_frames
                            && silence_frames >= silence_frame_limit;

                        if vad_stop || timed_out {
                            if vad_stop {
                                info!(
                                    speech_frames,
                                    silence_frames,
                                    "VAD: end of speech detected"
                                );
                            } else {
                                info!("streaming timeout (hard cap)");
                            }
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
                            speech_detected = false;
                            silence_frames = 0;
                            speech_frames = 0;
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

                    // Send the tail of the pre-roll buffer. Most of the buffer
                    // contains the wake word, so we only send the last few frames
                    // which may capture the start of the actual command.
                    sequence = 0;
                    timestamp = 0;
                    let preroll = ring.drain();
                    let keep_frames = 6; // ~120ms of trailing audio
                    let skip = preroll.len().saturating_sub(keep_frames);
                    debug!(
                        total_preroll = preroll.len(),
                        skipped = skip,
                        sending = preroll.len() - skip,
                        "sending tail of pre-roll"
                    );
                    for frame in preroll.into_iter().skip(skip) {
                        let pkt = Packet::Audio(AudioData {
                            session_id,
                            sequence,
                            flags: pronghorn_wire::audio_flags::PRE_ROLL,
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
                    Packet::Keepalive(_) => {
                        // Server echoed our keepalive — connection is alive.
                        // Do NOT echo back (would create infinite ping-pong).
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

/// Compute RMS (root mean square) of i16 PCM samples in a frame's Bytes payload.
fn compute_rms(samples: &bytes::Bytes) -> f64 {
    let count = samples.len() / 2;
    if count == 0 {
        return 0.0;
    }
    let sum_sq: f64 = samples
        .chunks_exact(2)
        .map(|b| {
            let s = i16::from_le_bytes([b[0], b[1]]) as f64;
            s * s
        })
        .sum();
    (sum_sq / count as f64).sqrt()
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
