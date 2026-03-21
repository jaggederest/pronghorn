use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pronghorn_pipeline::{
    IntentDispatch, SttDispatch, TtsDispatch, create_intent, create_stt, create_tts,
};
use pronghorn_wire::{
    ControlType, Keepalive, PROTOCOL_VERSION, Packet, SessionManager, SessionState, Transport,
    Welcome,
};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::config::ServerConfig;
use crate::error::ServerError;
use crate::orchestrator::{self, SessionHandle};

pub async fn run_server(config: ServerConfig) -> Result<(), ServerError> {
    let transport = Arc::new(Transport::bind(config.transport.bind_address).await?);
    let addr = transport.local_addr()?;
    info!(bind = %addr, "server listening");

    // Create pipeline backends once at startup (shared across all sessions)
    info!("initializing pipeline backends...");
    let stt = Arc::new(create_stt(&config.pipeline.stt)?);
    info!(backend = ?config.pipeline.stt.backend, "STT ready");
    let tts = Arc::new(create_tts(&config.pipeline.tts)?);
    info!(backend = ?config.pipeline.tts.backend, "TTS ready");
    let intent = Arc::new(create_intent(&config.pipeline.intent)?);
    info!(backend = ?config.pipeline.intent.backend, "intent ready");

    let jitter_delay = config.transport.jitter_buffer_delay;
    let mut sessions = SessionManager::new();
    let mut handles: HashMap<u32, SessionHandle> = HashMap::new();

    let keepalive_dur = Duration::from_millis(config.transport.keepalive_interval_ms);
    let session_timeout = Duration::from_millis(config.transport.session_timeout_ms);
    let mut keepalive_timer = tokio::time::interval(keepalive_dur);
    let mut reap_timer = tokio::time::interval(Duration::from_secs(10));

    // Don't fire immediately on first tick
    keepalive_timer.tick().await;
    reap_timer.tick().await;

    info!("server ready, waiting for connections");

    loop {
        tokio::select! {
            result = transport.recv_from() => {
                let (packet, addr) = result?;
                handle_packet(
                    packet,
                    addr,
                    &stt,
                    &tts,
                    &intent,
                    jitter_delay,
                    &transport,
                    &mut sessions,
                    &mut handles,
                ).await?;
            }
            _ = keepalive_timer.tick() => {
                send_keepalives(&transport, &sessions).await;
            }
            _ = reap_timer.tick() => {
                let deadline = Instant::now() - session_timeout;
                let stale = sessions.reap_stale(deadline);
                for session in &stale {
                    info!(session_id = session.id, "reaped stale session");
                    if let Some(handle) = handles.remove(&session.id) {
                        handle.task.abort();
                    }
                }
            }
        }

        // Check for completed orchestrator tasks
        collect_completed(&mut handles, &mut sessions);
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_packet(
    packet: Packet,
    addr: std::net::SocketAddr,
    stt: &Arc<SttDispatch>,
    tts: &Arc<TtsDispatch>,
    intent: &Arc<IntentDispatch>,
    jitter_delay: u16,
    transport: &Arc<Transport>,
    sessions: &mut SessionManager,
    handles: &mut HashMap<u32, SessionHandle>,
) -> Result<(), ServerError> {
    match packet {
        Packet::Hello(h) => {
            let sid = sessions.create(addr);
            info!(session_id = sid, %addr, client_version = h.client_version, "new session");
            let welcome = Packet::Welcome(Welcome {
                session_id: sid,
                server_version: PROTOCOL_VERSION as u32,
            });
            transport.send_to(&welcome, addr).await?;
        }
        Packet::Audio(audio) => {
            sessions.touch(audio.session_id);
            if let Some(handle) = handles.get(&audio.session_id)
                && handle.audio_tx.send(audio).await.is_err()
            {
                debug!(
                    session_id = handle.audio_tx.capacity(),
                    "orchestrator channel closed"
                );
            }
        }
        Packet::Control(ctrl) => {
            sessions.touch(ctrl.session_id);
            match ctrl.control_type {
                ControlType::StartListening => {
                    if let Some(session) = sessions.get_mut(ctrl.session_id)
                        && session.state == SessionState::Connected
                    {
                        session.state = SessionState::Listening;
                        info!(session_id = ctrl.session_id, "start listening");

                        let handle = orchestrator::spawn_orchestrator(
                            ctrl.session_id,
                            Arc::clone(stt),
                            Arc::clone(tts),
                            Arc::clone(intent),
                            Arc::clone(transport),
                            session.remote_addr,
                            jitter_delay,
                        );
                        handles.insert(ctrl.session_id, handle);
                    }
                }
                ControlType::StopListening => {
                    info!(session_id = ctrl.session_id, "stop listening");
                    if let Some(handle) = handles.remove(&ctrl.session_id) {
                        drop(handle.audio_tx);
                        let (dummy_tx, _) = mpsc::channel(1);
                        handles.insert(
                            ctrl.session_id,
                            SessionHandle {
                                audio_tx: dummy_tx,
                                task: handle.task,
                            },
                        );
                    }
                    if let Some(session) = sessions.get_mut(ctrl.session_id) {
                        session.state = SessionState::Processing;
                    }
                }
                ControlType::SessionEnd => {
                    info!(session_id = ctrl.session_id, "session end");
                    if let Some(handle) = handles.remove(&ctrl.session_id) {
                        handle.task.abort();
                    }
                    sessions.remove(ctrl.session_id);
                }
                _ => {
                    debug!(session_id = ctrl.session_id, control = ?ctrl.control_type, "unhandled control");
                }
            }
        }
        Packet::Keepalive(k) => {
            sessions.touch(k.session_id);
            transport
                .send_to(
                    &Packet::Keepalive(Keepalive {
                        session_id: k.session_id,
                    }),
                    addr,
                )
                .await?;
        }
        Packet::Welcome(_) => {
            warn!(%addr, "unexpected Welcome packet");
        }
    }
    Ok(())
}

async fn send_keepalives(transport: &Transport, sessions: &SessionManager) {
    let _ = (transport, sessions);
}

fn collect_completed(handles: &mut HashMap<u32, SessionHandle>, sessions: &mut SessionManager) {
    let completed: Vec<u32> = handles
        .iter()
        .filter(|(_, h)| h.task.is_finished())
        .map(|(id, _)| *id)
        .collect();

    for id in completed {
        handles.remove(&id);
        if let Some(session) = sessions.get_mut(id) {
            session.state = SessionState::Connected;
            debug!(session_id = id, "orchestrator complete, session idle");
        }
    }
}
