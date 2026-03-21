use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

    let mut sessions = SessionManager::new();
    let mut handles: HashMap<u32, SessionHandle> = HashMap::new();

    let keepalive_dur = Duration::from_millis(config.transport.keepalive_interval_ms);
    let session_timeout = Duration::from_millis(config.transport.session_timeout_ms);
    let mut keepalive_timer = tokio::time::interval(keepalive_dur);
    let mut reap_timer = tokio::time::interval(Duration::from_secs(10));

    // Don't fire immediately on first tick
    keepalive_timer.tick().await;
    reap_timer.tick().await;

    loop {
        tokio::select! {
            result = transport.recv_from() => {
                let (packet, addr) = result?;
                handle_packet(
                    packet,
                    addr,
                    &config,
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

async fn handle_packet(
    packet: Packet,
    addr: std::net::SocketAddr,
    config: &ServerConfig,
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
                            config,
                            Arc::clone(transport),
                            session.remote_addr,
                        );
                        handles.insert(ctrl.session_id, handle);
                    }
                }
                ControlType::StopListening => {
                    info!(session_id = ctrl.session_id, "stop listening");
                    if let Some(handle) = handles.remove(&ctrl.session_id) {
                        // Drop audio_tx to signal end-of-stream.
                        // The orchestrator task continues to process remaining audio,
                        // run intent, TTS, and send response back.
                        drop(handle.audio_tx);
                        // Re-insert with a dummy sender so we can track the task
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
            // Echo keepalive back
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
            // Server shouldn't receive Welcome
            warn!(%addr, "unexpected Welcome packet");
        }
    }
    Ok(())
}

async fn send_keepalives(transport: &Transport, sessions: &SessionManager) {
    // SessionManager doesn't expose an iterator, so we track addresses separately.
    // For now, keepalives are sent in response to received keepalives (echo pattern).
    // TODO: proactive keepalives once SessionManager exposes iteration
    let _ = (transport, sessions);
}

/// Check for orchestrator tasks that have completed and clean up.
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
