use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Instant;

/// Lifecycle states of a Pronghorn session.
///
/// ```text
/// Connected ──► Listening ──► Processing ──► Speaking ──┐
///     ▲                                                  │
///     └──────────────────────────────────────────────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Handshake complete, idle, waiting for wake word.
    Connected,
    /// Client is streaming audio (post-wake-word).
    Listening,
    /// Server is running the pipeline (STT → intent → TTS).
    Processing,
    /// Server is streaming TTS audio back to the client.
    Speaking,
}

#[derive(Debug)]
pub struct Session {
    pub id: u32,
    pub remote_addr: SocketAddr,
    pub state: SessionState,
    pub last_seen: Instant,
}

/// Manages the set of active sessions, keyed by both id and remote address.
pub struct SessionManager {
    sessions: HashMap<u32, Session>,
    addr_to_session: HashMap<SocketAddr, u32>,
    next_id: u32,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            addr_to_session: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create a new session for `remote_addr` and return the assigned id.
    pub fn create(&mut self, remote_addr: SocketAddr) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let session = Session {
            id,
            remote_addr,
            state: SessionState::Connected,
            last_seen: Instant::now(),
        };
        self.sessions.insert(id, session);
        self.addr_to_session.insert(remote_addr, id);
        id
    }

    pub fn get(&self, id: u32) -> Option<&Session> {
        self.sessions.get(&id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut Session> {
        self.sessions.get_mut(&id)
    }

    pub fn get_by_addr(&self, addr: &SocketAddr) -> Option<&Session> {
        self.addr_to_session
            .get(addr)
            .and_then(|id| self.sessions.get(id))
    }

    /// Update the last-seen timestamp for a session.
    pub fn touch(&mut self, id: u32) {
        if let Some(session) = self.sessions.get_mut(&id) {
            session.last_seen = Instant::now();
        }
    }

    pub fn remove(&mut self, id: u32) -> Option<Session> {
        if let Some(session) = self.sessions.remove(&id) {
            self.addr_to_session.remove(&session.remote_addr);
            Some(session)
        } else {
            None
        }
    }

    /// Remove all sessions that haven't been seen since `deadline`.
    pub fn reap_stale(&mut self, deadline: Instant) -> Vec<Session> {
        let stale_ids: Vec<u32> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.last_seen < deadline)
            .map(|(id, _)| *id)
            .collect();

        stale_ids
            .into_iter()
            .filter_map(|id| self.remove(id))
            .collect()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}
