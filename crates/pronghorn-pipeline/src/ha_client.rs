//! Home Assistant WebSocket API client.
//!
//! Connects to HA's native WebSocket API, authenticates with a long-lived
//! access token, fetches the entity registry, subscribes to state changes,
//! and calls services for intent actions.
//!
//! Enable the `ha` feature to compile the network implementation.
//! Without the feature, only the data types are available and `connect()`
//! returns an error.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;

#[cfg(feature = "ha")]
use std::sync::Arc;
#[cfg(feature = "ha")]
use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(feature = "ha")]
use tokio::sync::oneshot;
#[cfg(feature = "ha")]
use tracing::info;

use crate::config::HaConfig;

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum HaError {
    #[error("WebSocket connection failed: {0}")]
    Connect(String),

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Request failed (id={id}): {message}")]
    Request { id: u32, message: String },

    #[error("Channel closed")]
    ChannelClosed,
}

// ── Data types ───────────────────────────────────────────────────────────────

/// A single entity from the HA entity registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInfo {
    /// Primary entity ID, e.g. `light.living_room`.
    pub entity_id: String,
    /// User-assigned friendly name (if set).
    pub name: Option<String>,
    /// User-assigned aliases for voice matching.
    pub aliases: Vec<String>,
    /// Area ID this entity belongs to, if any.
    pub area_id: Option<String>,
    /// Labels attached to the entity.
    #[serde(default)]
    pub labels: Vec<String>,
}

/// A state change event from HA.
#[derive(Debug, Clone)]
pub struct StateChange {
    pub entity_id: String,
    pub new_state: Option<serde_json::Value>,
    pub old_state: Option<serde_json::Value>,
}

// ── Client ───────────────────────────────────────────────────────────────────

/// Home Assistant WebSocket client.
///
/// Constructed via [`HaClient::connect`]. Internally runs a background
/// dispatch task that reads from the WebSocket and routes messages to
/// pending request waiters or subscription listeners.
///
/// Clone-safe: all handles are reference-counted.
#[derive(Clone)]
pub struct HaClient {
    #[cfg(feature = "ha")]
    cmd_tx: mpsc::Sender<net::Command>,
    #[cfg(feature = "ha")]
    next_id: Arc<AtomicU32>,
}

impl HaClient {
    /// Connect to HA and authenticate.
    ///
    /// Returns an `HaClient` backed by a background dispatch task.
    pub async fn connect(config: &HaConfig) -> Result<Self, HaError> {
        #[cfg(feature = "ha")]
        {
            net::connect_impl(config).await
        }
        #[cfg(not(feature = "ha"))]
        {
            let _ = config;
            Err(HaError::Connect("ha feature not enabled".into()))
        }
    }

    /// Fetch the full entity registry.
    ///
    /// Returns all entities known to HA, including friendly names, aliases,
    /// and area assignments. Call once at startup to seed the fuzzy matcher.
    pub async fn entities(&self) -> Result<Vec<EntityInfo>, HaError> {
        #[cfg(feature = "ha")]
        {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let msg = serde_json::json!({
                "id": id,
                "type": "config/entity_registry/list",
            });
            let result = self.request(id, msg).await?;
            let entities: Vec<EntityInfo> = serde_json::from_value(result)
                .map_err(|e| HaError::Protocol(format!("entity registry parse error: {e}")))?;
            info!(count = entities.len(), "fetched entity registry");
            Ok(entities)
        }
        #[cfg(not(feature = "ha"))]
        {
            Err(HaError::Connect("ha feature not enabled".into()))
        }
    }

    /// Subscribe to state change events.
    ///
    /// Returns a channel receiver that yields a `StateChange` for every
    /// `state_changed` event HA emits. The channel closes when the
    /// WebSocket disconnects.
    pub async fn subscribe_state_changes(&self) -> Result<mpsc::Receiver<StateChange>, HaError> {
        #[cfg(feature = "ha")]
        {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let msg = serde_json::json!({
                "id": id,
                "type": "subscribe_events",
                "event_type": "state_changed",
            });
            // Confirm subscription accepted before returning.
            let _result = self.request(id, msg).await?;

            // Register listener for subsequent event messages on this sub id.
            let (event_tx, event_rx_raw) = mpsc::channel::<serde_json::Value>(64);
            self.cmd_tx
                .send(net::Command::RegisterSub { id, tx: event_tx })
                .await
                .map_err(|_| HaError::ChannelClosed)?;

            // Translate raw event JSON into typed StateChange values.
            let (sc_tx, sc_rx) = mpsc::channel::<StateChange>(64);
            tokio::spawn(async move {
                let mut raw = event_rx_raw;
                while let Some(event) = raw.recv().await {
                    let sc = parse_state_change(&event);
                    if sc_tx.send(sc).await.is_err() {
                        break;
                    }
                }
            });
            info!(sub_id = id, "subscribed to state_changed events");
            Ok(sc_rx)
        }
        #[cfg(not(feature = "ha"))]
        {
            Err(HaError::Connect("ha feature not enabled".into()))
        }
    }

    /// Call a HA service.
    ///
    /// # Arguments
    /// - `domain` — service domain, e.g. `"light"`
    /// - `service` — service name, e.g. `"turn_on"`
    /// - `entity_id` — target entity, e.g. `"light.living_room"`
    /// - `service_data` — extra fields (pass `serde_json::Value::Null` for none)
    pub async fn call_service(
        &self,
        domain: &str,
        service: &str,
        entity_id: &str,
        service_data: serde_json::Value,
    ) -> Result<(), HaError> {
        #[cfg(feature = "ha")]
        {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let mut msg = serde_json::json!({
                "id": id,
                "type": "call_service",
                "domain": domain,
                "service": service,
                "target": { "entity_id": entity_id },
            });
            if !service_data.is_null() {
                msg["service_data"] = service_data;
            }
            tracing::debug!(id, domain, service, entity_id, "call_service");
            let _result = self.request(id, msg).await?;
            Ok(())
        }
        #[cfg(not(feature = "ha"))]
        {
            let _ = (domain, service, entity_id, service_data);
            Err(HaError::Connect("ha feature not enabled".into()))
        }
    }

    // ── Internal helpers (ha feature only) ───────────────────────────────────

    /// Send a request and wait for the matching result message.
    #[cfg(feature = "ha")]
    async fn request(
        &self,
        id: u32,
        msg: serde_json::Value,
    ) -> Result<serde_json::Value, HaError> {
        let (tx, rx) = oneshot::channel();

        // Register before sending to avoid a race.
        self.cmd_tx
            .send(net::Command::RegisterPending { id, tx })
            .await
            .map_err(|_| HaError::ChannelClosed)?;
        self.cmd_tx
            .send(net::Command::Send(msg))
            .await
            .map_err(|_| HaError::ChannelClosed)?;

        rx.await.map_err(|_| HaError::ChannelClosed)?
    }
}

// ── Feature-gated network implementation ─────────────────────────────────────

#[cfg(feature = "ha")]
pub(super) mod net {
    use std::collections::HashMap;

    use futures_util::{SinkExt, StreamExt};
    use tokio::sync::{oneshot, Mutex};
    use tokio_tungstenite::{connect_async, tungstenite::Message};
    use tracing::{debug, info, warn};

    use super::*;

    /// Internal commands sent from callers to the background dispatch task.
    pub enum Command {
        Send(serde_json::Value),
        RegisterPending {
            id: u32,
            tx: oneshot::Sender<Result<serde_json::Value, HaError>>,
        },
        RegisterSub {
            id: u32,
            tx: mpsc::Sender<serde_json::Value>,
        },
    }

    pub(super) async fn connect_impl(config: &HaConfig) -> Result<HaClient, HaError> {
        info!(url = %config.url, "connecting to Home Assistant");

        let (ws_stream, _) = connect_async(&config.url)
            .await
            .map_err(|e| HaError::Connect(e.to_string()))?;

        let (mut writer, mut reader) = ws_stream.split();

        // ── Auth handshake ────────────────────────────────────────────────
        let first = reader
            .next()
            .await
            .ok_or_else(|| HaError::Protocol("connection closed before auth_required".into()))?
            .map_err(|e| HaError::Protocol(e.to_string()))?;

        let first_json = ws_msg_to_json(first)?;
        if first_json.get("type").and_then(|v| v.as_str()) != Some("auth_required") {
            return Err(HaError::Protocol(format!(
                "expected auth_required, got: {first_json}"
            )));
        }

        let auth_msg = serde_json::json!({
            "type": "auth",
            "access_token": config.token,
        });
        writer
            .send(Message::Text(auth_msg.to_string().into()))
            .await
            .map_err(|e| HaError::Connect(e.to_string()))?;

        let auth_resp = reader
            .next()
            .await
            .ok_or_else(|| HaError::Auth("connection closed during auth".into()))?
            .map_err(|e| HaError::Auth(e.to_string()))?;

        let auth_json = ws_msg_to_json(auth_resp)?;
        match auth_json.get("type").and_then(|v| v.as_str()) {
            Some("auth_ok") => {
                info!("Home Assistant authentication successful");
            }
            Some("auth_invalid") => {
                let msg = auth_json
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                return Err(HaError::Auth(msg.to_string()));
            }
            other => {
                return Err(HaError::Auth(format!(
                    "unexpected auth response type: {other:?}"
                )));
            }
        }

        // ── Spawn dispatch task ───────────────────────────────────────────
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>(64);
        tokio::spawn(dispatch_loop(writer, reader, cmd_rx));

        Ok(HaClient {
            cmd_tx,
            next_id: Arc::new(AtomicU32::new(1)),
        })
    }

    /// Background dispatch task: routes outbound sends and inbound responses.
    async fn dispatch_loop(
        writer: impl SinkExt<Message, Error = tokio_tungstenite::tungstenite::Error>
            + Unpin
            + Send
            + 'static,
        mut reader: impl StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>>
            + Unpin
            + Send
            + 'static,
        mut cmd_rx: mpsc::Receiver<Command>,
    ) {
        let writer = Arc::new(Mutex::new(writer));
        type PendingMap = Mutex<HashMap<u32, oneshot::Sender<Result<serde_json::Value, HaError>>>>;
        let pending: Arc<PendingMap> = Arc::new(Mutex::new(HashMap::new()));
        let subs: Arc<Mutex<HashMap<u32, mpsc::Sender<serde_json::Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        loop {
            tokio::select! {
                Some(cmd) = cmd_rx.recv() => {
                    match cmd {
                        Command::Send(json) => {
                            let text = json.to_string();
                            let mut w = writer.lock().await;
                            if let Err(e) = w.send(Message::Text(text.into())).await {
                                warn!(err = %e, "ws send failed");
                            }
                        }
                        Command::RegisterPending { id, tx } => {
                            pending.lock().await.insert(id, tx);
                        }
                        Command::RegisterSub { id, tx } => {
                            subs.lock().await.insert(id, tx);
                        }
                    }
                }

                msg = reader.next() => {
                    match msg {
                        None => {
                            debug!("WebSocket closed by server");
                            break;
                        }
                        Some(Err(e)) => {
                            warn!(err = %e, "WebSocket read error");
                            break;
                        }
                        Some(Ok(raw)) => {
                            let json = match ws_msg_to_json(raw) {
                                Ok(j) => j,
                                Err(_) => continue,
                            };
                            route_message(json, &pending, &subs).await;
                        }
                    }
                }
            }
        }
    }

    type PendingMap = Mutex<HashMap<u32, oneshot::Sender<Result<serde_json::Value, HaError>>>>;

    /// Route an inbound HA message to the correct waiter or subscriber.
    async fn route_message(
        json: serde_json::Value,
        pending: &PendingMap,
        subs: &Mutex<HashMap<u32, mpsc::Sender<serde_json::Value>>>,
    ) {
        let msg_type = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let id = json.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);

        match msg_type {
            "result" => {
                let Some(id) = id else { return };
                let success = json
                    .get("success")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if let Some(tx) = pending.lock().await.remove(&id) {
                    let outcome = if success {
                        let result =
                            json.get("result").cloned().unwrap_or(serde_json::Value::Null);
                        Ok(result)
                    } else {
                        let err_msg = json
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown error")
                            .to_string();
                        Err(HaError::Request { id, message: err_msg })
                    };
                    let _ = tx.send(outcome);
                }
            }

            "event" => {
                let Some(id) = id else { return };
                let event = json
                    .get("event")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let subs = subs.lock().await;
                if let Some(tx) = subs.get(&id)
                    && tx.send(event).await.is_err()
                {
                    debug!(sub_id = id, "subscription listener dropped");
                }
            }

            other => {
                debug!(msg_type = other, "unhandled HA message type");
            }
        }
    }

    pub(super) fn ws_msg_to_json(msg: Message) -> Result<serde_json::Value, HaError> {
        match msg {
            Message::Text(text) => serde_json::from_str(&text)
                .map_err(|e| HaError::Protocol(format!("JSON parse error: {e}"))),
            Message::Close(_) => Err(HaError::Protocol("connection closed".into())),
            _ => Err(HaError::Protocol("non-text frame".into())),
        }
    }
}

// ── State change parser ───────────────────────────────────────────────────────

#[cfg(feature = "ha")]
fn parse_state_change(event: &serde_json::Value) -> StateChange {
    let data = event.get("data");
    StateChange {
        entity_id: data
            .and_then(|d| d.get("entity_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        new_state: data.and_then(|d| d.get("new_state")).cloned(),
        old_state: data.and_then(|d| d.get("old_state")).cloned(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ha")]
    #[test]
    fn parse_state_change_extracts_fields() {
        let event = serde_json::json!({
            "event_type": "state_changed",
            "data": {
                "entity_id": "light.kitchen",
                "new_state": {"state": "on"},
                "old_state": {"state": "off"},
            }
        });
        let sc = parse_state_change(&event);
        assert_eq!(sc.entity_id, "light.kitchen");
        assert!(sc.new_state.is_some());
        assert!(sc.old_state.is_some());
    }

    #[cfg(feature = "ha")]
    #[test]
    fn parse_state_change_handles_missing_data() {
        let event = serde_json::json!({});
        let sc = parse_state_change(&event);
        assert_eq!(sc.entity_id, "");
        assert!(sc.new_state.is_none());
        assert!(sc.old_state.is_none());
    }

    #[test]
    fn entity_info_deserializes() {
        let json = serde_json::json!({
            "entity_id": "light.living_room",
            "name": "Living Room",
            "aliases": ["living room lights", "lounge"],
            "area_id": "living_room",
            "labels": [],
        });
        let entity: EntityInfo = serde_json::from_value(json).unwrap();
        assert_eq!(entity.entity_id, "light.living_room");
        assert_eq!(entity.name.as_deref(), Some("Living Room"));
        assert_eq!(entity.aliases, vec!["living room lights", "lounge"]);
        assert_eq!(entity.area_id.as_deref(), Some("living_room"));
    }

    #[test]
    fn entity_info_handles_missing_optional_fields() {
        let json = serde_json::json!({
            "entity_id": "sensor.temperature",
            "aliases": [],
        });
        let entity: EntityInfo = serde_json::from_value(json).unwrap();
        assert_eq!(entity.entity_id, "sensor.temperature");
        assert!(entity.name.is_none());
        assert!(entity.area_id.is_none());
    }

    #[test]
    fn ha_config_default_url() {
        let cfg = crate::config::HaConfig::default();
        assert!(cfg.url.starts_with("ws://"));
        assert!(cfg.token.is_empty());
    }

    #[tokio::test]
    async fn connect_without_feature_returns_err() {
        let cfg = crate::config::HaConfig::default();
        let result = HaClient::connect(&cfg).await;
        // Always Err without a live HA server; with ha feature it's a connect
        // error, without the feature it's a different error message.
        assert!(result.is_err());
    }
}
