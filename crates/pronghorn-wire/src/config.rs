use std::net::SocketAddr;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TransportConfig {
    /// Address the server binds to.
    pub bind_address: SocketAddr,
    /// Jitter buffer playout delay in frames.
    /// Higher = more jitter tolerance, more latency.
    /// Recommended: 3 (LAN/Ethernet), 5 (WiFi).
    pub jitter_buffer_delay: u16,
    /// Keepalive interval in milliseconds.
    pub keepalive_interval_ms: u64,
    /// Session timeout in milliseconds. Sessions with no activity
    /// for this long are reaped.
    pub session_timeout_ms: u64,
    /// How often to check for stale sessions, in milliseconds.
    pub reap_interval_ms: u64,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            bind_address: ([0, 0, 0, 0], 9999).into(),
            jitter_buffer_delay: 3,
            keepalive_interval_ms: 5_000,
            session_timeout_ms: 30_000,
            reap_interval_ms: 10_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips_through_toml() {
        let config = TransportConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: TransportConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.bind_address, config.bind_address);
        assert_eq!(parsed.jitter_buffer_delay, config.jitter_buffer_delay);
        assert_eq!(parsed.keepalive_interval_ms, config.keepalive_interval_ms);
        assert_eq!(parsed.session_timeout_ms, config.session_timeout_ms);
    }

    #[test]
    fn partial_toml_uses_defaults() {
        let toml_str = r#"jitter_buffer_delay = 5"#;
        let parsed: TransportConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(parsed.jitter_buffer_delay, 5);
        assert_eq!(parsed.bind_address, SocketAddr::from(([0, 0, 0, 0], 9999)));
    }
}
