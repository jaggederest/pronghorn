mod config;

use std::path::PathBuf;

use tracing::info;

use crate::config::ServerConfig;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config_path = PathBuf::from("server.toml");
    let config = ServerConfig::load_or_default(&config_path).unwrap_or_else(|e| {
        eprintln!("error loading config: {e}");
        std::process::exit(1);
    });

    info!("pronghorn-server v{}", env!("CARGO_PKG_VERSION"));
    info!(
        bind = %config.transport.bind_address,
        jitter_delay = config.transport.jitter_buffer_delay,
        stt_backend = ?config.pipeline.stt.backend,
        tts_backend = ?config.pipeline.tts.backend,
        intent_backend = ?config.pipeline.intent.backend,
        "config loaded"
    );

    // TODO: event loop (Phase 3)
}
