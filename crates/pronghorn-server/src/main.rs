use std::path::PathBuf;

use pronghorn_server::config::ServerConfig;
use pronghorn_server::event_loop;
use tracing::info;

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

    if let Err(e) = event_loop::run_server(config).await {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}
