mod config;

use std::path::PathBuf;

use tracing::info;

use crate::config::PronghornConfig;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config_path = PathBuf::from("pronghorn.toml");
    let config = PronghornConfig::load_or_default(&config_path).unwrap_or_else(|e| {
        eprintln!("error loading config: {e}");
        std::process::exit(1);
    });

    info!("pronghorn v{}", env!("CARGO_PKG_VERSION"));
    info!(
        bind = %config.transport.bind_address,
        preroll_frames = config.audio.preroll_frames,
        jitter_delay = config.transport.jitter_buffer_delay,
        "config loaded"
    );

    if config.wake.enabled {
        info!(backend = ?config.wake.backend, "wake word detection enabled");
    } else {
        info!("wake word detection disabled (set [wake] enabled = true to activate)");
    }
}
