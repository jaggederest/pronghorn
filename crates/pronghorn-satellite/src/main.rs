mod config;

use std::path::PathBuf;

use tracing::info;

use crate::config::SatelliteConfig;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config_path = PathBuf::from("satellite.toml");
    let config = SatelliteConfig::load_or_default(&config_path).unwrap_or_else(|e| {
        eprintln!("error loading config: {e}");
        std::process::exit(1);
    });

    info!("pronghorn-satellite v{}", env!("CARGO_PKG_VERSION"));
    info!(
        server = %config.transport.server_address,
        preroll_frames = config.audio.preroll_frames,
        wake_enabled = config.wake.enabled,
        "config loaded"
    );

    // TODO: event loop (Phase 4)
}
