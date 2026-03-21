use std::path::PathBuf;

use pronghorn_audio::AudioFrame;
use pronghorn_satellite::config::SatelliteConfig;
use pronghorn_satellite::event_loop;
use tokio::sync::mpsc;
use tracing::info;

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

    // TODO (Phase 6): replace with real mic capture and speaker playback
    let (_audio_tx, audio_rx) = mpsc::channel::<AudioFrame>(64);
    let (speaker_tx, _speaker_rx) = mpsc::channel::<AudioFrame>(64);

    if let Err(e) = event_loop::run_satellite(config, audio_rx, speaker_tx).await {
        eprintln!("satellite error: {e}");
        std::process::exit(1);
    }
}
