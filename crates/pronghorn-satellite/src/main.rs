use std::path::PathBuf;
use std::time::Duration;

use pronghorn_audio::AudioFrame;
use pronghorn_satellite::audio_io;
use pronghorn_satellite::config::SatelliteConfig;
use pronghorn_satellite::event_loop;
use tokio::sync::mpsc;
use tracing::{info, warn};

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

    let mut backoff = Duration::from_secs(1);
    let max_backoff = Duration::from_secs(30);

    loop {
        // Create fresh audio channels for each connection attempt.
        // Audio capture/playback streams hold the other halves and are
        // dropped+recreated each iteration so the mic/speaker reset cleanly.
        let (audio_tx, audio_rx) = mpsc::channel::<AudioFrame>(64);
        let (speaker_tx, speaker_rx) = mpsc::channel::<AudioFrame>(64);

        let _capture_stream = match audio_io::start_capture(audio_tx) {
            Ok(stream) => Some(stream),
            Err(e) => {
                warn!(%e, "failed to start audio capture, running without mic");
                None
            }
        };

        let _playback_stream = match audio_io::start_playback(speaker_rx) {
            Ok(stream) => Some(stream),
            Err(e) => {
                warn!(%e, "failed to start audio playback, running without speaker");
                None
            }
        };

        info!("connecting to server...");
        match event_loop::run_satellite(config.clone(), audio_rx, speaker_tx).await {
            Ok(()) => {
                // Clean shutdown (audio source closed) — exit the process
                info!("satellite shut down cleanly");
                break;
            }
            Err(e) => {
                warn!(%e, backoff_ms = backoff.as_millis() as u64, "satellite error, reconnecting");
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(max_backoff);
            }
        }
    }
}
