use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use pronghorn_audio::{AudioFormat, AudioFrame};
use tokio::sync::mpsc;
use tracing::{error, info};

/// Start capturing audio from the default input device.
///
/// Returns a `Stream` (must be kept alive) and audio frames are sent to `audio_tx`.
/// Captures at 16kHz mono i16, matching `AudioFormat::SPEECH`.
pub fn start_capture(audio_tx: mpsc::Sender<AudioFrame>) -> Result<Stream, AudioIoError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(AudioIoError::NoInputDevice)?;

    info!(device = ?device.description(), "using input device");

    let desired_config = cpal::StreamConfig {
        channels: 1,
        sample_rate: 16_000,
        buffer_size: cpal::BufferSize::Default,
    };

    let config = match find_compatible_input_config(&device) {
        Some(cfg) => cfg,
        None => {
            info!("device doesn't advertise 16kHz mono i16, trying anyway");
            desired_config
        }
    };

    info!(
        sample_rate = config.sample_rate,
        channels = config.channels,
        "capture config"
    );

    let stream = device.build_input_stream(
        &config,
        move |data: &[i16], _info: &cpal::InputCallbackInfo| {
            // Chunk into 20ms frames (320 samples at 16kHz mono = 640 bytes)
            let frame_samples = 320;
            for chunk in data.chunks(frame_samples) {
                if chunk.len() < frame_samples {
                    continue; // skip partial frames at stream boundaries
                }
                let pcm_bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
                let frame = AudioFrame::new(AudioFormat::SPEECH, Bytes::from(pcm_bytes));
                // Non-blocking send — drop frames if pipeline is slow
                let _ = audio_tx.try_send(frame);
            }
        },
        move |err| {
            error!("capture stream error: {err}");
        },
        None,
    )?;

    stream.play()?;
    info!("audio capture started");
    Ok(stream)
}

/// Start playing audio to the default output device.
///
/// Returns a `Stream` (must be kept alive). Audio frames from `speaker_rx` are played.
pub fn start_playback(mut speaker_rx: mpsc::Receiver<AudioFrame>) -> Result<Stream, AudioIoError> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or(AudioIoError::NoOutputDevice)?;

    info!(device = ?device.description(), "using output device");

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: 16_000,
        buffer_size: cpal::BufferSize::Default,
    };

    // Buffer for samples waiting to be played
    let (sample_tx, sample_rx) = std::sync::mpsc::channel::<Vec<i16>>();

    // Spawn a task to convert AudioFrames → i16 sample buffers
    tokio::spawn(async move {
        while let Some(frame) = speaker_rx.recv().await {
            let samples: Vec<i16> = frame
                .samples
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect();
            if sample_tx.send(samples).is_err() {
                break;
            }
        }
    });

    let mut pending: Vec<i16> = Vec::new();

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [i16], _info: &cpal::OutputCallbackInfo| {
            let mut written = 0;

            // Drain pending samples first
            while written < data.len() && !pending.is_empty() {
                data[written] = pending.remove(0);
                written += 1;
            }

            // Pull new frames
            while written < data.len() {
                match sample_rx.try_recv() {
                    Ok(samples) => {
                        for &s in &samples {
                            if written < data.len() {
                                data[written] = s;
                                written += 1;
                            } else {
                                pending.push(s);
                            }
                        }
                    }
                    Err(_) => break,
                }
            }

            // Fill remainder with silence
            for sample in &mut data[written..] {
                *sample = 0;
            }
        },
        move |err| {
            error!("playback stream error: {err}");
        },
        None,
    )?;

    stream.play()?;
    info!("audio playback started");
    Ok(stream)
}

/// Try to find an input config that matches 16kHz mono i16.
fn find_compatible_input_config(device: &cpal::Device) -> Option<cpal::StreamConfig> {
    let configs = device.supported_input_configs().ok()?;
    for cfg in configs {
        if cfg.channels() == 1
            && cfg.min_sample_rate() <= 16_000
            && cfg.max_sample_rate() >= 16_000
            && cfg.sample_format() == SampleFormat::I16
        {
            return Some(cfg.with_sample_rate(16_000).into());
        }
    }
    None
}

#[derive(Debug, thiserror::Error)]
pub enum AudioIoError {
    #[error("no input device available")]
    NoInputDevice,

    #[error("no output device available")]
    NoOutputDevice,

    #[error("audio stream error: {0}")]
    Stream(#[from] cpal::BuildStreamError),

    #[error("failed to start audio stream: {0}")]
    Play(#[from] cpal::PlayStreamError),
}
