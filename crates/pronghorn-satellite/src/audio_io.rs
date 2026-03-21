use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use pronghorn_audio::{AudioFormat, AudioFrame};
use tokio::sync::mpsc;
use tracing::{error, info};

/// Start capturing audio from the default input device.
///
/// Captures at the device's native rate and resamples to 16kHz mono i16.
/// Returns a `Stream` (must be kept alive) and audio frames are sent to `audio_tx`.
pub fn start_capture(audio_tx: mpsc::Sender<AudioFrame>) -> Result<Stream, AudioIoError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(AudioIoError::NoInputDevice)?;

    info!(device = ?device.description(), "using input device");

    // Try 16kHz first, fall back to device default
    let config = match find_compatible_input_config(&device) {
        Some(cfg) => cfg,
        None => {
            // Use the device's default config and resample
            let default = device.default_input_config().map_err(|_| {
                AudioIoError::Stream(cpal::BuildStreamError::StreamConfigNotSupported)
            })?;
            info!(
                sample_rate = default.sample_rate(),
                channels = default.channels(),
                format = ?default.sample_format(),
                "using device default config (will resample to 16kHz)"
            );
            cpal::StreamConfig {
                channels: default.channels(),
                sample_rate: default.sample_rate(),
                buffer_size: cpal::BufferSize::Default,
            }
        }
    };

    let native_rate = config.sample_rate;
    let native_channels = config.channels;
    info!(
        sample_rate = native_rate,
        channels = native_channels,
        "capture config"
    );

    let needs_resample = native_rate != 16_000;
    let needs_downmix = native_channels > 1;

    // Simple integer decimation ratio (e.g., 48000/16000 = 3)
    let decimate_ratio = if needs_resample {
        native_rate / 16_000
    } else {
        1
    };

    // Accumulator for partial frames across callbacks
    let accumulator = std::sync::Arc::new(std::sync::Mutex::new(Vec::<i16>::with_capacity(640)));

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _info: &cpal::InputCallbackInfo| {
            // Convert to mono
            let mono_samples: Vec<f32> = if needs_downmix {
                data.chunks(native_channels as usize)
                    .map(|ch| ch.iter().sum::<f32>() / native_channels as f32)
                    .collect()
            } else {
                data.to_vec()
            };

            // Decimate to 16kHz
            let resampled: Vec<i16> = if needs_resample {
                mono_samples
                    .iter()
                    .step_by(decimate_ratio as usize)
                    .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                    .collect()
            } else {
                mono_samples
                    .iter()
                    .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                    .collect()
            };

            // Accumulate and emit 20ms frames (320 samples at 16kHz)
            let frame_samples = 320;
            let mut acc = accumulator.lock().unwrap();
            acc.extend_from_slice(&resampled);

            while acc.len() >= frame_samples {
                let chunk: Vec<i16> = acc.drain(..frame_samples).collect();
                let pcm_bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
                let frame = AudioFrame::new(AudioFormat::SPEECH, Bytes::from(pcm_bytes));
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

    // Use device default and upsample from 16kHz if needed
    let default = device
        .default_output_config()
        .map_err(|_| AudioIoError::Stream(cpal::BuildStreamError::StreamConfigNotSupported))?;

    let config = cpal::StreamConfig {
        channels: default.channels(),
        sample_rate: default.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let output_rate = config.sample_rate;
    let output_channels = config.channels;
    let upsample_ratio = output_rate / 16_000;

    info!(
        sample_rate = output_rate,
        channels = output_channels,
        "playback config"
    );

    // Buffer for samples waiting to be played
    let (sample_tx, sample_rx) = std::sync::mpsc::channel::<Vec<f32>>();

    // Spawn a task to convert AudioFrames → f32 sample buffers (upsampled + channel-expanded)
    tokio::spawn(async move {
        while let Some(frame) = speaker_rx.recv().await {
            let samples_i16: Vec<i16> = frame
                .samples
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect();

            // Upsample and expand channels
            let mut output = Vec::with_capacity(
                samples_i16.len() * upsample_ratio as usize * output_channels as usize,
            );
            for &s in &samples_i16 {
                let f = s as f32 / 32768.0;
                // Repeat sample for upsampling (simple, not ideal but functional)
                for _ in 0..upsample_ratio {
                    for _ in 0..output_channels {
                        output.push(f);
                    }
                }
            }

            if sample_tx.send(output).is_err() {
                break;
            }
        }
    });

    let mut pending: Vec<f32> = Vec::new();

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _info: &cpal::OutputCallbackInfo| {
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
                        for s in samples {
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
                *sample = 0.0;
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
