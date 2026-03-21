use std::sync::{Arc, Mutex};

use audioadapter_buffers::direct::InterleavedSlice;
use bytes::Bytes;
use cpal::Stream;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use pronghorn_audio::{AudioFormat, AudioFrame};
use rubato::{Fft, FixedSync, Resampler as _};
use tokio::sync::mpsc;
use tracing::{error, info};

/// Start capturing audio from the default input device.
///
/// Captures at the device's native rate/format, resamples to 16kHz mono f32
/// via rubato (high-quality FFT-based), converts to i16 PCM AudioFrames.
pub fn start_capture(audio_tx: mpsc::Sender<AudioFrame>) -> Result<Stream, AudioIoError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(AudioIoError::NoInputDevice)?;

    info!(device = ?device.description(), "using input device");

    let default = device
        .default_input_config()
        .map_err(|_| AudioIoError::Stream(cpal::BuildStreamError::StreamConfigNotSupported))?;

    let native_rate = default.sample_rate();
    let native_channels = default.channels();

    info!(
        sample_rate = native_rate,
        channels = native_channels,
        format = ?default.sample_format(),
        "capture: using device native config, resampling to 16kHz"
    );

    let config = cpal::StreamConfig {
        channels: native_channels,
        sample_rate: native_rate,
        buffer_size: cpal::BufferSize::Default,
    };

    // Shared state: accumulator + resampler
    let needs_resample = native_rate != 16_000;
    let resampler = if needs_resample {
        // rubato chunk size: 20ms at native rate
        let chunk_size = (native_rate as usize * 20) / 1000;
        Some(Arc::new(Mutex::new(
            Fft::<f32>::new(
                native_rate as usize,
                16_000,
                chunk_size,
                1, // sub_chunks
                1, // mono (we downmix first)
                FixedSync::Input,
            )
            .expect("valid resampler params"),
        )))
    } else {
        None
    };

    let accumulator = Arc::new(Mutex::new(Vec::<f32>::with_capacity(4096)));

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _info: &cpal::InputCallbackInfo| {
            // Step 1: Downmix to mono
            let mono: Vec<f32> = if native_channels > 1 {
                data.chunks(native_channels as usize)
                    .map(|ch| ch.iter().sum::<f32>() / native_channels as f32)
                    .collect()
            } else {
                data.to_vec()
            };

            // Step 2: Resample to 16kHz (or pass through if already 16kHz)
            let resampled = if let Some(ref rs) = resampler {
                let mut acc = accumulator.lock().unwrap();
                acc.extend_from_slice(&mono);

                let mut rs = rs.lock().unwrap();
                let chunk_size = rs.input_frames_max();
                let mut output = Vec::new();

                while acc.len() >= chunk_size {
                    let chunk = &acc[..chunk_size];
                    let adapter =
                        InterleavedSlice::new(chunk, 1, chunk.len()).unwrap();
                    match rs.process(&adapter, 0, None) {
                        Ok(result) => {
                            output.extend_from_slice(&result.take_data());
                        }
                        Err(e) => {
                            error!("capture resample error: {e}");
                            break;
                        }
                    }
                    acc.drain(..chunk_size);
                }

                output
            } else {
                mono
            };

            // Step 3: Convert f32 → i16 PCM and chunk into 20ms frames (320 samples)
            let frame_samples = 320;
            // Use a static accumulator for partial frames across callbacks
            // (reusing the same pattern — accumulate resampled i16 samples)
            let samples_i16: Vec<i16> = resampled
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();

            // We need to accumulate across callbacks for framing
            // For simplicity, use a thread-local buffer
            thread_local! {
                static FRAME_BUF: std::cell::RefCell<Vec<i16>> = std::cell::RefCell::new(Vec::with_capacity(640));
            }

            FRAME_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                buf.extend_from_slice(&samples_i16);

                while buf.len() >= frame_samples {
                    let chunk: Vec<i16> = buf.drain(..frame_samples).collect();
                    let pcm_bytes: Vec<u8> =
                        chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
                    let frame =
                        AudioFrame::new(AudioFormat::SPEECH, Bytes::from(pcm_bytes));
                    let _ = audio_tx.try_send(frame);
                }
            });
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
/// Receives 16kHz mono i16 AudioFrames, upsamples to the device's native
/// rate via rubato, expands to device channel count.
pub fn start_playback(mut speaker_rx: mpsc::Receiver<AudioFrame>) -> Result<Stream, AudioIoError> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or(AudioIoError::NoOutputDevice)?;

    info!(device = ?device.description(), "using output device");

    let default = device
        .default_output_config()
        .map_err(|_| AudioIoError::Stream(cpal::BuildStreamError::StreamConfigNotSupported))?;

    let output_rate = default.sample_rate();
    let output_channels = default.channels();

    info!(
        sample_rate = output_rate,
        channels = output_channels,
        "playback config"
    );

    let config = cpal::StreamConfig {
        channels: output_channels,
        sample_rate: output_rate,
        buffer_size: cpal::BufferSize::Default,
    };

    // Resampler: 16kHz → native output rate
    let needs_resample = output_rate != 16_000;
    let resampler = if needs_resample {
        let chunk_size = (16_000usize * 20) / 1000; // 20ms at 16kHz = 320
        Some(Arc::new(Mutex::new(
            Fft::<f32>::new(
                16_000,
                output_rate as usize,
                chunk_size,
                1,
                1, // mono, expand channels after
                FixedSync::Input,
            )
            .expect("valid playback resampler params"),
        )))
    } else {
        None
    };

    // Channel for resampled+expanded f32 samples ready for playback
    let (sample_tx, sample_rx) = std::sync::mpsc::channel::<Vec<f32>>();

    // Task: receive AudioFrames → resample → expand channels → send to playback
    let rs_clone = resampler.clone();
    tokio::spawn(async move {
        while let Some(frame) = speaker_rx.recv().await {
            // Convert i16 → f32 mono
            let mono: Vec<f32> = frame
                .samples
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();

            // Resample 16kHz → native rate
            let resampled = if let Some(ref rs) = rs_clone {
                let mut rs = rs.lock().unwrap();
                let chunk_size = rs.input_frames_max();
                let mut output = Vec::new();

                // Pad if needed (last chunk)
                let mut input = mono;
                if input.len() < chunk_size {
                    input.resize(chunk_size, 0.0);
                }

                for chunk in input.chunks(chunk_size) {
                    let mut padded;
                    let data = if chunk.len() < chunk_size {
                        padded = chunk.to_vec();
                        padded.resize(chunk_size, 0.0);
                        &padded[..]
                    } else {
                        chunk
                    };
                    let adapter = InterleavedSlice::new(data, 1, data.len()).unwrap();
                    match rs.process(&adapter, 0, None) {
                        Ok(result) => output.extend_from_slice(&result.take_data()),
                        Err(e) => {
                            tracing::error!("playback resample error: {e}");
                            break;
                        }
                    }
                }
                output
            } else {
                mono
            };

            // Expand mono → device channels (duplicate samples)
            let expanded: Vec<f32> = resampled
                .iter()
                .flat_map(|&s| std::iter::repeat_n(s, output_channels as usize))
                .collect();

            if sample_tx.send(expanded).is_err() {
                break;
            }
        }
    });

    let mut pending: Vec<f32> = Vec::new();

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _info: &cpal::OutputCallbackInfo| {
            let mut written = 0;

            // Drain pending
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
