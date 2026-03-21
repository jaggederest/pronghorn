//! Test Kokoro TTS with text input, save output as WAV.
//!
//! Usage:
//!   cargo run -p pronghorn-pipeline --features kokoro --example test_kokoro -- "Hello, I am Pronghorn."

use std::path::PathBuf;

use pronghorn_audio::AudioFrame;
use pronghorn_pipeline::config::KokoroConfig;
use pronghorn_pipeline::kokoro::KokoroTts;
use pronghorn_pipeline::tts::TextToSpeech;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello, I am Pronghorn, the fast voice assistant.".to_string());

    println!("Text: {text}");

    let config = KokoroConfig {
        model_path: PathBuf::from("models/kokoro/model_quantized.onnx"),
        tokenizer_path: PathBuf::from("models/kokoro/tokenizer.json"),
        voices_path: PathBuf::from("models/kokoro/voices"),
        voice: "bm_daniel".into(),
    };

    println!("Loading Kokoro model...");
    let tts = KokoroTts::new(&config)?;

    let (audio_tx, mut audio_rx) = mpsc::channel::<AudioFrame>(256);

    println!("Synthesizing...");
    let start = std::time::Instant::now();
    tts.synthesize(&text, audio_tx).await?;
    let elapsed = start.elapsed();

    // Collect all frames
    let mut all_samples: Vec<i16> = Vec::new();
    while let Ok(frame) = audio_rx.try_recv() {
        for chunk in frame.samples.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            all_samples.push(sample);
        }
    }

    let duration_s = all_samples.len() as f32 / 16_000.0;
    println!(
        "Generated {} samples ({:.1}s of audio) in {:.1}s ({:.1}x realtime)",
        all_samples.len(),
        duration_s,
        elapsed.as_secs_f32(),
        duration_s / elapsed.as_secs_f32()
    );

    // Save as WAV
    let output_path = "models/kokoro_output.wav";
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for &sample in &all_samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    println!("Saved to {output_path}");

    Ok(())
}
