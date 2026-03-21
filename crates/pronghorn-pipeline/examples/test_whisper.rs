//! Test Whisper STT with a WAV file.
//!
//! Usage:
//!   cargo run -p pronghorn-pipeline --features whisper --example test_whisper -- models/test_16k.wav

use std::path::PathBuf;

use bytes::Bytes;
use pronghorn_audio::{AudioFormat, AudioFrame};
use pronghorn_pipeline::config::WhisperConfig;
use pronghorn_pipeline::stt::{SpeechToText, Transcript};
use pronghorn_pipeline::whisper::WhisperStt;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let wav_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/test_16k.wav".to_string());

    println!("Loading WAV: {wav_path}");

    // Read WAV file
    let mut reader = hound::WavReader::open(&wav_path)?;
    let spec = reader.spec();
    println!(
        "WAV: {}Hz, {} channels, {} bits",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );

    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    println!(
        "Samples: {} ({:.1}s)",
        samples.len(),
        samples.len() as f32 / spec.sample_rate as f32
    );

    // Create WhisperStt
    let config = WhisperConfig {
        model_dir: PathBuf::from("models/whisper-base"),
        language: "en".into(),
    };

    println!("Loading Whisper model...");
    let stt = WhisperStt::new(&config)?;

    // Feed audio as frames through the channel
    let (audio_tx, audio_rx) = mpsc::channel::<AudioFrame>(64);
    let (transcript_tx, mut transcript_rx) = mpsc::channel::<Transcript>(4);

    // Spawn STT
    let stt_handle = tokio::spawn(async move { stt.transcribe(audio_rx, transcript_tx).await });

    // Send audio as 20ms frames (320 samples = 640 bytes)
    let frame_samples = 320;
    for chunk in samples.chunks(frame_samples) {
        let pcm_bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame = AudioFrame::new(AudioFormat::SPEECH, Bytes::from(pcm_bytes));
        audio_tx.send(frame).await?;
    }
    drop(audio_tx); // signal end of stream

    println!("Transcribing...");
    stt_handle.await??;

    // Print transcript
    while let Some(transcript) = transcript_rx.recv().await {
        println!(
            "[{}] {}",
            if transcript.is_final {
                "FINAL"
            } else {
                "partial"
            },
            transcript.text
        );
    }

    Ok(())
}
