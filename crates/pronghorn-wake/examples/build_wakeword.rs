//! Build a rustpotter .rpw wake word reference file from WAV samples.
//!
//! Usage:
//!   cargo run -p pronghorn-wake --features rustpotter --example build_wakeword -- \
//!     --name "hey_jarvis" \
//!     --output models/hey_jarvis.rpw \
//!     models/wake/16k/sample1.wav models/wake/16k/sample2.wav ...

use rustpotter::WakewordRefBuildFromFiles;

struct Builder;
impl WakewordRefBuildFromFiles for Builder {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut name = "hey_jarvis".to_string();
    let mut output = "models/hey_jarvis.rpw".to_string();
    let mut wav_files: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                i += 1;
                name = args[i].clone();
            }
            "--output" => {
                i += 1;
                output = args[i].clone();
            }
            _ => {
                wav_files.push(args[i].clone());
            }
        }
        i += 1;
    }

    if wav_files.is_empty() {
        eprintln!("Usage: build_wakeword [--name NAME] [--output PATH] file1.wav file2.wav ...");
        std::process::exit(1);
    }

    println!(
        "Building wake word reference '{name}' from {} samples",
        wav_files.len()
    );
    for f in &wav_files {
        println!("  {f}");
    }

    let wakeword = Builder::new_from_sample_files(
        name.clone(),
        None, // use default threshold
        None, // use default avg_threshold
        wav_files,
        42, // mfcc_size (rustpotter default)
    )?;

    use rustpotter::WakewordSave;
    wakeword.save_to_file(&output)?;

    println!("Saved to {output}");
    Ok(())
}
