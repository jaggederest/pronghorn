use std::time::Instant;

use pronghorn_audio::AudioFrame;
use rustpotter::{
    AudioFmt, DetectorConfig, Endianness, Rustpotter, RustpotterConfig, SampleFormat,
};
use tracing::debug;

use crate::config::RustpotterBackendConfig;
use crate::detector::{Detection, WakeWordDetector};
use crate::error::WakeError;
use crate::reframe::Reframer;

/// Wake word detector backed by the Rustpotter engine.
pub struct RustpotterDetector {
    engine: Rustpotter,
    reframer: Reframer,
}

impl RustpotterDetector {
    pub fn new(config: &RustpotterBackendConfig) -> Result<Self, WakeError> {
        if config.wake_words.is_empty() {
            return Err(WakeError::NoWakeWords);
        }

        let rp_config = RustpotterConfig {
            fmt: AudioFmt {
                sample_rate: 16_000,
                sample_format: SampleFormat::I16,
                channels: 1,
                endianness: Endianness::Little,
            },
            detector: DetectorConfig {
                threshold: config.threshold,
                avg_threshold: config.avg_threshold,
                eager: config.eager,
                min_scores: config.min_scores,
                ..DetectorConfig::default()
            },
            ..RustpotterConfig::default()
        };

        let mut engine =
            Rustpotter::new(&rp_config).map_err(|e| WakeError::BackendInit(e.to_string()))?;

        for (key, path) in &config.wake_words {
            let path_str = path.to_string_lossy();
            engine
                .add_wakeword_from_file(key, &path_str)
                .map_err(|e| WakeError::ModelLoad {
                    path: path.clone(),
                    reason: e.to_string(),
                })?;
            debug!(wake_word = key, path = %path_str, "loaded wake word");
        }

        let bytes_per_frame = engine.get_bytes_per_frame();

        Ok(Self {
            engine,
            reframer: Reframer::new(bytes_per_frame),
        })
    }
}

impl WakeWordDetector for RustpotterDetector {
    fn process_frame(&mut self, frame: &AudioFrame) -> Result<Option<Detection>, WakeError> {
        for chunk in self.reframer.push(&frame.samples) {
            if let Some(det) = self.engine.process_bytes(&chunk) {
                return Ok(Some(Detection {
                    wake_word: det.name,
                    score: det.score,
                    timestamp: Instant::now(),
                }));
            }
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.engine.reset();
        self.reframer.reset();
    }
}
