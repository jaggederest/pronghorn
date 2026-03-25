use serde::{Deserialize, Serialize};

use crate::format::AudioFormat;
use crate::ring_buffer::{DEFAULT_PREROLL_FRAMES, DEFAULT_WAKE_WORD_DISCARD_FRAMES};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Audio format for capture and playback.
    pub format: AudioFormat,
    /// Number of 20ms frames to keep in the pre-roll ring buffer.
    pub preroll_frames: usize,
    /// Number of 20ms frames to discard from the start of the pre-roll after wake word
    /// detection. Skips the wake word itself so it is not transcribed by STT.
    /// Set to 0 to disable discarding.
    pub wake_word_discard_frames: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            format: AudioFormat::SPEECH,
            preroll_frames: DEFAULT_PREROLL_FRAMES,
            wake_word_discard_frames: DEFAULT_WAKE_WORD_DISCARD_FRAMES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips_through_toml() {
        let config = AudioConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: AudioConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.format, config.format);
        assert_eq!(parsed.preroll_frames, config.preroll_frames);
        assert_eq!(
            parsed.wake_word_discard_frames,
            config.wake_word_discard_frames
        );
    }

    #[test]
    fn partial_toml_uses_defaults() {
        let toml_str = r#"preroll_frames = 20"#;
        let parsed: AudioConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(parsed.preroll_frames, 20);
        assert_eq!(parsed.format, AudioFormat::SPEECH); // defaulted
        assert_eq!(
            parsed.wake_word_discard_frames,
            DEFAULT_WAKE_WORD_DISCARD_FRAMES
        ); // defaulted
    }

    #[test]
    fn wake_word_discard_frames_is_configurable() {
        let toml_str = r#"wake_word_discard_frames = 0"#;
        let parsed: AudioConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(parsed.wake_word_discard_frames, 0);
    }
}
