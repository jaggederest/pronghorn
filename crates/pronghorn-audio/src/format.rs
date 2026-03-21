/// Audio sample format descriptor.
///
/// Defines the shape of audio data: sample rate, channel count, and bytes per sample.
/// All wire protocol audio packets reference a format so both sides agree on how
/// to interpret the raw PCM bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AudioFormat {
    /// Samples per second (e.g. 16000 for 16 kHz)
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Bytes per sample per channel (2 for i16)
    pub sample_size: u16,
}

impl AudioFormat {
    /// 16 kHz mono 16-bit signed — the standard for speech recognition pipelines.
    pub const SPEECH: Self = Self {
        sample_rate: 16_000,
        channels: 1,
        sample_size: 2,
    };

    /// Bytes per second of raw PCM audio in this format.
    pub const fn bytes_per_second(&self) -> u32 {
        self.sample_rate * self.channels as u32 * self.sample_size as u32
    }

    /// Bytes of PCM data in a frame of the given duration.
    pub const fn frame_bytes(&self, duration_ms: u32) -> usize {
        (self.bytes_per_second() * duration_ms / 1000) as usize
    }

    /// Total sample count (across all channels) in a frame of the given duration.
    pub const fn frame_samples(&self, duration_ms: u32) -> usize {
        (self.sample_rate * duration_ms / 1000) as usize * self.channels as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_format_math() {
        let f = AudioFormat::SPEECH;
        assert_eq!(f.bytes_per_second(), 32_000);
        // 20ms frame at 16kHz mono i16 = 320 samples = 640 bytes
        assert_eq!(f.frame_samples(20), 320);
        assert_eq!(f.frame_bytes(20), 640);
    }
}
