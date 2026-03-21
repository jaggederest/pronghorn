use bytes::Bytes;

use crate::format::AudioFormat;

/// A frame of raw PCM audio data with its format descriptor.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub format: AudioFormat,
    pub samples: Bytes,
}

impl AudioFrame {
    pub fn new(format: AudioFormat, samples: Bytes) -> Self {
        Self { format, samples }
    }

    /// Duration of this frame in milliseconds.
    pub fn duration_ms(&self) -> u32 {
        let bps = self.format.bytes_per_second();
        if bps == 0 {
            return 0;
        }
        (self.samples.len() as u64 * 1000 / bps as u64) as u32
    }

    /// Number of samples in this frame (total across all channels).
    pub fn sample_count(&self) -> usize {
        self.samples.len() / self.format.sample_size as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_duration() {
        let frame = AudioFrame::new(
            AudioFormat::SPEECH,
            Bytes::from(vec![0u8; 640]), // 20ms at 16kHz mono i16
        );
        assert_eq!(frame.duration_ms(), 20);
        assert_eq!(frame.sample_count(), 320);
    }
}
