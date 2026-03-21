use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler as _};

/// Resamples audio from one sample rate to another.
///
/// Used to convert Kokoro TTS output (24kHz) to our wire format (16kHz).
/// Uses rubato's FFT-based resampler for clean integer-ratio conversion.
pub struct Resampler {
    inner: Fft<f32>,
}

impl Resampler {
    /// Create a 24kHz → 16kHz resampler (for Kokoro TTS output).
    ///
    /// Chunk size of 480 = 20ms at 24kHz input. Produces 320 samples = 20ms at 16kHz.
    pub fn new_24k_to_16k() -> Self {
        let chunk_size = 480; // 20ms at 24kHz
        let inner = Fft::<f32>::new(
            24_000, // input rate
            16_000, // output rate
            chunk_size,
            1, // sub_chunks
            1, // mono
            FixedSync::Input,
        )
        .expect("valid resampler params for 24k→16k");

        Self { inner }
    }

    /// Resample a buffer of f32 mono samples.
    ///
    /// Input: mono f32 samples at 24kHz.
    /// Returns: mono f32 samples at 16kHz.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        let chunk_size = self.inner.input_frames_max();
        let mut output = Vec::with_capacity(input.len() * 2 / 3 + 64);

        for chunk in input.chunks(chunk_size) {
            // Pad the last chunk to the required size if needed
            let padded;
            let data = if chunk.len() < chunk_size {
                padded = {
                    let mut v = chunk.to_vec();
                    v.resize(chunk_size, 0.0);
                    v
                };
                &padded[..]
            } else {
                chunk
            };
            let adapter = InterleavedSlice::new(data, 1, data.len()).unwrap();

            match self.inner.process(&adapter, 0, None) {
                Ok(result) => {
                    let data = result.take_data();
                    output.extend_from_slice(&data);
                }
                Err(e) => {
                    tracing::error!("resample error: {e}");
                    break;
                }
            }
        }

        output
    }

    /// Reset the resampler's internal state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_24k_to_16k_produces_output() {
        let mut rs = Resampler::new_24k_to_16k();

        // 24000 samples at 24kHz = 1 second
        let input: Vec<f32> = (0..24000).map(|i| (i as f32 / 24000.0).sin()).collect();
        let output = rs.process(&input);

        // Should produce roughly 16000 samples (some tolerance for delay/padding)
        assert!(
            output.len() > 14000 && output.len() < 18000,
            "expected ~16000 samples, got {}",
            output.len()
        );
    }

    #[test]
    fn resample_empty_input() {
        let mut rs = Resampler::new_24k_to_16k();
        let output = rs.process(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn resample_preserves_silence() {
        let mut rs = Resampler::new_24k_to_16k();
        let input = vec![0.0f32; 24000];
        let output = rs.process(&input);

        for sample in &output {
            assert!(sample.abs() < 0.01, "expected silence, got {sample}");
        }
    }
}
