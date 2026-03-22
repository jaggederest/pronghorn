use bytes::Bytes;

#[cfg(feature = "sherpa")]
use crate::config::VadConfig;
#[cfg(feature = "sherpa")]
use crate::error::PipelineError;

/// Convert 16-bit little-endian PCM bytes to f32 samples in [-1.0, 1.0].
pub fn i16_bytes_to_f32(bytes: &Bytes) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect()
}

#[cfg(feature = "sherpa")]
pub use sherpa_rs::silero_vad::SileroVad;

/// Create a Silero VAD instance from config.
#[cfg(feature = "sherpa")]
pub fn create_vad(config: &VadConfig) -> Result<sherpa_rs::silero_vad::SileroVad, PipelineError> {
    use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};

    let model_path = config.model_path.to_string_lossy().to_string();
    let vad_config = SileroVadConfig {
        model: model_path,
        min_silence_duration: config.min_silence_duration,
        min_speech_duration: config.min_speech_duration,
        max_speech_duration: 15.0, // match satellite hard timeout
        threshold: config.threshold,
        sample_rate: 16000,
        window_size: 512,
        ..SileroVadConfig::default()
    };

    // Buffer 30s of audio internally (matches satellite hard timeout with margin)
    let vad = SileroVad::new(vad_config, 30.0)
        .map_err(|e| PipelineError::Backend(format!("failed to create Silero VAD: {e}")))?;

    tracing::info!(
        model = %config.model_path.display(),
        threshold = config.threshold,
        min_silence = config.min_silence_duration,
        "Silero VAD initialized"
    );

    Ok(vad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i16_bytes_conversion() {
        // Silence (zeros)
        let silence = Bytes::from(vec![0u8; 640]);
        let samples = i16_bytes_to_f32(&silence);
        assert_eq!(samples.len(), 320);
        assert!(samples.iter().all(|&s| s == 0.0));

        // Max positive i16 = 32767
        let max_pos = Bytes::from(vec![0xFF, 0x7F]);
        let samples = i16_bytes_to_f32(&max_pos);
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 32767.0 / 32768.0).abs() < 1e-5);

        // Min negative i16 = -32768
        let min_neg = Bytes::from(vec![0x00, 0x80]);
        let samples = i16_bytes_to_f32(&min_neg);
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - (-1.0)).abs() < 1e-5);
    }
}
