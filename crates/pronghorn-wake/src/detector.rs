use std::time::Instant;

use pronghorn_audio::AudioFrame;

use crate::error::WakeError;

/// A wake word detection event.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Name/key of the wake word that matched.
    pub wake_word: String,
    /// Confidence score (0.0–1.0).
    pub score: f32,
    /// When the detection occurred.
    pub timestamp: Instant,
}

/// Pluggable wake word detection backend.
///
/// Implementations consume 20ms `AudioFrame`s and may signal detections.
/// The trait is deliberately **synchronous** — backends perform CPU-bound
/// inference, and callers run them on a dedicated thread with channels.
///
/// Object-safe: can be used as `Box<dyn WakeWordDetector>`.
pub trait WakeWordDetector: Send {
    /// Feed a single audio frame to the detector.
    ///
    /// Returns `Some(Detection)` when a wake word is spotted.
    /// Returns `None` on every other frame.
    fn process_frame(&mut self, frame: &AudioFrame) -> Result<Option<Detection>, WakeError>;

    /// Reset internal state (call after detection or between sessions).
    fn reset(&mut self);
}
