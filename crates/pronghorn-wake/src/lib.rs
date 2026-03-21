pub mod config;
pub mod detector;
pub mod error;
#[cfg(any(feature = "rustpotter", test))]
pub(crate) mod reframe;

#[cfg(feature = "rustpotter")]
pub mod rustpotter_backend;

pub use config::{RustpotterBackendConfig, WakeBackend, WakeConfig};
pub use detector::{Detection, WakeWordDetector};
pub use error::WakeError;

/// Create a boxed detector from config, dispatching on the backend enum.
pub fn create_detector(config: &WakeConfig) -> Result<Box<dyn WakeWordDetector>, WakeError> {
    match config.backend {
        WakeBackend::Rustpotter => {
            #[cfg(feature = "rustpotter")]
            {
                let det = rustpotter_backend::RustpotterDetector::new(&config.rustpotter)?;
                Ok(Box::new(det))
            }
            #[cfg(not(feature = "rustpotter"))]
            {
                Err(WakeError::BackendNotAvailable("rustpotter".into()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::time::Instant;

    use bytes::Bytes;
    use pronghorn_audio::{AudioFormat, AudioFrame, RingBuffer};

    use super::*;

    /// Mock detector with scripted detection sequence.
    struct MockDetector {
        detections: VecDeque<Option<Detection>>,
    }

    impl MockDetector {
        fn new(detections: Vec<Option<Detection>>) -> Self {
            Self {
                detections: detections.into(),
            }
        }
    }

    impl WakeWordDetector for MockDetector {
        fn process_frame(&mut self, _frame: &AudioFrame) -> Result<Option<Detection>, WakeError> {
            Ok(self.detections.pop_front().flatten())
        }

        fn reset(&mut self) {
            self.detections.clear();
        }
    }

    fn silence_frame() -> AudioFrame {
        AudioFrame::new(AudioFormat::SPEECH, Bytes::from(vec![0u8; 640]))
    }

    #[test]
    fn mock_detector_fires_on_schedule() {
        let mut det = MockDetector::new(vec![
            None,
            None,
            Some(Detection {
                wake_word: "hey_pronghorn".into(),
                score: 0.95,
                timestamp: Instant::now(),
            }),
        ]);

        assert!(det.process_frame(&silence_frame()).unwrap().is_none());
        assert!(det.process_frame(&silence_frame()).unwrap().is_none());
        let d = det.process_frame(&silence_frame()).unwrap().unwrap();
        assert_eq!(d.wake_word, "hey_pronghorn");
        assert!(d.score > 0.9);
    }

    #[test]
    fn mock_detector_reset_clears() {
        let mut det = MockDetector::new(vec![Some(Detection {
            wake_word: "test".into(),
            score: 1.0,
            timestamp: Instant::now(),
        })]);
        det.reset();
        // After reset, queue is empty → returns None
        assert!(det.process_frame(&silence_frame()).unwrap().is_none());
    }

    #[test]
    fn preroll_drain_on_detection() {
        // Simulate: push frames to ring buffer + mock detector, drain on detection
        let mut ring = RingBuffer::new(15);
        let mut det = MockDetector::new(vec![
            None,
            None,
            None,
            None,
            None, // 5 frames before detection
            Some(Detection {
                wake_word: "test".into(),
                score: 0.9,
                timestamp: Instant::now(),
            }),
        ]);

        let mut detected = false;
        let mut preroll_frames = Vec::new();

        for i in 0u8..10 {
            let frame = AudioFrame::new(AudioFormat::SPEECH, Bytes::from(vec![i; 640]));
            ring.push(frame.clone());

            if let Some(_d) = det.process_frame(&frame).unwrap() {
                preroll_frames = ring.drain();
                detected = true;
                det.reset();
                break;
            }
        }

        assert!(detected);
        // Ring buffer had frames 0..6 when detection fired on frame 5
        assert_eq!(preroll_frames.len(), 6);
        assert_eq!(preroll_frames[0].samples[0], 0);
        assert_eq!(preroll_frames[5].samples[0], 5);
    }

    #[test]
    fn trait_object_works() {
        let det: Box<dyn WakeWordDetector> = Box::new(MockDetector::new(vec![None]));
        // Just verify it compiles and can be boxed — object safety check
        drop(det);
    }
}
