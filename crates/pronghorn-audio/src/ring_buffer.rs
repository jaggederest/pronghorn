use std::collections::VecDeque;

use crate::AudioFrame;

/// Recommended pre-roll capacity: 15 frames at 20ms = 300ms lookback.
/// Covers wake word detector latency and avoids clipping the start of commands.
pub const DEFAULT_PREROLL_FRAMES: usize = 15;

/// Default frames to discard from the start of the pre-roll after wake word detection.
/// 25 frames × 20ms = 500ms — covers short wake words like "Hey Pronghorn".
pub const DEFAULT_WAKE_WORD_DISCARD_FRAMES: usize = 25;

/// Fixed-capacity ring buffer for audio pre-roll.
///
/// On the client (satellite) side, audio frames are pushed continuously.
/// When the buffer is full, the oldest frame is silently dropped.
/// On wake word detection the buffer is drained to recover the audio
/// that was captured just before the trigger — the "pre-roll".
pub struct RingBuffer {
    frames: VecDeque<AudioFrame>,
    capacity: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a frame, dropping the oldest if at capacity.
    pub fn push(&mut self, frame: AudioFrame) {
        if self.frames.len() == self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    /// Drain all buffered frames (oldest first) for pre-roll transmission.
    pub fn drain(&mut self) -> Vec<AudioFrame> {
        self.frames.drain(..).collect()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;
    use crate::AudioFormat;

    fn make_frame(id_byte: u8) -> AudioFrame {
        // Use a distinctive byte so we can tell frames apart
        AudioFrame::new(AudioFormat::SPEECH, Bytes::from(vec![id_byte; 640]))
    }

    #[test]
    fn basic_push_drain() {
        let mut rb = RingBuffer::new(5);
        rb.push(make_frame(1));
        rb.push(make_frame(2));
        assert_eq!(rb.len(), 2);

        let frames = rb.drain();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].samples[0], 1);
        assert_eq!(frames[1].samples[0], 2);
        assert!(rb.is_empty());
    }

    #[test]
    fn overflow_drops_oldest() {
        let mut rb = RingBuffer::new(3);
        rb.push(make_frame(1));
        rb.push(make_frame(2));
        rb.push(make_frame(3));
        rb.push(make_frame(4)); // drops frame 1
        rb.push(make_frame(5)); // drops frame 2

        assert_eq!(rb.len(), 3);
        let frames = rb.drain();
        assert_eq!(frames[0].samples[0], 3);
        assert_eq!(frames[1].samples[0], 4);
        assert_eq!(frames[2].samples[0], 5);
    }

    #[test]
    fn drain_is_idempotent() {
        let mut rb = RingBuffer::new(3);
        rb.push(make_frame(1));
        let _ = rb.drain();
        let frames = rb.drain();
        assert!(frames.is_empty());
    }

    #[test]
    fn fifteen_frame_preroll() {
        // Simulate the actual use case: 15-frame pre-roll at 20ms = 300ms lookback.
        // 300ms covers wake word detector latency and catches the start of commands.
        let mut rb = RingBuffer::new(15);
        for i in 0..40u8 {
            rb.push(make_frame(i));
        }
        assert_eq!(rb.len(), 15);

        let frames = rb.drain();
        // Should have frames 25..40 (the last 15)
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.samples[0], (25 + i) as u8);
        }
    }

    /// Verify the wake word discard logic used in the satellite event loop:
    ///   `preroll.into_iter().skip(discard.min(preroll.len()))`
    ///
    /// Tests the arithmetic that prevents wake word audio from reaching STT.
    #[test]
    fn wake_word_discard() {
        // Normal case: 150-frame pre-roll, 25-frame discard → 125 frames sent
        let mut rb = RingBuffer::new(150);
        for i in 0u8..150 {
            rb.push(make_frame(i));
        }
        let frames = rb.drain();
        let discard = 25usize.min(frames.len());
        let sent: Vec<_> = frames.into_iter().skip(discard).collect();
        assert_eq!(sent.len(), 125, "should send 150 - 25 = 125 frames");
        assert_eq!(
            sent[0].samples[0], 25,
            "first sent frame should be frame #25"
        );

        // Edge: discard >= buffer → send nothing
        let mut rb2 = RingBuffer::new(10);
        for i in 0u8..5 {
            rb2.push(make_frame(i));
        }
        let frames2 = rb2.drain();
        let discard2 = 10usize.min(frames2.len()); // clamps to 5
        let sent2: Vec<_> = frames2.into_iter().skip(discard2).collect();
        assert!(
            sent2.is_empty(),
            "discard >= preroll.len() should send nothing"
        );

        // Edge: discard = 0 → send all frames (no skip at all)
        let mut rb3 = RingBuffer::new(5);
        for i in 0u8..5 {
            rb3.push(make_frame(i));
        }
        let frames3 = rb3.drain();
        let discard3 = 0usize.min(frames3.len());
        let sent3: Vec<_> = frames3.into_iter().skip(discard3).collect();
        assert_eq!(sent3.len(), 5, "zero discard should send all frames");
    }
}
