use std::collections::HashMap;

use crate::packet::AudioData;

/// Sequence-ordered playout buffer for incoming audio packets.
///
/// UDP doesn't guarantee ordering, so packets may arrive out of sequence.
/// The jitter buffer holds packets and releases them in order, absorbing
/// network jitter at the cost of a small fixed delay (`playout_delay` frames).
///
/// Typical values:
/// - LAN (Ethernet): playout_delay = 2–3 (40–60ms)
/// - WiFi: playout_delay = 4–5 (80–100ms)
pub struct JitterBuffer {
    packets: HashMap<u16, AudioData>,
    /// Next sequence number to play out.
    next_seq: Option<u16>,
    /// Number of frames to accumulate before starting playout.
    playout_delay: u16,
    state: State,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Accumulating initial frames before playout begins.
    Buffering { count: u16 },
    /// Actively releasing frames in sequence order.
    Playing,
}

impl JitterBuffer {
    /// Create a new jitter buffer.
    ///
    /// `playout_delay` is the number of frames to buffer before the first
    /// `pop()` will return anything. Higher values absorb more jitter but
    /// add latency.
    pub fn new(playout_delay: u16) -> Self {
        Self {
            packets: HashMap::new(),
            next_seq: None,
            playout_delay,
            state: State::Buffering { count: 0 },
        }
    }

    /// Insert a received packet.
    ///
    /// Duplicate and late packets (sequence < next_seq) are silently dropped.
    pub fn push(&mut self, packet: AudioData) {
        // First packet sets the baseline
        if self.next_seq.is_none() {
            self.next_seq = Some(packet.sequence);
        }

        // Drop packets that are behind the play cursor
        if let Some(next) = self.next_seq
            && seq_before(packet.sequence, next)
        {
            return;
        }

        self.packets.insert(packet.sequence, packet);

        if let State::Buffering { ref mut count } = self.state {
            *count += 1;
            if *count >= self.playout_delay {
                self.state = State::Playing;
            }
        }
    }

    /// Pop the next frame in sequence order, if available.
    ///
    /// Returns `None` if still buffering or if the next sequence hasn't
    /// arrived yet (gap in the stream).
    pub fn pop(&mut self) -> Option<AudioData> {
        if self.state != State::Playing {
            return None;
        }
        let seq = self.next_seq?;
        let packet = self.packets.remove(&seq)?;
        self.next_seq = Some(seq.wrapping_add(1));
        Some(packet)
    }

    /// Skip the next expected sequence number (declare it lost).
    ///
    /// Use this when you've waited long enough and need to move on.
    pub fn skip(&mut self) {
        if let Some(seq) = self.next_seq {
            self.packets.remove(&seq);
            self.next_seq = Some(seq.wrapping_add(1));
        }
    }

    /// The sequence number we're waiting on, if any.
    pub fn next_seq(&self) -> Option<u16> {
        self.next_seq
    }

    /// Number of packets currently buffered.
    pub fn buffered(&self) -> usize {
        self.packets.len()
    }

    /// Whether playout has started.
    pub fn is_playing(&self) -> bool {
        self.state == State::Playing
    }

    /// Reset to initial state (e.g., between listening sessions).
    pub fn reset(&mut self) {
        self.packets.clear();
        self.next_seq = None;
        self.state = State::Buffering { count: 0 };
    }
}

/// Returns true if `a` is strictly "before" `b` in sequence space (with u16 wrapping).
///
/// Uses the standard TCP-style comparison: a is before b if
/// `b.wrapping_sub(a)` is in the range `1..=32767`. The value 32768
/// (exactly half the space) is excluded to maintain strict ordering —
/// at that distance the relation would be true in both directions.
fn seq_before(a: u16, b: u16) -> bool {
    let diff = b.wrapping_sub(a);
    diff > 0 && diff < 32768
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    fn make_audio(seq: u16) -> AudioData {
        AudioData {
            session_id: 1,
            sequence: seq,
            flags: 0,
            timestamp: seq as u32 * 320,
            payload: Bytes::from(vec![seq as u8; 640]),
        }
    }

    #[test]
    fn in_order_delivery() {
        let mut jb = JitterBuffer::new(3);
        jb.push(make_audio(0));
        jb.push(make_audio(1));
        assert!(jb.pop().is_none(), "should still be buffering");

        jb.push(make_audio(2));
        assert!(jb.is_playing());
        assert_eq!(jb.pop().unwrap().sequence, 0);
        assert_eq!(jb.pop().unwrap().sequence, 1);
        assert_eq!(jb.pop().unwrap().sequence, 2);
        assert!(jb.pop().is_none());
    }

    #[test]
    fn out_of_order_reordering() {
        let mut jb = JitterBuffer::new(3);
        jb.push(make_audio(0));
        jb.push(make_audio(2)); // 1 is late
        jb.push(make_audio(1)); // 1 arrives

        assert_eq!(jb.pop().unwrap().sequence, 0);
        assert_eq!(jb.pop().unwrap().sequence, 1);
        assert_eq!(jb.pop().unwrap().sequence, 2);
    }

    #[test]
    fn missing_packet_blocks_then_skip() {
        let mut jb = JitterBuffer::new(2);
        jb.push(make_audio(0));
        jb.push(make_audio(2)); // 1 is missing

        assert_eq!(jb.pop().unwrap().sequence, 0);
        assert!(jb.pop().is_none(), "blocked waiting for seq 1");

        jb.skip(); // declare 1 lost
        assert_eq!(jb.pop().unwrap().sequence, 2);
    }

    #[test]
    fn duplicate_packet_ignored() {
        let mut jb = JitterBuffer::new(2);
        jb.push(make_audio(0));
        jb.push(make_audio(0)); // dupe
        jb.push(make_audio(1));

        assert_eq!(jb.pop().unwrap().sequence, 0);
        assert_eq!(jb.pop().unwrap().sequence, 1);
        assert!(jb.pop().is_none());
    }

    #[test]
    fn late_packet_dropped() {
        let mut jb = JitterBuffer::new(1);
        jb.push(make_audio(0));
        assert_eq!(jb.pop().unwrap().sequence, 0);

        // Seq 0 arrives again (retransmit or duplicate) — should be dropped
        jb.push(make_audio(0));
        assert!(jb.pop().is_none());
        assert_eq!(jb.buffered(), 0);
    }

    #[test]
    fn reset_clears_state() {
        let mut jb = JitterBuffer::new(2);
        jb.push(make_audio(0));
        jb.push(make_audio(1));
        assert!(jb.is_playing());

        jb.reset();
        assert!(!jb.is_playing());
        assert_eq!(jb.buffered(), 0);
        assert!(jb.next_seq().is_none());
    }

    #[test]
    fn sequence_wrapping() {
        let mut jb = JitterBuffer::new(2);
        jb.push(make_audio(u16::MAX - 1));
        jb.push(make_audio(u16::MAX));

        assert_eq!(jb.pop().unwrap().sequence, u16::MAX - 1);
        assert_eq!(jb.pop().unwrap().sequence, u16::MAX);

        // Next should be 0 (wrapped)
        jb.push(make_audio(0));
        // Need to reach playout again? No — once Playing, stays Playing.
        assert_eq!(jb.pop().unwrap().sequence, 0);
    }

    #[test]
    fn pre_roll_burst() {
        // Simulate the server receiving a burst of pre-roll frames followed by live
        let mut jb = JitterBuffer::new(3);

        // Client flushes 10 pre-roll frames in a burst
        for seq in 0..10u16 {
            jb.push(AudioData {
                flags: crate::packet::audio_flags::PRE_ROLL,
                ..make_audio(seq)
            });
        }

        // All should come out in order
        for expected_seq in 0..10u16 {
            let frame = jb.pop().unwrap();
            assert_eq!(frame.sequence, expected_seq);
            assert!(frame.is_pre_roll());
        }

        // Live frames continue
        jb.push(AudioData {
            flags: 0,
            ..make_audio(10)
        });
        let frame = jb.pop().unwrap();
        assert_eq!(frame.sequence, 10);
        assert!(!frame.is_pre_roll());
    }

    #[test]
    fn seq_before_basic() {
        assert!(super::seq_before(0, 1));
        assert!(super::seq_before(100, 200));
        assert!(!super::seq_before(1, 0));
        assert!(!super::seq_before(5, 5));
    }

    #[test]
    fn seq_before_wrapping() {
        // u16::MAX is "before" 0 in sequence space
        assert!(super::seq_before(u16::MAX, 0));
        assert!(super::seq_before(u16::MAX - 5, 3));
        // But 0 is NOT before u16::MAX (that would be going backwards)
        assert!(!super::seq_before(0, u16::MAX));
    }
}
