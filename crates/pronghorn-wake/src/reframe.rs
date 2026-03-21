/// Adapts a stream of fixed-size audio chunks to a different chunk size.
///
/// Pronghorn produces 20ms frames (640 bytes at 16kHz/16-bit/mono).
/// Rustpotter needs 30ms chunks (960 bytes at the same format).
/// The reframer accumulates input bytes and yields complete output chunks.
///
/// Cadence for 640→960: after 3 input frames (1920 bytes) we get exactly
/// 2 output chunks (1920 bytes). No audio is lost or duplicated.
pub struct Reframer {
    buffer: Vec<u8>,
    output_size: usize,
}

impl Reframer {
    pub fn new(output_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(output_size * 2),
            output_size,
        }
    }

    /// Push audio bytes and return an iterator over complete output chunks.
    pub fn push(&mut self, data: &[u8]) -> ReframerIter<'_> {
        self.buffer.extend_from_slice(data);
        ReframerIter { reframer: self }
    }

    /// Discard any partial accumulated data.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Bytes currently buffered (partial, not yet a full output chunk).
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }
}

pub struct ReframerIter<'a> {
    reframer: &'a mut Reframer,
}

impl<'a> Iterator for ReframerIter<'a> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.reframer.buffer.len() >= self.reframer.output_size {
            let chunk = self.reframer.buffer[..self.reframer.output_size].to_vec();
            self.reframer.buffer.drain(..self.reframer.output_size);
            Some(chunk)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_640_to_960() {
        let mut rf = Reframer::new(960);

        // Frame 1: 640 bytes → no output yet
        let chunks: Vec<_> = rf.push(&[1u8; 640]).collect();
        assert!(chunks.is_empty());
        assert_eq!(rf.buffered(), 640);

        // Frame 2: 640 bytes (total 1280) → one 960-byte chunk, 320 remain
        let chunks: Vec<_> = rf.push(&[2u8; 640]).collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 960);
        assert_eq!(rf.buffered(), 320);
        // First 640 bytes should be 1s, next 320 should be 2s
        assert!(chunks[0][..640].iter().all(|&b| b == 1));
        assert!(chunks[0][640..].iter().all(|&b| b == 2));

        // Frame 3: 640 bytes (total 960) → one 960-byte chunk, 0 remain
        let chunks: Vec<_> = rf.push(&[3u8; 640]).collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 960);
        assert_eq!(rf.buffered(), 0);
        // First 320 bytes from frame 2 leftover, then 640 from frame 3
        assert!(chunks[0][..320].iter().all(|&b| b == 2));
        assert!(chunks[0][320..].iter().all(|&b| b == 3));
    }

    #[test]
    fn three_frames_yield_two_chunks() {
        let mut rf = Reframer::new(960);
        let mut total_chunks = 0;

        for _ in 0..3 {
            total_chunks += rf.push(&[0u8; 640]).count();
        }
        assert_eq!(total_chunks, 2);
        assert_eq!(rf.buffered(), 0);
    }

    #[test]
    fn six_frames_yield_four_chunks() {
        let mut rf = Reframer::new(960);
        let mut total_chunks = 0;

        for _ in 0..6 {
            total_chunks += rf.push(&[0u8; 640]).count();
        }
        assert_eq!(total_chunks, 4);
        assert_eq!(rf.buffered(), 0);
    }

    #[test]
    fn reset_clears_partial() {
        let mut rf = Reframer::new(960);
        let _ = rf.push(&[0u8; 640]).count();
        assert_eq!(rf.buffered(), 640);
        rf.reset();
        assert_eq!(rf.buffered(), 0);
    }

    #[test]
    fn large_push_yields_multiple_chunks() {
        let mut rf = Reframer::new(960);
        // Push 2880 bytes at once → 3 chunks, 0 remain
        let chunks: Vec<_> = rf.push(&[0u8; 2880]).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(rf.buffered(), 0);
    }

    #[test]
    fn exact_size_push() {
        let mut rf = Reframer::new(960);
        let chunks: Vec<_> = rf.push(&[0u8; 960]).collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(rf.buffered(), 0);
    }
}
