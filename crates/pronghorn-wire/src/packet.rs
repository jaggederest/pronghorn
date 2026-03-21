use bytes::Bytes;

use crate::error::WireError;

pub const PROTOCOL_VERSION: u8 = 1;

/// Size of the fixed packet header in bytes.
///
/// Layout:
///   [0]     protocol version
///   [1]     packet type discriminant
///   [2]     flags (per-type)
///   [3]     reserved
///   [4..8]  session id (big-endian u32, 0 for Hello)
pub const HEADER_SIZE: usize = 8;

// ── packet type discriminants ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketType {
    Hello = 0x01,
    Welcome = 0x02,
    Keepalive = 0x03,
    Audio = 0x10,
    Control = 0x20,
}

// ── top-level packet enum ───────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Packet {
    Hello(Hello),
    Welcome(Welcome),
    Keepalive(Keepalive),
    Audio(AudioData),
    Control(Control),
}

// ── individual packet types ─────────────────────────────────────────

/// Sent by a client to initiate a session.
#[derive(Debug, Clone)]
pub struct Hello {
    pub client_version: u32,
}

/// Server response to Hello, assigns a session id.
#[derive(Debug, Clone)]
pub struct Welcome {
    pub session_id: u32,
    pub server_version: u32,
}

/// Bidirectional heartbeat to keep the session alive.
#[derive(Debug, Clone, Copy)]
pub struct Keepalive {
    pub session_id: u32,
}

/// A chunk of raw PCM audio.
///
/// Wire layout after the 8-byte header:
///   [8..10]   sequence number (big-endian u16)
///   [10..12]  reserved
///   [12..16]  timestamp in samples (big-endian u32)
///   [16..]    PCM payload
#[derive(Debug, Clone)]
pub struct AudioData {
    pub session_id: u32,
    pub sequence: u16,
    /// Timestamp in sample units. At 16 kHz this wraps every ~74 hours.
    pub timestamp: u32,
    pub payload: Bytes,
}

/// Session lifecycle and pipeline control signals.
///
/// Wire layout after the 8-byte header:
///   [8]       control type discriminant
///   [9..12]   reserved
///   [12..]    optional payload
#[derive(Debug, Clone)]
pub struct Control {
    pub session_id: u32,
    pub control_type: ControlType,
    pub payload: Bytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ControlType {
    /// Client detected a wake word and is about to stream audio.
    StartListening = 0x01,
    /// End of speech input.
    StopListening = 0x02,
    /// Server is about to stream TTS audio back.
    StartSpeaking = 0x03,
    /// Server finished streaming TTS audio.
    StopSpeaking = 0x04,
    /// Something went wrong.
    Error = 0xFE,
    /// Graceful session teardown.
    SessionEnd = 0xFF,
}

impl TryFrom<u8> for ControlType {
    type Error = WireError;

    fn try_from(value: u8) -> Result<Self, <Self as TryFrom<u8>>::Error> {
        match value {
            0x01 => Ok(Self::StartListening),
            0x02 => Ok(Self::StopListening),
            0x03 => Ok(Self::StartSpeaking),
            0x04 => Ok(Self::StopSpeaking),
            0xFE => Ok(Self::Error),
            0xFF => Ok(Self::SessionEnd),
            _ => Err(WireError::UnknownControlType(value)),
        }
    }
}
