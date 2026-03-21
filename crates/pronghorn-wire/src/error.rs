use thiserror::Error;

#[derive(Debug, Error)]
pub enum WireError {
    #[error("packet too short: expected at least {expected} bytes, got {actual}")]
    PacketTooShort { expected: usize, actual: usize },

    #[error("unsupported protocol version: {0}")]
    UnsupportedVersion(u8),

    #[error("unknown packet type: 0x{0:02x}")]
    UnknownPacketType(u8),

    #[error("unknown control type: 0x{0:02x}")]
    UnknownControlType(u8),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
