use bytes::{Buf, BufMut, Bytes, BytesMut};

use crate::error::WireError;
use crate::packet::*;

impl Packet {
    /// Serialize this packet into `buf`.
    pub fn encode(&self, buf: &mut BytesMut) {
        match self {
            Packet::Hello(h) => {
                put_header(buf, PacketType::Hello, 0, 0);
                buf.put_u32(h.client_version);
            }
            Packet::Welcome(w) => {
                put_header(buf, PacketType::Welcome, 0, w.session_id);
                buf.put_u32(w.server_version);
            }
            Packet::Keepalive(k) => {
                put_header(buf, PacketType::Keepalive, 0, k.session_id);
            }
            Packet::Audio(a) => {
                put_header(buf, PacketType::Audio, a.flags, a.session_id);
                buf.put_u16(a.sequence);
                buf.put_u16(0); // reserved
                buf.put_u32(a.timestamp);
                buf.put_slice(&a.payload);
            }
            Packet::Control(c) => {
                put_header(buf, PacketType::Control, 0, c.session_id);
                buf.put_u8(c.control_type as u8);
                buf.put_u8(0); // reserved
                buf.put_u16(0); // reserved
                buf.put_slice(&c.payload);
            }
        }
    }

    /// Deserialize a packet from raw bytes.
    pub fn decode(mut buf: Bytes) -> Result<Self, WireError> {
        if buf.remaining() < HEADER_SIZE {
            return Err(WireError::PacketTooShort {
                expected: HEADER_SIZE,
                actual: buf.remaining(),
            });
        }

        let version = buf.get_u8();
        if version != PROTOCOL_VERSION {
            return Err(WireError::UnsupportedVersion(version));
        }

        let packet_type = buf.get_u8();
        let flags = buf.get_u8();
        let _reserved = buf.get_u8();
        let session_id = buf.get_u32();

        match packet_type {
            x if x == PacketType::Hello as u8 => {
                ensure_remaining(&buf, 4, "Hello")?;
                Ok(Packet::Hello(Hello {
                    client_version: buf.get_u32(),
                }))
            }
            x if x == PacketType::Welcome as u8 => {
                ensure_remaining(&buf, 4, "Welcome")?;
                Ok(Packet::Welcome(Welcome {
                    session_id,
                    server_version: buf.get_u32(),
                }))
            }
            x if x == PacketType::Keepalive as u8 => {
                Ok(Packet::Keepalive(Keepalive { session_id }))
            }
            x if x == PacketType::Audio as u8 => {
                ensure_remaining(&buf, 8, "Audio")?;
                let sequence = buf.get_u16();
                let _reserved = buf.get_u16();
                let timestamp = buf.get_u32();
                let payload = buf.copy_to_bytes(buf.remaining());
                Ok(Packet::Audio(AudioData {
                    session_id,
                    sequence,
                    flags,
                    timestamp,
                    payload,
                }))
            }
            x if x == PacketType::Control as u8 => {
                ensure_remaining(&buf, 4, "Control")?;
                let control_type_byte = buf.get_u8();
                let _reserved1 = buf.get_u8();
                let _reserved2 = buf.get_u16();
                let control_type = ControlType::try_from(control_type_byte)?;
                let payload = buf.copy_to_bytes(buf.remaining());
                Ok(Packet::Control(Control {
                    session_id,
                    control_type,
                    payload,
                }))
            }
            _ => Err(WireError::UnknownPacketType(packet_type)),
        }
    }
}

fn put_header(buf: &mut BytesMut, ptype: PacketType, flags: u8, session_id: u32) {
    buf.put_u8(PROTOCOL_VERSION);
    buf.put_u8(ptype as u8);
    buf.put_u8(flags);
    buf.put_u8(0); // reserved
    buf.put_u32(session_id);
}

fn ensure_remaining(buf: &Bytes, needed: usize, context: &str) -> Result<(), WireError> {
    if buf.remaining() < needed {
        return Err(WireError::PacketTooShort {
            expected: HEADER_SIZE + needed,
            actual: HEADER_SIZE + buf.remaining(),
        });
    }
    let _ = context; // used only if we want richer errors later
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(packet: &Packet) -> Packet {
        let mut buf = BytesMut::with_capacity(1500);
        packet.encode(&mut buf);
        Packet::decode(buf.freeze()).unwrap()
    }

    #[test]
    fn hello_round_trip() {
        let original = Packet::Hello(Hello { client_version: 42 });
        let decoded = round_trip(&original);
        match decoded {
            Packet::Hello(h) => assert_eq!(h.client_version, 42),
            other => panic!("expected Hello, got {other:?}"),
        }
    }

    #[test]
    fn welcome_round_trip() {
        let original = Packet::Welcome(Welcome {
            session_id: 7,
            server_version: 1,
        });
        let decoded = round_trip(&original);
        match decoded {
            Packet::Welcome(w) => {
                assert_eq!(w.session_id, 7);
                assert_eq!(w.server_version, 1);
            }
            other => panic!("expected Welcome, got {other:?}"),
        }
    }

    #[test]
    fn keepalive_round_trip() {
        let original = Packet::Keepalive(Keepalive { session_id: 3 });
        let decoded = round_trip(&original);
        match decoded {
            Packet::Keepalive(k) => assert_eq!(k.session_id, 3),
            other => panic!("expected Keepalive, got {other:?}"),
        }
    }

    #[test]
    fn audio_round_trip() {
        let payload = Bytes::from(vec![0xABu8; 640]);
        let original = Packet::Audio(AudioData {
            session_id: 1,
            sequence: 99,
            flags: 0,
            timestamp: 16000,
            payload: payload.clone(),
        });
        let decoded = round_trip(&original);
        match decoded {
            Packet::Audio(a) => {
                assert_eq!(a.session_id, 1);
                assert_eq!(a.sequence, 99);
                assert_eq!(a.timestamp, 16000);
                assert_eq!(a.payload, payload);
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn control_round_trip() {
        let original = Packet::Control(Control {
            session_id: 5,
            control_type: ControlType::StartListening,
            payload: Bytes::new(),
        });
        let decoded = round_trip(&original);
        match decoded {
            Packet::Control(c) => {
                assert_eq!(c.session_id, 5);
                assert_eq!(c.control_type, ControlType::StartListening);
                assert!(c.payload.is_empty());
            }
            other => panic!("expected Control, got {other:?}"),
        }
    }

    #[test]
    fn decode_too_short() {
        let buf = Bytes::from(vec![0u8; 3]);
        assert!(Packet::decode(buf).is_err());
    }

    #[test]
    fn decode_bad_version() {
        let mut buf = BytesMut::with_capacity(12);
        buf.put_u8(0xFF); // bad version
        buf.put_u8(PacketType::Keepalive as u8);
        buf.put_u8(0);
        buf.put_u8(0);
        buf.put_u32(0);
        assert!(Packet::decode(buf.freeze()).is_err());
    }
}
