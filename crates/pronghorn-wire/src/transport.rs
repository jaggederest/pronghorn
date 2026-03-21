use std::net::SocketAddr;

use bytes::{Bytes, BytesMut};
use tokio::net::UdpSocket;

use crate::error::WireError;
use crate::packet::Packet;

/// Maximum safe UDP payload size (Ethernet MTU 1500 − IP 20 − UDP 8).
pub const MAX_PACKET_SIZE: usize = 1472;

/// Thin async wrapper around a UDP socket that speaks the Pronghorn wire protocol.
pub struct Transport {
    socket: UdpSocket,
}

impl Transport {
    /// Bind to the given address.  Use port 0 to let the OS pick.
    pub async fn bind(addr: SocketAddr) -> Result<Self, WireError> {
        let socket = UdpSocket::bind(addr).await?;
        Ok(Self { socket })
    }

    /// Wrap an already-bound tokio `UdpSocket`.
    pub fn from_socket(socket: UdpSocket) -> Self {
        Self { socket }
    }

    pub fn local_addr(&self) -> Result<SocketAddr, WireError> {
        Ok(self.socket.local_addr()?)
    }

    /// Encode and send a packet to `addr`.
    pub async fn send_to(&self, packet: &Packet, addr: SocketAddr) -> Result<usize, WireError> {
        let mut buf = BytesMut::with_capacity(MAX_PACKET_SIZE);
        packet.encode(&mut buf);
        let sent = self.socket.send_to(&buf, addr).await?;
        Ok(sent)
    }

    /// Receive and decode the next packet.  Returns the packet and the sender address.
    pub async fn recv_from(&self) -> Result<(Packet, SocketAddr), WireError> {
        let mut buf = vec![0u8; MAX_PACKET_SIZE];
        let (len, addr) = self.socket.recv_from(&mut buf).await?;
        let bytes = Bytes::copy_from_slice(&buf[..len]);
        let packet = Packet::decode(bytes)?;
        Ok((packet, addr))
    }
}
