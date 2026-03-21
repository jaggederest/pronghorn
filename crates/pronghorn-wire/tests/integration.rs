use std::net::SocketAddr;
use std::time::Instant;

use bytes::Bytes;
use pronghorn_wire::*;

fn localhost(port: u16) -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], port))
}

// ── handshake ───────────────────────────────────────────────────────

#[tokio::test]
async fn handshake_hello_welcome() {
    let server = Transport::bind(localhost(0)).await.unwrap();
    let client = Transport::bind(localhost(0)).await.unwrap();
    let server_addr = server.local_addr().unwrap();

    // Client sends Hello
    let hello = Packet::Hello(Hello { client_version: 1 });
    client.send_to(&hello, server_addr).await.unwrap();

    // Server receives it
    let (pkt, client_addr) = server.recv_from().await.unwrap();
    let client_ver = match pkt {
        Packet::Hello(h) => h.client_version,
        other => panic!("expected Hello, got {other:?}"),
    };
    assert_eq!(client_ver, 1);

    // Server responds with Welcome
    let mut sessions = SessionManager::new();
    let sid = sessions.create(client_addr);
    let welcome = Packet::Welcome(Welcome {
        session_id: sid,
        server_version: PROTOCOL_VERSION as u32,
    });
    server.send_to(&welcome, client_addr).await.unwrap();

    // Client receives Welcome
    let (pkt, _) = client.recv_from().await.unwrap();
    match pkt {
        Packet::Welcome(w) => {
            assert_eq!(w.session_id, sid);
        }
        other => panic!("expected Welcome, got {other:?}"),
    }
}

// ── audio round-trip ────────────────────────────────────────────────

#[tokio::test]
async fn audio_packet_fidelity() {
    let server = Transport::bind(localhost(0)).await.unwrap();
    let client = Transport::bind(localhost(0)).await.unwrap();
    let server_addr = server.local_addr().unwrap();

    // Generate a 20ms frame of synthetic audio (640 bytes at 16kHz/16-bit/mono).
    // Sine wave at 440 Hz so the bytes aren't all zeros.
    let samples: Vec<i16> = (0..320)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            (f32::sin(2.0 * std::f32::consts::PI * 440.0 * t) * i16::MAX as f32) as i16
        })
        .collect();
    let payload: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let payload = Bytes::from(payload);

    let audio = Packet::Audio(AudioData {
        session_id: 1,
        sequence: 0,
        timestamp: 0,
        payload: payload.clone(),
    });
    client.send_to(&audio, server_addr).await.unwrap();

    let (pkt, _) = server.recv_from().await.unwrap();
    match pkt {
        Packet::Audio(a) => {
            assert_eq!(a.session_id, 1);
            assert_eq!(a.sequence, 0);
            assert_eq!(a.timestamp, 0);
            assert_eq!(
                a.payload, payload,
                "audio payload must survive the round-trip bit-exact"
            );
        }
        other => panic!("expected Audio, got {other:?}"),
    }
}

// ── multi-packet audio stream ordering ──────────────────────────────

#[tokio::test]
async fn audio_stream_sequence() {
    let server = Transport::bind(localhost(0)).await.unwrap();
    let client = Transport::bind(localhost(0)).await.unwrap();
    let server_addr = server.local_addr().unwrap();

    let frame = Bytes::from(vec![0u8; 640]);

    // Send 50 frames (1 second of audio at 20ms frames)
    for seq in 0u16..50 {
        let pkt = Packet::Audio(AudioData {
            session_id: 1,
            sequence: seq,
            timestamp: seq as u32 * 320, // 320 samples per 20ms frame
            payload: frame.clone(),
        });
        client.send_to(&pkt, server_addr).await.unwrap();
    }

    // Receive all 50 and verify sequence numbers are present
    let mut received = Vec::new();
    for _ in 0..50 {
        let (pkt, _) = server.recv_from().await.unwrap();
        if let Packet::Audio(a) = pkt {
            received.push(a.sequence);
        }
    }

    // On localhost UDP shouldn't drop or reorder, but we only assert all arrived
    assert_eq!(received.len(), 50);
    received.sort();
    let expected: Vec<u16> = (0..50).collect();
    assert_eq!(received, expected);
}

// ── control flow ────────────────────────────────────────────────────

#[tokio::test]
async fn control_start_stop_listening() {
    let server = Transport::bind(localhost(0)).await.unwrap();
    let client = Transport::bind(localhost(0)).await.unwrap();
    let server_addr = server.local_addr().unwrap();

    // Client signals wake word detected
    let start = Packet::Control(Control {
        session_id: 1,
        control_type: ControlType::StartListening,
        payload: Bytes::new(),
    });
    client.send_to(&start, server_addr).await.unwrap();

    let (pkt, _) = server.recv_from().await.unwrap();
    match pkt {
        Packet::Control(c) => assert_eq!(c.control_type, ControlType::StartListening),
        other => panic!("expected Control, got {other:?}"),
    }

    // Server signals end of speech
    let client_addr = client.local_addr().unwrap();
    let stop = Packet::Control(Control {
        session_id: 1,
        control_type: ControlType::StopListening,
        payload: Bytes::new(),
    });
    server.send_to(&stop, client_addr).await.unwrap();

    let (pkt, _) = client.recv_from().await.unwrap();
    match pkt {
        Packet::Control(c) => assert_eq!(c.control_type, ControlType::StopListening),
        other => panic!("expected Control, got {other:?}"),
    }
}

// ── session management ──────────────────────────────────────────────

#[test]
fn session_manager_lifecycle() {
    let mut mgr = SessionManager::new();
    let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();

    let id = mgr.create(addr);
    assert_eq!(id, 1);

    let session = mgr.get(id).unwrap();
    assert_eq!(session.state, SessionState::Connected);
    assert_eq!(session.remote_addr, addr);

    // Lookup by address
    let session = mgr.get_by_addr(&addr).unwrap();
    assert_eq!(session.id, id);

    // State transition
    mgr.get_mut(id).unwrap().state = SessionState::Listening;
    assert_eq!(mgr.get(id).unwrap().state, SessionState::Listening);

    // Remove
    let removed = mgr.remove(id).unwrap();
    assert_eq!(removed.id, id);
    assert!(mgr.get(id).is_none());
    assert!(mgr.get_by_addr(&addr).is_none());
}

#[test]
fn session_manager_reap_stale() {
    let mut mgr = SessionManager::new();
    let addr1: SocketAddr = "127.0.0.1:1001".parse().unwrap();
    let addr2: SocketAddr = "127.0.0.1:1002".parse().unwrap();

    let _id1 = mgr.create(addr1);
    let id2 = mgr.create(addr2);

    // Touch only session 2
    mgr.touch(id2);

    // Reap everything older than "now" (session 1 was created before touch)
    // Since both were created nearly simultaneously, use a future deadline
    let deadline = Instant::now() + std::time::Duration::from_secs(1);
    let reaped = mgr.reap_stale(deadline);
    // Both are older than a second from now
    assert_eq!(reaped.len(), 2);
}

// ── latency sanity check ────────────────────────────────────────────

#[tokio::test]
async fn localhost_round_trip_under_1ms() {
    let server = Transport::bind(localhost(0)).await.unwrap();
    let client = Transport::bind(localhost(0)).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let client_addr = client.local_addr().unwrap();

    // Warm up
    let ping = Packet::Keepalive(Keepalive { session_id: 1 });
    client.send_to(&ping, server_addr).await.unwrap();
    let _ = server.recv_from().await.unwrap();

    // Measure
    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        client.send_to(&ping, server_addr).await.unwrap();
        let (pkt, _) = server.recv_from().await.unwrap();
        server.send_to(&pkt, client_addr).await.unwrap();
        let _ = client.recv_from().await.unwrap();
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() / iterations;

    // On localhost this should be well under 1ms per round-trip
    assert!(
        avg_us < 1000,
        "average round-trip {avg_us}µs exceeds 1ms — something is wrong"
    );
    eprintln!("average keepalive round-trip: {avg_us}µs over {iterations} iterations");
}
