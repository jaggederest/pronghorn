use std::net::SocketAddr;
use std::time::Duration;

use bytes::Bytes;
use pronghorn_audio::AudioConfig;
use pronghorn_pipeline::PipelineConfig;
use pronghorn_wire::*;

fn localhost(port: u16) -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], port))
}

// ── Full satellite-to-server flow ──────────────────────────────────

#[tokio::test]
async fn full_echo_pipeline_over_udp() {
    // Start a real server in the background
    let server_transport = Transport::bind(localhost(0)).await.unwrap();
    let server_addr = server_transport.local_addr().unwrap();

    let config = pronghorn_server::config::ServerConfig {
        audio: AudioConfig::default(),
        transport: TransportConfig {
            bind_address: server_addr,
            jitter_buffer_delay: 2,
            keepalive_interval_ms: 60_000,
            session_timeout_ms: 30_000,
        },
        pipeline: PipelineConfig::default(),
    };

    // We need the server to use the same socket. Since run_server binds its own,
    // let's drop ours and have the server bind to the same port.
    drop(server_transport);

    // Spawn the server
    let server_handle =
        tokio::spawn(async move { pronghorn_server::event_loop::run_server(config).await });

    // Give the server a moment to bind
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Client
    let client = Transport::bind(localhost(0)).await.unwrap();

    // 1. Handshake
    client
        .send_to(&Packet::Hello(Hello { client_version: 1 }), server_addr)
        .await
        .unwrap();

    let (pkt, _) = client.recv_from().await.unwrap();
    let session_id = match pkt {
        Packet::Welcome(w) => {
            assert_eq!(w.server_version, PROTOCOL_VERSION as u32);
            w.session_id
        }
        other => panic!("expected Welcome, got {other:?}"),
    };

    // 2. StartListening
    client
        .send_to(
            &Packet::Control(Control {
                session_id,
                control_type: ControlType::StartListening,
                payload: Bytes::new(),
            }),
            server_addr,
        )
        .await
        .unwrap();

    // 3. Send audio packets (5 pre-roll + 5 live = 10 total)
    for seq in 0u16..10 {
        let flags = if seq < 5 { audio_flags::PRE_ROLL } else { 0 };
        let pkt = Packet::Audio(AudioData {
            session_id,
            sequence: seq,
            flags,
            timestamp: seq as u32 * 320,
            payload: Bytes::from(vec![seq as u8; 640]),
        });
        client.send_to(&pkt, server_addr).await.unwrap();
    }

    // 4. StopListening
    client
        .send_to(
            &Packet::Control(Control {
                session_id,
                control_type: ControlType::StopListening,
                payload: Bytes::new(),
            }),
            server_addr,
        )
        .await
        .unwrap();

    // 5. Receive StartSpeaking
    let pkt = recv_with_timeout(&client, Duration::from_secs(2)).await;
    match pkt {
        Packet::Control(c) => {
            assert_eq!(c.control_type, ControlType::StartSpeaking);
            assert_eq!(c.session_id, session_id);
        }
        other => panic!("expected StartSpeaking, got {other:?}"),
    }

    // 6. Receive TTS audio frames
    // EchoStt produces "received 10 frames"
    // EchoIntent produces "You said: received 10 frames" (30 chars)
    // EchoTts produces 30 frames (1 per char)
    let mut tts_frame_count = 0u16;
    loop {
        let pkt = recv_with_timeout(&client, Duration::from_secs(2)).await;
        match pkt {
            Packet::Audio(a) => {
                assert_eq!(a.session_id, session_id);
                tts_frame_count += 1;
            }
            Packet::Control(c) if c.control_type == ControlType::StopSpeaking => {
                break;
            }
            other => panic!("expected Audio or StopSpeaking, got {other:?}"),
        }
    }

    // "You said: received 10 frames" = 28 characters = 28 TTS frames
    assert_eq!(
        tts_frame_count, 28,
        "expected 28 TTS frames for echo response"
    );

    // Clean up
    server_handle.abort();
}

async fn recv_with_timeout(transport: &Transport, timeout: Duration) -> Packet {
    match tokio::time::timeout(timeout, transport.recv_from()).await {
        Ok(Ok((pkt, _))) => pkt,
        Ok(Err(e)) => panic!("recv error: {e}"),
        Err(_) => panic!("recv timed out after {timeout:?}"),
    }
}
