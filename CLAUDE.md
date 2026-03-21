# Pronghorn

Fast, low-latency voice assistant protocol. Native UDP streaming replacement for Home Assistant's Wyoming protocol.

## Build & Test

```bash
cargo build --workspace                                   # build all crates
cargo test --workspace                                    # run all tests
cargo clippy --workspace --all-targets -- -D warnings     # lint
cargo fmt --all                                           # format
cargo build -p pronghorn-satellite                        # satellite binary only
cargo build -p pronghorn-server                           # server binary only
```

## Workspace Crates

### Libraries
- `pronghorn-audio` — Audio types: format descriptors, PCM frames, ring buffer
- `pronghorn-wake` — Wake word detection: pluggable trait, reframer, config (rustpotter behind feature flag)
- `pronghorn-wire` — Wire protocol: packet codec, UDP transport, session management
- `pronghorn-pipeline` — Pipeline stage traits: STT, TTS, Intent (echo backends for dev)

### Binaries
- `pronghorn-satellite` — Thin satellite: wake word + audio streaming (for Pi Zero 2)
- `pronghorn-server` — Server: pipeline orchestration (STT → Intent → TTS)

## Architecture

- Pure Rust, async with tokio
- UDP wire-level streaming for minimal latency (not chunked HTTP/WebSocket like Wyoming)
- Always-open connections to eliminate handshake overhead
- Separate satellite/server binaries — satellite is dependency-minimal for Pi Zero 2
- Pipeline stages connected by tokio mpsc channels for streaming with backpressure
- External STT/TTS services (faster-whisper, Piper) via persistent connections

## Wire Protocol

Packet types: Hello, Welcome, Keepalive, Audio, Control.
Audio: 20ms frames of 16kHz/16-bit/mono PCM (640 bytes per frame).
Session lifecycle: Connected → Listening → Processing → Speaking → Connected.

## Code Style

- Rust 2024 edition
- `thiserror` for error types
- `tracing` for structured logging (not `log`/`println!`)
- Run `cargo fmt` and `cargo clippy --workspace --all-targets -- -D warnings` before committing
