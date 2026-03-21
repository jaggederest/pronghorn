# Pronghorn

Fast, low-latency voice assistant protocol. Native UDP streaming replacement for Home Assistant's Wyoming protocol.

## Build & Test

```bash
cargo build                    # build all crates
cargo test --workspace         # run all tests (unit + integration)
cargo clippy --workspace --all-targets -- -D warnings  # lint
cargo fmt --all                # format
cargo run                      # run the main binary
```

## Workspace Crates

- `pronghorn-audio` — Audio types: format descriptors, PCM frames
- `pronghorn-wire` — Wire protocol: packet codec, UDP transport, session management

## Architecture

- Pure Rust, async with tokio
- UDP wire-level streaming for minimal latency (not chunked HTTP/WebSocket like Wyoming)
- Always-open connections to eliminate handshake overhead
- 8-byte fixed packet header, big-endian, zero-copy codec via `bytes`

## Wire Protocol

Packet types: Hello, Welcome, Keepalive, Audio, Control.
Audio: 20ms frames of 16kHz/16-bit/mono PCM (640 bytes per frame).
Session lifecycle: Connected → Listening → Processing → Speaking → Connected.

## Code Style

- Rust 2024 edition
- `thiserror` for error types
- `tracing` for structured logging (not `log`/`println!`)
- Run `cargo fmt` and `cargo clippy --workspace --all-targets -- -D warnings` before committing
