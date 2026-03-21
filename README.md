# 🦌 Pronghorn

> ~~🙈🙉🙊⌨️~~ — 100% pure AI-generated code. No humans were mass-harmed in the making of this codebase.

Fast, low-latency voice assistant protocol. Native UDP streaming replacement for [Home Assistant](https://www.home-assistant.io/)'s [Wyoming protocol](https://www.home-assistant.io/integrations/wyoming/).

## Why?

Wyoming uses chunked HTTP/WebSocket for audio data. Every voice interaction pays the cost of connection setup, serialization overhead, and buffering delays. Closed-source assistants (Alexa, Google Home) don't have this problem because they control the full stack.

Pronghorn uses **native UDP streaming** with always-open connections and a custom wire protocol designed for minimum latency. The goal is to make the open-source voice assistant *faster* than the closed-source ones.

## Architecture

```
┌─────────────────────┐         UDP          ┌─────────────────────────┐
│  Satellite (Pi Zero) │◄───────────────────►│  Server (HA host)        │
│                       │   wire protocol     │                         │
│  mic → ring buffer    │                     │  jitter buffer → STT    │
│  wake word detection  │                     │  intent processing      │
│  audio streaming      │                     │  TTS → audio streaming  │
│  speaker playback     │                     │                         │
└─────────────────────┘                      └─────────────────────────┘
```

- **20ms audio frames** (16kHz/16-bit/mono PCM, 640 bytes) — smaller than a single TCP handshake
- **8-byte packet header** — version, type, flags, session ID
- **Pre-roll ring buffer** — 300ms lookback for future wake-word-aware speech capture (currently discarded to avoid transcribing the wake word; see TODO for screening plan)
- **Jitter buffer** — sequence-ordered playout absorbs network timing variance
- **Pipeline stages** connected by async channels — STT, Intent, TTS stream data as it's produced

## Workspace

| Crate | Purpose |
|-------|---------|
| `pronghorn-audio` | Audio types, PCM frames, ring buffer |
| `pronghorn-wake` | Pluggable wake word detection (rustpotter backend) |
| `pronghorn-wire` | UDP wire protocol, transport, sessions, jitter buffer |
| `pronghorn-pipeline` | Pipeline stage traits: STT, TTS, Intent |
| `pronghorn-satellite` | Thin satellite binary for Pi Zero 2 |
| `pronghorn-server` | Server binary with pipeline orchestration |

## Build

```bash
cargo build --workspace
cargo test --workspace
cargo build -p pronghorn-satellite    # just the satellite
cargo build -p pronghorn-server       # just the server
```

## Status

Early development. The wire protocol, pipeline traits, and server event loop are functional with echo backends. See [CLAUDE.md](CLAUDE.md) for development notes.

## License

MIT OR Apache-2.0
