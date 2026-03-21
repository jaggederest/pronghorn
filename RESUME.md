# Pronghorn Session Resume Context

Last session: ~30 hours, 2026-03-21. Went from empty directory to working voice assistant.

## What Pronghorn Is

A low-latency voice assistant protocol in Rust, replacing Home Assistant's Wyoming protocol. Native UDP streaming instead of chunked HTTP/WebSocket. Separate satellite (Pi Zero 2) and server (Pi 5 / desktop) binaries.

## Current Architecture

```
Satellite (pronghorn-satellite):
  mic → cpal (native rate) → rubato resample → 16kHz AudioFrames
  → RingBuffer (15 frames pre-roll, sends last 6 on wake)
  → rustpotter wake word ("hey jarvis", .rpw reference file)
  → energy-based VAD (threshold 200, 500ms silence = stop)
  → UDP stream to server

Server (pronghorn-server):
  UDP recv → JitterBuffer (with gap-skip on packet loss)
  → Sherpa-ONNX Zipformer STT (offline batch, 115ms for 5s audio, 52x realtime)
  → Ollama LLM intent (gemma3:1b, ~500ms, "You are Jarvis" persona)
  → Kokoro TTS via kokoroxide (quantized int8, 2x realtime, bm_daniel voice)
  → UDP stream back → satellite speaker playback
```

## Workspace Layout

```
crates/
  pronghorn-audio/       — AudioFormat, AudioFrame, RingBuffer, Resampler
  pronghorn-wake/        — WakeWordDetector trait, rustpotter backend (feature-gated)
  pronghorn-wire/        — UDP packets, Transport, SessionManager, JitterBuffer
  pronghorn-pipeline/    — STT/TTS/Intent traits + backends (sherpa, whisper, kokoro, ollama)
  pronghorn-satellite/   — Thin binary: wake + audio I/O + UDP streaming
  pronghorn-server/      — Server binary: pipeline orchestration
```

## Feature Flags

```bash
# Satellite
cargo build -p pronghorn-satellite --release --features rustpotter

# Server (current best combo)
cargo build -p pronghorn-server --release --features sherpa,kokoro,ollama

# Also available but slower/alternative:
#   whisper  — Whisper-base via rusty-whisper/tract (5s latency, use sherpa instead)
#   sherpa   — Sherpa-ONNX Zipformer STT (115ms latency, recommended)
#   kokoro   — Kokoro TTS via kokoroxide/ort (2x realtime, recommended)
#   ollama   — Ollama LLM intent (gemma3:1b)
```

## Running

```bash
# Terminal 1: server
RUST_LOG=info ./target/release/pronghorn-server

# Terminal 2: satellite
RUST_LOG=info ./target/release/pronghorn-satellite
```

Config: `server.toml` and `satellite.toml` in project root.

NOTE: sherpa-rs needs `install_name_tool -add_rpath @executable_path target/release/pronghorn-server` on macOS after building.

## Model Files (in models/, gitignored)

- `sherpa-zipformer-en-offline/` — encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt (int8, ~71MB)
- `kokoro/` — model_quantized.onnx (92MB), tokenizer.json, voices/bm_daniel.bin
- `whisper-base/` — encoder, decoder, mel_filters, tokenizer, pos_emb (unused if using sherpa)
- `hey_jarvis.rpw` — wake word reference (8 recordings)

## Forks (on jaggederest GitHub)

- **jaggederest/tract** (unpin-half branch) — changes `half = "=2.4.1"` to `>= 2.4, <3` for candle/tract coexistence
- **jaggederest/rusty-whisper** — tract 0.20→0.21, patches tract via unpin-half fork
- **jaggederest/kokoroxide** — ort 1.16→2.0, ndarray 0.15→0.17, rodio removed
- **jaggederest/rustpotter** — candle-core 0.2→0.9.1, fixes rand conflict

## Key Bugs Fixed

- Keepalive ping-pong loop (satellite echoed server's echo)
- JitterBuffer stall on packet loss (skip after 5 stalls, flush on stream end)
- Session address spoofing (validate sender matches session)
- seq_before half-range bug (`<= 32768` → `< 32768`)
- Audio I/O: integer decimation replaced with rubato FFT resampling
- Whisper language config was hardcoded to "en"
- Resampler tail chunk zero-padding

## Current Performance

| Stage | Latency |
|-------|---------|
| VAD silence detect | ~500ms |
| Sherpa STT | ~115ms |
| Ollama LLM | ~500ms |
| Kokoro TTS | ~2.1s for ~4s audio |
| Total (end of speech → TTS complete) | ~3.2s |

## TODO / Next Steps (see TODO.md)

1. **Streaming STT** via sherpa-onnx online C API (partial transcripts every ~320ms)
2. **Word-trie intent resolver** from HA sentence templates (instant command execution, no LLM)
3. **HA WebSocket integration** (entity discovery, call_service)
4. **Sherpa Kokoro TTS consolidation** (waiting for int8 model from sherpa-onnx)
5. **Listening sounds** ("Yes?" / "Got it." via pre-generated Kokoro clips)
6. VAD improvements (Silero VAD), pre-roll wake word screening, barge-in
