# TODO

## Wire real backends into server orchestrator

The server currently hardcodes `EchoStt`/`EchoTts`/`EchoIntent` in the orchestrator. Wire up `create_stt`/`create_tts`/`create_intent` factory functions so the server config actually selects the backend. This is the last step before a live end-to-end demo.

File: `crates/pronghorn-server/src/orchestrator.rs`

## Unblock wake word detection

Rustpotter upstream has dep conflicts (candle-core/half). Options:
1. Fork and fix the one-line DType match exhaustiveness in the Priler fork
2. Wait for upstream PR #15 merge
3. Switch to a different wake word engine

Track: https://github.com/GiviMAD/rustpotter/pull/15
File: `crates/pronghorn-wake/Cargo.toml`

## Intent processing

The echo intent backend just parrots back the transcript. Need a real intent processor that can:
- Parse voice commands ("turn on the kitchen lights")
- Map to Home Assistant entity IDs and services
- Generate natural language responses

File: `crates/pronghorn-pipeline/src/intent.rs`

## Performance

- Whisper STT: 0.85x realtime on M-series (10s audio → 8.5s). Acceptable but could be faster with whisper-tiny or streaming.
- Kokoro TTS: 1.7x realtime with quantized model. fp32 model would improve quality (bm_daniel sounds slightly congested with int8).
- Consider whisper-tiny for simple home commands where accuracy is less critical.

## Future work

- Home Assistant direct device API integration (bypass conversation pipeline)
- mDNS satellite discovery (currently config-based server address)
- Barge-in support (interrupt TTS playback with new wake word)
- Opus codec for compressed audio over WiFi
- Voice activity detection (VAD) to auto-detect end of speech instead of manual StopListening
- Streaming STT (process audio as it arrives, not batch after StopListening)
