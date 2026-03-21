# TODO

## Next: Swap Whisper for Sherpa-ONNX streaming STT

Replace rusty-whisper (batch, ~5s latency) with sherpa-onnx's streaming Zipformer transducer (~320ms chunks, partial transcripts). The 20M-param model is 27MB on disk and runs at RTF <0.05 on Pi 5.

Integration path: `sherpa-rs` crate (wraps C API) or direct FFI to sherpa-onnx. Start in batch mode for the swap, then enable true streaming with partial transcripts.

See: `docs/streaming-stt-research.md` for full analysis.

## Progressive intent resolver (word-trie + LLM fallback)

Build a word-level trie from Home Assistant's Hassil sentence templates. As words arrive from streaming STT, walk the trie to resolve commands deterministically in microseconds. Fall back to LLM only when the pattern goes off-script.

Target: simple commands ("turn on the kitchen lights") execute in <500ms from end of speech — no LLM round-trip.

See: `docs/streaming-stt-research.md` Part 2 for architecture.

## Home Assistant WebSocket integration

Connect to HA via WebSocket to:
- Fetch entity registry (names, areas, aliases) for fuzzy matching
- Execute actions via `call_service`
- Subscribe to entity updates

## Listening/acknowledgment sounds

Play "Yes?" (wake detected) and "Got it." (processing) via pre-generated Kokoro clips on the satellite. Play locally without server round-trip.

## Pre-roll wake word screening

Capture longer pre-roll (~1s), discard wake-word-length prefix, send remaining frames. Avoids both clipping the command start and transcribing the wake word.

## VAD improvements

- Current energy-based VAD works but threshold needs per-mic calibration
- Consider Silero VAD (2MB ONNX) for more robust endpoint detection
- Sherpa-ONNX bundles Silero VAD if we switch STT

## Whisper-tiny support (if keeping Whisper as fallback)

rusty-whisper hardcodes 6 decoder layers (whisper-base). Tiny has 4, small has 12. Fork needs parameterization by layer count.

## Performance notes

- Whisper-base STT: ~5s for ~2.5s audio (M-series). Bottleneck.
- Kokoro TTS: 1.7x realtime with quantized model. fp32 improves quality.
- Resampler tail chunk fixed (zero-pad).
- VAD: 500ms silence detection working at threshold 200.

## Future work

- Moonshine v2 streaming ONNX exports — benchmark against Zipformer when available
- Opus codec for compressed audio over WiFi
- Barge-in support (interrupt TTS with new wake word)
- Multiple satellite support (different wake word samples per user/location)
