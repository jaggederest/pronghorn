# TODO

## Listening/acknowledgment sounds

Play audio cues on the satellite when the pipeline state changes:
- **Wake word detected → "Yes?"** (short Kokoro TTS clip, pre-generated)
- **StopListening/processing complete → "Got it."** (short Kokoro TTS clip, pre-generated)

Pre-generate the clips with Kokoro (bm_daniel voice) and bundle as WAV files.
Play them locally on the satellite without round-tripping through the server.
This matches the UX of the HA satellite pipeline's beep sounds but with
a natural voice instead of a tone.

File: `crates/pronghorn-satellite/src/event_loop.rs`, `crates/pronghorn-satellite/src/audio_io.rs`

## Pre-roll wake word screening

Currently pre-roll is discarded because it contains the tail of the wake word
(e.g., "hey jarvis" gets transcribed as "service" by Whisper). Future approach:
capture a longer pre-roll (e.g., 1 second), estimate the wake word duration
(~800ms for "hey jarvis"), discard that prefix, send the remaining frames.
This allows natural speech that overlaps with the wake word to be captured.

File: `crates/pronghorn-satellite/src/event_loop.rs`

## Intent processing

The echo intent backend just parrots back the transcript. Need a real intent processor that can:
- Parse voice commands ("turn on the kitchen lights")
- Map to Home Assistant entity IDs and services
- Generate natural language responses

File: `crates/pronghorn-pipeline/src/intent.rs`

## Voice activity detection (VAD)

Replace the 5-second streaming timeout with real VAD to detect when the user
stops speaking. Options:
- Simple energy-based: stop when RMS drops below threshold for N frames
- Silero VAD (small ONNX model, very accurate)
- WebRTC VAD

File: `crates/pronghorn-satellite/src/event_loop.rs`

## Performance

- Whisper STT: 0.85x realtime on M-series (10s audio → 8.5s). Try whisper-tiny for simple commands.
- Kokoro TTS: 1.7x realtime with quantized model. fp32 model improves voice quality.
- Resample error on last TTS chunk — resampler needs tail-flush handling.

## Future work

- Home Assistant direct device API integration (bypass conversation pipeline)
- mDNS satellite discovery (currently config-based server address)
- Barge-in support (interrupt TTS playback with new wake word)
- Opus codec for compressed audio over WiFi
- Streaming STT (process audio as it arrives, not batch after StopListening)
