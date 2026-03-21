# TODO

## Whisper STT: fix tract/time dep chain for `--features whisper`

`rusty-whisper` v0.1.3 is wired up behind the `whisper` feature flag. The code handles audio collection, temp WAV writing, and inference dispatch. However, `rusty-whisper` → `tract-onnx` → `time 0.3.23` doesn't compile on Rust 1.93+ (type inference issue in `time`).

**Options to unblock:**
1. Wait for `rusty-whisper` to update tract (tract 0.21+ should fix it)
2. Build Whisper inference from scratch using `mel_spec` + `ort` (bypasses tract entirely)
3. Fork `rusty-whisper` and bump tract

The implementation in `whisper.rs` is complete — it just needs a working dep chain.

File: `crates/pronghorn-pipeline/src/whisper.rs`

## Kokoro TTS: phonemization + ONNX inference

The `KokoroTts` backend has the 24→16kHz resampling pipeline wired up, but synthesis is stubbed:

1. **Phonemization** — Convert text → phoneme token IDs. Kokoro expects pre-phonemized input. May need espeak-ng bindings or a pure-Rust phonemizer.
2. **Voice embedding** — Load voice .npy/.bin files from the voices directory.
3. **ONNX inference** — Run Kokoro model with token IDs + voice embedding → f32 audio at 24kHz.

File: `crates/pronghorn-pipeline/src/kokoro.rs`
