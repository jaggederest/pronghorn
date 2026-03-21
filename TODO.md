# TODO

## Whisper STT: mel spectrogram + ONNX inference

The `WhisperStt` backend has audio collection and i16→f32 conversion wired up, but the actual inference pipeline is stubbed:

1. **Mel spectrogram preprocessing** — 80-bin log-mel spectrogram from 16kHz PCM. Requires FFT (rustfft is already in the dep tree) → mel filterbank → log scaling. Whisper uses 25ms window, 10ms hop.
2. **ONNX encoder/decoder** — Load Whisper ONNX model via `ort`, run encoder on mel features, autoregressive decoder to produce token IDs.
3. **Token decoding** — Map token IDs back to text. Need Whisper's tokenizer vocabulary.

File: `crates/pronghorn-pipeline/src/whisper.rs`

## Kokoro TTS: phonemization + ONNX inference

The `KokoroTts` backend has the 24→16kHz resampling pipeline wired up, but synthesis is stubbed:

1. **Phonemization** — Convert text → phoneme token IDs. Kokoro expects pre-phonemized input. May need espeak-ng bindings or a pure-Rust phonemizer.
2. **Voice embedding** — Load voice .npy/.bin files from the voices directory.
3. **ONNX inference** — Run Kokoro model with token IDs + voice embedding → f32 audio at 24kHz.

File: `crates/pronghorn-pipeline/src/kokoro.rs`
