# TODO

## End-to-end testing with real models

Both inference backends are implemented and feature-gated. Next step is testing with real model files:

1. **Whisper STT** (`--features whisper`): Download whisper-base ONNX models from `onnx-community/whisper-base` on HuggingFace to `models/whisper-base/`
2. **Kokoro TTS** (`--features kokoro`): Download Kokoro ONNX model + tokenizer from `onnx-community/Kokoro-82M-v1.0-ONNX` + voice files from `hexgrad/Kokoro-82M` to `models/kokoro/`

## Future work

- Real intent processing (beyond echo backend)
- Home Assistant direct device API integration
- mDNS satellite discovery
- Barge-in support (interrupt TTS playback with new wake word)
- Opus codec for compressed audio over WiFi
