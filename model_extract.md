For Claude Code to regenerate all 5 files, they'll need:
Install: pip install openai-whisper onnx (this pulls in torch and everything else — it's hefty)
The model itself downloads automatically when you call whisper.load_model("base") — it fetches the .pt checkpoint from OpenAI's CDN. No manual download needed.
From there, the extraction is roughly:

multilingual.tiktoken and mel_filters.npz — these ship as static assets inside the whisper package itself, so just copy them out of the installed package's assets/ directory. No extraction needed.
positional_embedding.npz — extract from the loaded model as I showed earlier (model.decoder.positional_embedding)
encoder.onnx and decoder.onnx — these need torch.onnx.export against the encoder and decoder submodules with the right input shapes and tensor names

That last part is the tricky one. The export needs to match what rusty-whisper's tract code expects for input/output tensor names and dimensions. Since you have the repo cloned, I'd point Claude Code at the Rust source (particularly wherever it loads the ONNX files and sets up input tensors) so it can match the expected interface. Otherwise you'll be right back in "inscrutable shape mismatch" territory.
