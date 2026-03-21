#!/usr/bin/env python3
"""
Extract Whisper model files for use with rusty-whisper/pronghorn.

Produces 5 files in the output directory:
  - encoder_model.onnx
  - decoder_model_merged.onnx
  - multilingual.tiktoken
  - positional_embedding.npz
  - mel_filters.npz

Usage:
  pip install openai-whisper onnx
  python scripts/extract_whisper_model.py --model base --output models/whisper-base

The model size can be: tiny, base, small, medium, large
"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import whisper


def main():
    parser = argparse.ArgumentParser(description="Extract Whisper model for rusty-whisper")
    parser.add_argument("--model", default="base", help="Model size: tiny, base, small, medium, large")
    parser.add_argument("--output", default="models/whisper-base", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading whisper model '{args.model}'...")
    model = whisper.load_model(args.model)
    model.eval()

    # 1. Copy static assets from whisper package
    whisper_dir = os.path.dirname(whisper.__file__)
    assets_dir = os.path.join(whisper_dir, "assets")

    tiktoken_src = os.path.join(assets_dir, "multilingual.tiktoken")
    tiktoken_dst = os.path.join(args.output, "multilingual.tiktoken")
    if os.path.exists(tiktoken_src):
        shutil.copy2(tiktoken_src, tiktoken_dst)
        print(f"Copied {tiktoken_dst}")
    else:
        print(f"WARNING: {tiktoken_src} not found, checking alternatives...")
        # Try gpt2.tiktoken for English-only models
        for name in os.listdir(assets_dir):
            if name.endswith(".tiktoken"):
                shutil.copy2(os.path.join(assets_dir, name), os.path.join(args.output, name))
                print(f"  Copied {name}")

    mel_src = os.path.join(assets_dir, "mel_filters.npz")
    mel_dst = os.path.join(args.output, "mel_filters.npz")
    if os.path.exists(mel_src):
        shutil.copy2(mel_src, mel_dst)
        print(f"Copied {mel_dst}")
    else:
        print(f"WARNING: {mel_src} not found")

    # 2. Extract positional embedding
    pos_emb = model.decoder.positional_embedding.detach().cpu().numpy()
    pos_emb_dst = os.path.join(args.output, "positional_embedding.npz")
    np.savez(pos_emb_dst, pos_emb)
    print(f"Saved {pos_emb_dst} shape={pos_emb.shape}")

    # 3. Export encoder to ONNX
    print("Exporting encoder to ONNX...")
    encoder = model.encoder
    n_mels = model.dims.n_mels  # 80 for base
    n_audio_ctx = model.dims.n_audio_ctx  # 1500 for base

    # Encoder input: mel spectrogram [batch, n_mels, 3000]
    dummy_mel = torch.randn(1, n_mels, 3000)
    encoder_dst = os.path.join(args.output, "encoder_model.onnx")

    torch.onnx.export(
        encoder,
        dummy_mel,
        encoder_dst,
        input_names=["mel"],
        output_names=["audio_features"],
        dynamic_axes={
            "mel": {0: "batch"},
            "audio_features": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"Saved {encoder_dst}")

    # 4. Export decoder to ONNX
    # rusty-whisper expects: tokens, audio_features, pos_emb, k1, v1, k2, v2, ..., k6, v6
    print("Exporting decoder to ONNX...")

    n_text_state = model.dims.n_text_state  # 512 for base
    n_text_layer = model.dims.n_text_layer  # 6 for base

    class DecoderWrapper(torch.nn.Module):
        """Wrapper that takes flat inputs matching rusty-whisper's expected format."""
        def __init__(self, decoder, n_text_layer, n_text_state):
            super().__init__()
            self.decoder = decoder
            self.n_text_layer = n_text_layer
            self.n_text_state = n_text_state

        def forward(self, tokens, audio_features, pos_emb, *kv_args):
            # tokens: [batch, seq_len] int32
            # audio_features: [batch, 1500, n_text_state]
            # pos_emb: [batch, seq_len, n_text_state]
            # kv_args: k1, v1, k2, v2, ..., kN, vN

            # Build KV cache dict
            kv_cache = {}
            for i in range(self.n_text_layer):
                k = kv_args[i * 2]
                v = kv_args[i * 2 + 1]
                kv_cache[i] = (k, v)

            # Embed tokens + positional
            x = self.decoder.token_embedding(tokens.long()) + pos_emb

            # Cross-attention source
            xa = audio_features

            # Run through decoder blocks
            new_kv = {}
            for i, block in enumerate(self.decoder.blocks):
                old_k, old_v = kv_cache.get(i, (None, None))

                # Self-attention with KV cache
                residual = x
                x_norm = block.attn_ln(x)
                q = block.attn.query(x_norm)
                k = block.attn.key(x_norm)
                v = block.attn.value(x_norm)

                # Concat with cached KV
                if old_k is not None and old_k.shape[1] > 0:
                    k = torch.cat([old_k, k], dim=1)
                    v = torch.cat([old_v, v], dim=1)

                new_kv[i] = (k, v)

                # Attention
                n_head = block.attn.n_head
                head_dim = q.shape[-1] // n_head
                scale = head_dim ** -0.5

                q = q.view(*q.shape[:2], n_head, head_dim).permute(0, 2, 1, 3)
                k_h = k.view(*k.shape[:2], n_head, head_dim).permute(0, 2, 1, 3)
                v_h = v.view(*v.shape[:2], n_head, head_dim).permute(0, 2, 1, 3)

                attn_weights = torch.matmul(q, k_h.transpose(-2, -1)) * scale

                # Causal mask
                seq_len_q = q.shape[2]
                seq_len_k = k_h.shape[2]
                causal_mask = torch.triu(
                    torch.ones(seq_len_q, seq_len_k, device=q.device) * float('-inf'),
                    diagonal=seq_len_k - seq_len_q + 1
                )
                attn_weights = attn_weights + causal_mask

                attn_weights = torch.softmax(attn_weights, dim=-1)
                attn_out = torch.matmul(attn_weights, v_h)
                attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(*x.shape[:2], -1)
                attn_out = block.attn.out(attn_out)
                x = residual + attn_out

                # Cross-attention
                residual = x
                x_norm = block.cross_attn_ln(x)
                q_cross = block.cross_attn.query(x_norm)
                k_cross = block.cross_attn.key(xa)
                v_cross = block.cross_attn.value(xa)

                q_cross = q_cross.view(*q_cross.shape[:2], n_head, head_dim).permute(0, 2, 1, 3)
                k_cross = k_cross.view(*k_cross.shape[:2], n_head, head_dim).permute(0, 2, 1, 3)
                v_cross = v_cross.view(*v_cross.shape[:2], n_head, head_dim).permute(0, 2, 1, 3)

                cross_weights = torch.matmul(q_cross, k_cross.transpose(-2, -1)) * scale
                cross_weights = torch.softmax(cross_weights, dim=-1)
                cross_out = torch.matmul(cross_weights, v_cross)
                cross_out = cross_out.permute(0, 2, 1, 3).contiguous().view(*x.shape[:2], -1)
                cross_out = block.cross_attn.out(cross_out)
                x = residual + cross_out

                # FFN
                residual = x
                x = block.mlp_ln(x)
                x = block.mlp[0](x)
                x = torch.nn.functional.gelu(x)
                x = block.mlp[2](x)
                x = residual + x

            x = self.decoder.ln(x)
            logits = x @ self.decoder.token_embedding.weight.T

            # Flatten outputs: logits, k1, v1, k2, v2, ...
            outputs = [logits]
            for i in range(self.n_text_layer):
                outputs.append(new_kv[i][0])
                outputs.append(new_kv[i][1])

            return tuple(outputs)

    wrapper = DecoderWrapper(model.decoder, n_text_layer, n_text_state)
    wrapper.eval()

    # Dummy inputs matching rusty-whisper expectations
    seq_len = 4
    dummy_tokens = torch.zeros(1, seq_len, dtype=torch.int32)
    dummy_audio = torch.randn(1, n_audio_ctx, n_text_state)
    dummy_pos = torch.randn(1, seq_len, n_text_state)
    dummy_kv = [torch.zeros(1, 0, n_text_state) for _ in range(n_text_layer * 2)]

    all_inputs = [dummy_tokens, dummy_audio, dummy_pos] + dummy_kv

    input_names = ["tokens", "audio_features", "positional_embedding"]
    for i in range(n_text_layer):
        input_names.append(f"k{i+1}")
        input_names.append(f"v{i+1}")

    output_names = ["logits"]
    for i in range(n_text_layer):
        output_names.append(f"new_k{i+1}")
        output_names.append(f"new_v{i+1}")

    dynamic_axes = {
        "tokens": {0: "batch", 1: "seq_len"},
        "audio_features": {0: "batch"},
        "positional_embedding": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    }
    for i in range(n_text_layer):
        dynamic_axes[f"k{i+1}"] = {0: "batch", 1: "cache_len"}
        dynamic_axes[f"v{i+1}"] = {0: "batch", 1: "cache_len"}
        dynamic_axes[f"new_k{i+1}"] = {0: "batch", 1: "new_cache_len"}
        dynamic_axes[f"new_v{i+1}"] = {0: "batch", 1: "new_cache_len"}

    decoder_dst = os.path.join(args.output, "decoder_model_merged.onnx")

    torch.onnx.export(
        wrapper,
        tuple(all_inputs),
        decoder_dst,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )
    print(f"Saved {decoder_dst}")

    print(f"\nAll files saved to {args.output}/")
    print("Files:")
    for f in sorted(os.listdir(args.output)):
        size = os.path.getsize(os.path.join(args.output, f))
        print(f"  {f:40s} {size:>10,d} bytes")


if __name__ == "__main__":
    main()
