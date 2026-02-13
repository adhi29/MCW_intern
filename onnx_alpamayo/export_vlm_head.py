# SPDX-License-Identifier: Apache-2.0
"""
Export the Qwen3-VL LM head and embed_tokens to ONNX.

Two simple models:
  1. vlm_embed_tokens.onnx — nn.Embedding lookup: (B, seq_len) int64 → (B, seq_len, 4096)
  2. vlm_lm_head.onnx — Linear projection: (B, 1, 4096) → (B, 1, vocab_size)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class EmbedTokensForONNX(nn.Module):
    """Simple wrapper around nn.Embedding for ONNX export."""

    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len) int64
        Returns:
            embeddings: (B, seq_len, hidden_size)
        """
        return self.embed_tokens(input_ids)


class LMHeadForONNX(nn.Module):
    """Simple wrapper around the LM head Linear layer for ONNX export."""

    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, seq_len, hidden_size)
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        return self.lm_head(hidden_states)


def export_embed_tokens(model: AlpamayoR1, output_dir: str, opset_version: int = 17):
    """Export embed_tokens to ONNX."""
    print("Exporting VLM embed_tokens...")

    embed_tokens = model.vlm.model.language_model.embed_tokens
    wrapper = EmbedTokensForONNX(embed_tokens)
    wrapper = wrapper.float()
    wrapper.eval()

    B = 1
    seq_len = 16
    dummy_input_ids = torch.randint(0, 1000, (B, seq_len), dtype=torch.long)

    output_path = os.path.join(output_dir, "vlm_embed_tokens.onnx")

    with torch.no_grad():
        print("  Testing forward pass...")
        out = wrapper(dummy_input_ids)
        print(f"  embeddings shape: {out.shape}")

        print("  Exporting to ONNX...")
        try:
            torch.onnx.export(
                wrapper,
                (dummy_input_ids,),
                output_path,
                input_names=["input_ids"],
                output_names=["embeddings"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq_len"},
                    "embeddings": {0: "batch", 1: "seq_len"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
            print(f"  Exported to: {output_path}")
        except Exception as e:
            print(f"  Export failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    return output_path


def export_lm_head(model: AlpamayoR1, output_dir: str, opset_version: int = 17):
    """Export LM head to ONNX."""
    print("Exporting VLM LM head...")

    lm_head = model.vlm.lm_head
    wrapper = LMHeadForONNX(lm_head)
    wrapper = wrapper.float()
    wrapper.eval()

    B = 1
    hidden_size = model.vlm.config.text_config.hidden_size  # 4096

    # For decode: single token hidden state
    dummy_hidden = torch.randn(B, 1, hidden_size)

    output_path = os.path.join(output_dir, "vlm_lm_head.onnx")

    with torch.no_grad():
        print("  Testing forward pass...")
        out = wrapper(dummy_hidden)
        print(f"  logits shape: {out.shape}")

        print("  Exporting to ONNX...")
        try:
            torch.onnx.export(
                wrapper,
                (dummy_hidden,),
                output_path,
                input_names=["hidden_states"],
                output_names=["logits"],
                dynamic_axes={
                    "hidden_states": {0: "batch", 1: "seq_len"},
                    "logits": {0: "batch", 1: "seq_len"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
            print(f"  Exported to: {output_path}")
        except Exception as e:
            print(f"  Export failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-VL LM head and embed_tokens to ONNX")
    parser.add_argument("--model-path", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--output-dir", type=str, default="./onnx_models")
    parser.add_argument("--opset-version", type=int, default=17)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    model = AlpamayoR1.from_pretrained(
        args.model_path, dtype=torch.float32, device_map="cpu"
    )
    model = model.float()  # Ensure all weights are float32
    model.eval()

    export_embed_tokens(model, args.output_dir, args.opset_version)
    export_lm_head(model, args.output_dir, args.opset_version)

    print("\nDone!")


if __name__ == "__main__":
    main()
