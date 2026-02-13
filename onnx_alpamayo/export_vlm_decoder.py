# SPDX-License-Identifier: Apache-2.0
"""
Export the Qwen3-VL text decoder to ONNX (prefill + decode).

Two models are exported:
  1. vlm_decoder_prefill.onnx — processes the full input sequence (text+visual),
     returns hidden states and KV-cache. Handles DeepStack injection.
  2. vlm_decoder_decode.onnx — single-token autoregressive decode step,
     takes and returns flat KV-cache tensors.

DeepStack injection uses pre-expanded tensors (B, seq_len, hidden_size) with
zeros at non-visual positions, avoiding boolean indexing which ONNX can't
handle with dynamic shapes.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class DecoderPrefillForONNX(nn.Module):
    """
    Wraps the Qwen3-VL text decoder for prefill ONNX export.

    DeepStack visual features are passed as full-size tensors (B, seq_len, 4096)
    with zeros at non-visual positions. This avoids boolean indexing which
    ONNX can't handle dynamically.
    """

    def __init__(self, text_model):
        super().__init__()
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.config = text_model.config
        self.num_layers = len(self.layers)

        # Force eager attention for all layers
        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def forward(
        self,
        inputs_embeds: torch.Tensor,           # (B, seq_len, 4096)
        position_ids: torch.Tensor,             # (3, B, seq_len)
        attention_mask: torch.Tensor,           # (B, seq_len) — 1/0 padding mask
        deepstack_full_0: torch.Tensor,         # (B, seq_len, 4096) — pre-expanded, zeros at non-visual
        deepstack_full_1: torch.Tensor,         # (B, seq_len, 4096)
        deepstack_full_2: torch.Tensor,         # (B, seq_len, 4096)
    ):
        """
        Prefill forward pass through all decoder layers.

        DeepStack tensors are full-size (B, seq_len, 4096) with zeros at
        non-visual positions. They are simply added to hidden_states at
        the corresponding layer.

        Returns:
            last_hidden_state: (B, seq_len, 4096)
            past_keys: (num_layers, B, num_kv_heads, seq_len, head_dim)
            past_values: (num_layers, B, num_kv_heads, seq_len, head_dim)
        """
        B, seq_len, _ = inputs_embeds.shape
        deepstack_list = [deepstack_full_0, deepstack_full_1, deepstack_full_2]

        # Compute cache_position
        cache_position = torch.arange(seq_len, device=inputs_embeds.device)

        # Extract text position IDs
        text_position_ids = position_ids[0]  # (B, seq_len)

        # Compute position embeddings (3D MRoPE)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Build causal attention mask: (B, 1, seq_len, seq_len)
        causal_mask = torch.full(
            (seq_len, seq_len), torch.finfo(inputs_embeds.dtype).min,
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

        # Apply padding mask
        padding_mask = attention_mask[:, None, None, :]  # (B, 1, 1, seq_len)
        padding_mask = (1.0 - padding_mask) * torch.finfo(inputs_embeds.dtype).min
        attn_mask = causal_mask + padding_mask

        # Create DynamicCache to collect KV states
        past_key_values = DynamicCache()

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=True,
            )

            # DeepStack injection: simply add pre-expanded tensor
            # Non-visual positions have zeros, so only visual positions are affected
            if layer_idx < len(deepstack_list):
                hidden_states = hidden_states + deepstack_list[layer_idx]

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Extract KV-cache as stacked tensors
        keys_list = []
        values_list = []
        for layer_idx in range(self.num_layers):
            k, v = past_key_values[layer_idx]
            keys_list.append(k)
            values_list.append(v)

        past_keys = torch.stack(keys_list, dim=0)     # (L, B, H, S, D)
        past_values = torch.stack(values_list, dim=0)  # (L, B, H, S, D)

        return hidden_states, past_keys, past_values


class DecoderDecodeForONNX(nn.Module):
    """
    Wraps the Qwen3-VL text decoder for single-step decode ONNX export.

    Takes a single token embedding and existing KV-cache, returns updated
    hidden state and KV-cache. No DeepStack injection during decode.
    """

    def __init__(self, text_model):
        super().__init__()
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.config = text_model.config
        self.num_layers = len(self.layers)

        # Force eager attention for all layers
        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def forward(
        self,
        input_embeds: torch.Tensor,        # (B, 1, 4096)
        position_ids: torch.Tensor,         # (3, B, 1)
        attention_mask: torch.Tensor,       # (B, 1, 1, past_seq_len+1) — full 4D mask
        past_keys: torch.Tensor,            # (num_layers, B, num_kv_heads, past_seq_len, head_dim)
        past_values: torch.Tensor,          # (num_layers, B, num_kv_heads, past_seq_len, head_dim)
    ):
        """
        Single-step decode through all decoder layers.

        Returns:
            hidden_state: (B, 1, 4096)
            new_past_keys: (num_layers, B, num_kv_heads, past_seq_len+1, head_dim)
            new_past_values: (num_layers, B, num_kv_heads, past_seq_len+1, head_dim)
        """
        past_seq_len = past_keys.shape[3]
        cache_position = torch.tensor(
            [past_seq_len], device=input_embeds.device, dtype=torch.long
        )

        # Text position IDs
        text_position_ids = position_ids[0]  # (B, 1)

        # Position embeddings
        hidden_states = input_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Build DynamicCache from flat tensors
        past_key_values = DynamicCache()
        for layer_idx in range(self.num_layers):
            past_key_values.update(
                past_keys[layer_idx],
                past_values[layer_idx],
                layer_idx,
            )

        # Run through layers
        for layer_idx in range(self.num_layers):
            hidden_states = self.layers[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=True,
            )

        hidden_states = self.norm(hidden_states)

        # Extract updated KV-cache
        new_keys_list = []
        new_values_list = []
        for layer_idx in range(self.num_layers):
            k, v = past_key_values[layer_idx]
            new_keys_list.append(k)
            new_values_list.append(v)

        new_past_keys = torch.stack(new_keys_list, dim=0)
        new_past_values = torch.stack(new_values_list, dim=0)

        return hidden_states, new_past_keys, new_past_values


def export_decoder_prefill(model: AlpamayoR1, output_dir: str, opset_version: int = 17):
    """Export the text decoder prefill model to ONNX."""
    print("Exporting VLM decoder prefill...")

    text_model = model.vlm.model.language_model
    wrapper = DecoderPrefillForONNX(text_model)
    wrapper = wrapper.float()
    wrapper.eval()

    # Dummy inputs
    B = 1
    seq_len = 32
    hidden_size = model.vlm.config.text_config.hidden_size  # 4096

    dummy_embeds = torch.randn(B, seq_len, hidden_size)
    dummy_position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, B, -1).long()
    dummy_attention_mask = torch.ones(B, seq_len, dtype=torch.float32)
    # DeepStack: pre-expanded full-size tensors (zeros at non-visual positions)
    dummy_ds0 = torch.randn(B, seq_len, hidden_size) * 0.01  # small values
    dummy_ds1 = torch.randn(B, seq_len, hidden_size) * 0.01
    dummy_ds2 = torch.randn(B, seq_len, hidden_size) * 0.01

    output_path = os.path.join(output_dir, "vlm_decoder_prefill.onnx")

    with torch.no_grad():
        print("  Testing forward pass...")
        try:
            out = wrapper(
                dummy_embeds, dummy_position_ids, dummy_attention_mask,
                dummy_ds0, dummy_ds1, dummy_ds2
            )
            print(f"  hidden_states shape: {out[0].shape}")
            print(f"  past_keys shape: {out[1].shape}")
            print(f"  past_values shape: {out[2].shape}")
        except Exception as e:
            print(f"  Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        print("  Exporting to ONNX...")
        try:
            torch.onnx.export(
                wrapper,
                (
                    dummy_embeds, dummy_position_ids, dummy_attention_mask,
                    dummy_ds0, dummy_ds1, dummy_ds2
                ),
                output_path,
                input_names=[
                    "inputs_embeds", "position_ids", "attention_mask",
                    "deepstack_full_0", "deepstack_full_1", "deepstack_full_2",
                ],
                output_names=["last_hidden_state", "past_keys", "past_values"],
                dynamic_axes={
                    "inputs_embeds": {0: "batch", 1: "seq_len"},
                    "position_ids": {1: "batch", 2: "seq_len"},
                    "attention_mask": {0: "batch", 1: "seq_len"},
                    "deepstack_full_0": {0: "batch", 1: "seq_len"},
                    "deepstack_full_1": {0: "batch", 1: "seq_len"},
                    "deepstack_full_2": {0: "batch", 1: "seq_len"},
                    "last_hidden_state": {0: "batch", 1: "seq_len"},
                    "past_keys": {1: "batch", 3: "seq_len"},
                    "past_values": {1: "batch", 3: "seq_len"},
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


def export_decoder_decode(model: AlpamayoR1, output_dir: str, opset_version: int = 17):
    """Export the text decoder single-step decode model to ONNX."""
    print("Exporting VLM decoder decode...")

    text_model = model.vlm.model.language_model
    wrapper = DecoderDecodeForONNX(text_model)
    wrapper = wrapper.float()
    wrapper.eval()

    # Dummy inputs
    B = 1
    past_seq_len = 32
    hidden_size = model.vlm.config.text_config.hidden_size  # 4096
    num_layers = model.vlm.config.text_config.num_hidden_layers  # 36
    num_kv_heads = model.vlm.config.text_config.num_key_value_heads  # 8
    head_dim = hidden_size // model.vlm.config.text_config.num_attention_heads  # 128

    dummy_embeds = torch.randn(B, 1, hidden_size)
    dummy_position_ids = torch.tensor([[[past_seq_len]]], dtype=torch.long).expand(3, B, 1)
    dummy_attention_mask = torch.zeros(B, 1, 1, past_seq_len + 1, dtype=torch.float32)
    dummy_past_keys = torch.randn(num_layers, B, num_kv_heads, past_seq_len, head_dim)
    dummy_past_values = torch.randn(num_layers, B, num_kv_heads, past_seq_len, head_dim)

    output_path = os.path.join(output_dir, "vlm_decoder_decode.onnx")

    with torch.no_grad():
        print("  Testing forward pass...")
        try:
            out = wrapper(
                dummy_embeds, dummy_position_ids, dummy_attention_mask,
                dummy_past_keys, dummy_past_values
            )
            print(f"  hidden_state shape: {out[0].shape}")
            print(f"  new_past_keys shape: {out[1].shape}")
            print(f"  new_past_values shape: {out[2].shape}")
        except Exception as e:
            print(f"  Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        print("  Exporting to ONNX...")
        try:
            torch.onnx.export(
                wrapper,
                (
                    dummy_embeds, dummy_position_ids, dummy_attention_mask,
                    dummy_past_keys, dummy_past_values
                ),
                output_path,
                input_names=[
                    "input_embeds", "position_ids", "attention_mask",
                    "past_keys", "past_values",
                ],
                output_names=["hidden_state", "new_past_keys", "new_past_values"],
                dynamic_axes={
                    "input_embeds": {0: "batch"},
                    "position_ids": {1: "batch"},
                    "attention_mask": {0: "batch", 3: "past_seq_plus_one"},
                    "past_keys": {1: "batch", 3: "past_seq_len"},
                    "past_values": {1: "batch", 3: "past_seq_len"},
                    "hidden_state": {0: "batch"},
                    "new_past_keys": {1: "batch", 3: "new_seq_len"},
                    "new_past_values": {1: "batch", 3: "new_seq_len"},
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
    parser = argparse.ArgumentParser(description="Export Qwen3-VL text decoder to ONNX")
    parser.add_argument("--model-path", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--output-dir", type=str, default="./onnx_models")
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--prefill-only", action="store_true", help="Only export prefill model")
    parser.add_argument("--decode-only", action="store_true", help="Only export decode model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    model = AlpamayoR1.from_pretrained(
        args.model_path, dtype=torch.float32, device_map="cpu"
    )
    model = model.float()  # Ensure all weights are float32
    model.eval()

    if not args.decode_only:
        export_decoder_prefill(model, args.output_dir, args.opset_version)

    if not args.prefill_only:
        export_decoder_decode(model, args.output_dir, args.opset_version)

    print("\nDone!")


if __name__ == "__main__":
    main()
