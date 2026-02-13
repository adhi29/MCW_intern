# SPDX-License-Identifier: Apache-2.0
"""
Export the expert model to ONNX with KV-cache as flat tensor inputs.

This produces a faithful ONNX export that matches the real inference,
where the expert cross-attends to VLM's KV-cache.

Architecture:
- Expert: Qwen3VLTextModel with 36 layers, hidden_size=2048, 16 heads, 8 KV heads, head_dim=128
- VLM KV-cache: 36 layers x (B, 8, seq_len, 128) for both K and V
- Expert receives action embeddings (B, 64, 2048) and attends to VLM's cached K/V
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


class ExpertWithKVCacheModule(nn.Module):
    """
    Expert model wrapper that accepts flat KV-cache tensors for ONNX export.

    Instead of using DynamicCache (which has Python control flow),
    this wrapper manually iterates through expert layers, injecting
    pre-computed K/V states at each layer's attention.
    """

    def __init__(self, model: AlpamayoR1):
        super().__init__()
        self.layers = model.expert.layers
        self.norm = model.expert.norm
        self.rotary_emb = model.expert.rotary_emb
        self.num_layers = model.expert.config.num_hidden_layers
        self.hidden_size = model.expert.config.hidden_size
        self.num_heads = model.expert.config.num_attention_heads
        self.num_kv_heads = model.expert.config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

    def forward(
        self,
        action_embeds: torch.Tensor,   # (B, 64, 2048)
        position_ids: torch.Tensor,    # (3, B, 64) - 3D RoPE
        attention_mask: torch.Tensor,  # (B, 1, 64, seq_len+64) - pre-computed
        # KV-cache as a single stacked tensor for traceability
        past_keys: torch.Tensor,       # (num_layers, B, num_kv_heads, seq_len, head_dim)
        past_values: torch.Tensor,     # (num_layers, B, num_kv_heads, seq_len, head_dim)
    ) -> torch.Tensor:
        """
        Run expert transformer with pre-populated KV-cache from VLM.

        Returns:
            hidden_states for the action tokens: (B, 64, 2048)
        """
        # Build DynamicCache from flat tensors
        past_key_values = DynamicCache()
        for layer_idx in range(self.num_layers):
            # DynamicCache.update concatenates new K/V with existing.
            # We just set the initial cache by calling update with the full tensors.
            past_key_values.update(
                past_keys[layer_idx],
                past_values[layer_idx],
                layer_idx,
            )

        cache_seq_len = past_keys.shape[3]
        cache_position = torch.arange(
            cache_seq_len, cache_seq_len + action_embeds.shape[1],
            device=action_embeds.device,
        )

        hidden_states = action_embeds

        # Compute position embeddings (3D RoPE for Qwen3-VL)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Run through decoder layers
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids[0],  # text_position_ids
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=True,
                is_causal=False,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class FullDiffusionStepWithKVCache(nn.Module):
    """
    Full diffusion step: action_in_proj -> expert (with KV-cache) -> action_out_proj.

    This bundles all three components for a single ONNX model.
    """

    def __init__(self, model: AlpamayoR1):
        super().__init__()
        self.action_in_proj = model.action_in_proj
        self.expert_module = ExpertWithKVCacheModule(model)
        self.action_out_proj = model.action_out_proj
        self.num_layers = model.expert.config.num_hidden_layers

    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, 64, 2)
        timesteps: torch.Tensor,       # (B, 1)
        position_ids: torch.Tensor,    # (3, B, 64)
        attention_mask: torch.Tensor,  # (B, 1, 64, seq_len+64)
        past_keys: torch.Tensor,       # (num_layers, B, num_kv_heads, seq_len, head_dim)
        past_values: torch.Tensor,     # (num_layers, B, num_kv_heads, seq_len, head_dim)
    ) -> torch.Tensor:
        """
        One diffusion denoising step.

        Returns:
            predicted velocity: (B, 64, 2)
        """
        # Project noisy actions to embeddings
        action_embeds = self.action_in_proj(noisy_actions, timesteps)  # (B, 64, 2048)

        # Run expert with KV-cache
        expert_out = self.expert_module(
            action_embeds, position_ids, attention_mask, past_keys, past_values
        )

        # Extract action tokens and project to action space
        action_hidden = expert_out[:, -64:, :]
        pred_velocity = self.action_out_proj(action_hidden)  # (B, 64, 2)

        return pred_velocity


def export_expert_with_kvcache(model: AlpamayoR1, output_dir: str, opset_version: int = 17):
    """Export expert-only module with KV-cache inputs."""
    print("Exporting expert model with KV-cache inputs...")

    expert_module = ExpertWithKVCacheModule(model)
    expert_module = expert_module.float()
    expert_module.eval()

    B = 1
    action_len = 64
    context_len = 256  # Typical VLM context length
    hidden_size = model.expert.config.hidden_size
    num_kv_heads = model.expert.config.num_key_value_heads
    head_dim = hidden_size // model.expert.config.num_attention_heads
    num_layers = model.expert.config.num_hidden_layers

    # Dummy inputs
    dummy_action_embeds = torch.randn(B, action_len, hidden_size)
    dummy_position_ids = torch.arange(action_len).unsqueeze(0).unsqueeze(0).expand(3, B, -1)
    total_len = context_len + action_len
    dummy_mask = torch.zeros(B, 1, action_len, total_len)
    dummy_past_keys = torch.randn(num_layers, B, num_kv_heads, context_len, head_dim)
    dummy_past_values = torch.randn(num_layers, B, num_kv_heads, context_len, head_dim)

    output_path = os.path.join(output_dir, "expert_kvcache.onnx")

    with torch.no_grad():
        try:
            torch.onnx.export(
                expert_module,
                (dummy_action_embeds, dummy_position_ids, dummy_mask,
                 dummy_past_keys, dummy_past_values),
                output_path,
                input_names=[
                    "action_embeds", "position_ids", "attention_mask",
                    "past_keys", "past_values",
                ],
                output_names=["hidden_states"],
                dynamic_axes={
                    "action_embeds": {0: "batch_size"},
                    "position_ids": {1: "batch_size"},
                    "attention_mask": {0: "batch_size", 3: "total_length"},
                    "past_keys": {1: "batch_size", 3: "context_length"},
                    "past_values": {1: "batch_size", 3: "context_length"},
                    "hidden_states": {0: "batch_size"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
            print(f"Exported expert with KV-cache to: {output_path}")
        except Exception as e:
            print(f"Export failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    return output_path


def export_full_diffusion_step_kvcache(model: AlpamayoR1, output_dir: str,
                                        opset_version: int = 17):
    """Export full diffusion step (action_in_proj + expert + action_out_proj) with KV-cache."""
    print("Exporting full diffusion step with KV-cache inputs...")

    diff_module = FullDiffusionStepWithKVCache(model)
    diff_module = diff_module.float()
    diff_module.eval()

    B = 1
    action_len = 64
    context_len = 256
    hidden_size = model.expert.config.hidden_size
    num_kv_heads = model.expert.config.num_key_value_heads
    head_dim = hidden_size // model.expert.config.num_attention_heads
    num_layers = model.expert.config.num_hidden_layers

    # Dummy inputs
    dummy_actions = torch.randn(B, action_len, 2)
    dummy_timesteps = torch.rand(B, 1)
    dummy_position_ids = torch.arange(action_len).unsqueeze(0).unsqueeze(0).expand(3, B, -1)
    total_len = context_len + action_len
    dummy_mask = torch.zeros(B, 1, action_len, total_len)
    dummy_past_keys = torch.randn(num_layers, B, num_kv_heads, context_len, head_dim)
    dummy_past_values = torch.randn(num_layers, B, num_kv_heads, context_len, head_dim)

    output_path = os.path.join(output_dir, "diffusion_step_kvcache.onnx")

    with torch.no_grad():
        try:
            torch.onnx.export(
                diff_module,
                (dummy_actions, dummy_timesteps, dummy_position_ids, dummy_mask,
                 dummy_past_keys, dummy_past_values),
                output_path,
                input_names=[
                    "noisy_actions", "timesteps", "position_ids", "attention_mask",
                    "past_keys", "past_values",
                ],
                output_names=["predicted_velocity"],
                dynamic_axes={
                    "noisy_actions": {0: "batch_size"},
                    "timesteps": {0: "batch_size"},
                    "position_ids": {1: "batch_size"},
                    "attention_mask": {0: "batch_size", 3: "total_length"},
                    "past_keys": {1: "batch_size", 3: "context_length"},
                    "past_values": {1: "batch_size", 3: "context_length"},
                    "predicted_velocity": {0: "batch_size"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
            print(f"Exported full diffusion step with KV-cache to: {output_path}")
        except Exception as e:
            print(f"Export failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export Alpamayo R1 expert with KV-cache to ONNX"
    )
    parser.add_argument(
        "--model-path", type=str, default="nvidia/Alpamayo-R1-10B",
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./onnx_models",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--opset-version", type=int, default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--export-mode", type=str, default="both",
        choices=["expert", "full", "both"],
        help="Export expert only, full diffusion step, or both",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    model = AlpamayoR1.from_pretrained(
        args.model_path, dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    print(f"Expert: hidden_size={model.expert.config.hidden_size}, "
          f"layers={model.expert.config.num_hidden_layers}, "
          f"heads={model.expert.config.num_attention_heads}, "
          f"kv_heads={model.expert.config.num_key_value_heads}")

    if args.export_mode in ("expert", "both"):
        export_expert_with_kvcache(model, args.output_dir, args.opset_version)

    if args.export_mode in ("full", "both"):
        export_full_diffusion_step_kvcache(model, args.output_dir, args.opset_version)

    print("\nExport completed!")


if __name__ == "__main__":
    main()
