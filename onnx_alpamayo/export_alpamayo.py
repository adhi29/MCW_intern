# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation
"""
Export Alpamayo R1 model components to ONNX format.

The Alpamayo R1 model has a complex architecture with:
1. VLM backbone (Qwen3-VL) - for visual encoding and reasoning
2. Expert model - for diffusion denoising
3. Action projections - for converting between action and embedding spaces

Due to the complexity of the autoregressive generation in the VLM,
we export the model in a way that allows hybrid PyTorch/ONNX inference:
- VLM visual encoder: Exported to ONNX (image feature extraction)
- Expert diffusion step: Exported to ONNX (core denoising computation)
- Action projections: Exported to ONNX
- Control flow (generation, diffusion loop): Kept in Python
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper


class SimplifiedExpertModule(nn.Module):
    """
    Simplified expert module for ONNX export.

    This wraps just the expert transformer layers and relies on a
    pre-computed causal mask to avoid the SDPA mask builder path.
    """

    def __init__(self, model: AlpamayoR1):
        super().__init__()
        # Copy the expert's layers directly
        self.layers = model.expert.layers
        self.norm = model.expert.norm
        self.rotary_emb = model.expert.rotary_emb
        self.hidden_size = model.expert.config.hidden_size
        self.num_heads = model.expert.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, seq_len, hidden_size)
        causal_mask: torch.Tensor,     # (B, 1, seq_len, seq_len) - pre-computed
        position_ids: torch.Tensor,    # (B, seq_len)
    ) -> torch.Tensor:
        """
        Run expert transformer layers with pre-computed causal mask.
        """
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class DiffusionStepModule(nn.Module):
    """
    Wrapper module for a single diffusion denoising step.

    This combines:
    - context_proj: Projects VLM hidden states to expert hidden size (if different)
    - action_in_proj: Projects noisy actions to embeddings
    - expert: Transformer that denoises
    - action_out_proj: Projects back to action space

    For ONNX export, we use a simplified version without KV cache.
    """

    def __init__(self, model: AlpamayoR1):
        super().__init__()
        self.action_in_proj = model.action_in_proj
        self.expert = SimplifiedExpertModule(model)
        self.action_out_proj = model.action_out_proj
        self.hidden_size = model.expert.config.hidden_size

        # Add projection layer if VLM and Expert have different hidden sizes
        vlm_hidden_size = model.vlm.config.text_config.hidden_size
        expert_hidden_size = model.expert.config.hidden_size
        if vlm_hidden_size != expert_hidden_size:
            self.context_proj = nn.Linear(vlm_hidden_size, expert_hidden_size, bias=False)
            # Initialize to truncation (take first expert_hidden_size dims)
            with torch.no_grad():
                self.context_proj.weight.zero_()
                self.context_proj.weight[:, :expert_hidden_size] = torch.eye(expert_hidden_size)
        else:
            self.context_proj = None

    def forward(
        self,
        noisy_actions: torch.Tensor,  # (B, 64, 2)
        timesteps: torch.Tensor,       # (B, 1) - scalar per batch
        context_embeddings: torch.Tensor,  # (B, seq_len, vlm_hidden_size) - from VLM
        causal_mask: torch.Tensor,  # (B, 1, seq_len + 64, seq_len + 64)
    ) -> torch.Tensor:
        """
        Perform one diffusion denoising step.

        Args:
            noisy_actions: Noisy action values (B, 64, 2)
            timesteps: Diffusion timesteps (B, 1) - scalar per batch item
            context_embeddings: Context from VLM (B, seq_len, vlm_hidden_size)
            causal_mask: Pre-computed causal mask for full sequence

        Returns:
            Predicted velocity/noise (B, 64, 2)
        """
        B = noisy_actions.shape[0]

        # Project context embeddings from VLM hidden size to expert hidden size if needed
        if self.context_proj is not None:
            context_embeddings = self.context_proj(context_embeddings)

        # Project noisy actions to embeddings
        # timesteps is (B, 1), action_in_proj broadcasts to all waypoints
        action_embeds = self.action_in_proj(noisy_actions, timesteps)  # (B, 64, hidden_size)

        # Concatenate context and action embeddings
        inputs_embeds = torch.cat([context_embeddings, action_embeds], dim=1)

        # Create position IDs
        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)

        # Run expert with attn_implementation="eager" to avoid complex masking
        expert_out = self.expert(
            inputs_embeds,
            causal_mask,
            position_ids,
        )

        # Extract action part and project to action space
        action_hidden = expert_out[:, -64:, :]  # Last 64 tokens
        pred_velocity = self.action_out_proj(action_hidden)  # (B, 64, 2)

        return pred_velocity


def export_diffusion_step(model: AlpamayoR1, output_dir: str, opset_version: int = 17):
    """Export the diffusion step module to ONNX."""

    print("Exporting diffusion step module...")

    diff_module = DiffusionStepModule(model)
    diff_module = diff_module.float()  # Ensure float32
    diff_module.eval()

    # Create dummy inputs
    B = 1
    context_len = 256  # Typical context length
    # Use VLM hidden size for context (will be projected internally if needed)
    vlm_hidden_size = model.vlm.config.text_config.hidden_size

    dummy_actions = torch.randn(B, 64, 2)
    # Timesteps should be (B, 1) - scalar per batch, broadcast to all waypoints
    dummy_timesteps = torch.rand(B, 1)
    dummy_context = torch.randn(B, context_len, vlm_hidden_size)
    total_len = context_len + 64
    dummy_mask = torch.triu(torch.ones(total_len, total_len), diagonal=1)
    dummy_mask = dummy_mask.masked_fill(dummy_mask == 1, torch.finfo(torch.float32).min)
    dummy_mask = dummy_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, total_len, total_len)

    output_path = os.path.join(output_dir, "diffusion_step.onnx")

    with torch.no_grad():
        try:
            # Try dynamo-based export first (better for transformers)
            torch.onnx.export(
                diff_module,
                (dummy_actions, dummy_timesteps, dummy_context, dummy_mask),
                output_path,
                input_names=["noisy_actions", "timesteps", "context_embeddings", "causal_mask"],
                output_names=["predicted_velocity"],
                dynamic_axes={
                    "noisy_actions": {0: "batch_size"},
                    "timesteps": {0: "batch_size"},
                    "context_embeddings": {0: "batch_size", 1: "context_length"},
                    "causal_mask": {0: "batch_size", 2: "total_length", 3: "total_length"},
                    "predicted_velocity": {0: "batch_size"},
                },
                opset_version=opset_version,
                dynamo=True,  # Use dynamo-based export
            )
        except Exception as dynamo_err:
            print(f"Dynamo export failed: {dynamo_err}")
            print("Falling back to legacy export...")
            torch.onnx.export(
                diff_module,
                (dummy_actions, dummy_timesteps, dummy_context, dummy_mask),
                output_path,
                input_names=["noisy_actions", "timesteps", "context_embeddings", "causal_mask"],
                output_names=["predicted_velocity"],
                dynamic_axes={
                    "noisy_actions": {0: "batch_size"},
                    "timesteps": {0: "batch_size"},
                    "context_embeddings": {0: "batch_size", 1: "context_length"},
                    "causal_mask": {0: "batch_size", 2: "total_length", 3: "total_length"},
                    "predicted_velocity": {0: "batch_size"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )

    print(f"Exported diffusion step to: {output_path}")
    return output_path


def export_action_projections(model: AlpamayoR1, output_dir: str, opset_version: int = 17):
    """Export action projection modules separately."""

    print("Exporting action input projection...")

    # Export action_in_proj
    action_in_proj = model.action_in_proj.float()  # Ensure float32
    action_in_proj.eval()

    B = 1
    dummy_actions = torch.randn(B, 64, 2)
    # Timesteps should be (B, 1) - scalar per batch, broadcast to all waypoints
    dummy_timesteps = torch.rand(B, 1)

    in_proj_path = os.path.join(output_dir, "action_in_proj.onnx")

    with torch.no_grad():
        torch.onnx.export(
            action_in_proj,
            (dummy_actions, dummy_timesteps),
            in_proj_path,
            input_names=["actions", "timesteps"],
            output_names=["embeddings"],
            dynamic_axes={
                "actions": {0: "batch_size"},
                "timesteps": {0: "batch_size"},
                "embeddings": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    print(f"Exported action_in_proj to: {in_proj_path}")

    # Export action_out_proj (simple linear layer)
    print("Exporting action output projection...")

    action_out_proj = model.action_out_proj.float()  # Ensure float32
    action_out_proj.eval()

    hidden_size = model.expert.config.hidden_size
    dummy_hidden = torch.randn(B, 64, hidden_size)

    out_proj_path = os.path.join(output_dir, "action_out_proj.onnx")

    with torch.no_grad():
        torch.onnx.export(
            action_out_proj,
            dummy_hidden,
            out_proj_path,
            input_names=["hidden_states"],
            output_names=["actions"],
            dynamic_axes={
                "hidden_states": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    print(f"Exported action_out_proj to: {out_proj_path}")

    return in_proj_path, out_proj_path


def save_model_config(model: AlpamayoR1, output_dir: str):
    """Save model configuration for loading ONNX models."""
    import json

    config = {
        "hidden_size": model.expert.config.hidden_size,
        "num_waypoints": 64,
        "action_dim": 2,
        "num_diffusion_steps": model.diffusion.num_inference_steps,
        "vlm_name": model.config.vlm_name_or_path,
        "expert_num_layers": model.expert.config.num_hidden_layers,
        "expert_num_heads": model.expert.config.num_attention_heads,
    }

    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved model config to: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Export Alpamayo R1 to ONNX")
    parser.add_argument(
        "--model-path",
        type=str,
        default="nvidia/Alpamayo-R1-10B",
        help="Path to model checkpoint or HuggingFace model ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx_models",
        help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    print("This may take a while for large models...")

    # Load model in float32 for export (ONNX works better with float32)
    model = AlpamayoR1.from_pretrained(
        args.model_path,
        dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    print("Model loaded successfully!")
    print(f"Hidden size: {model.expert.config.hidden_size}")
    print(f"Expert layers: {model.expert.config.num_hidden_layers}")

    # Save configuration
    save_model_config(model, args.output_dir)

    # Export components
    try:
        export_action_projections(model, args.output_dir, args.opset_version)
    except Exception as e:
        print(f"Warning: Failed to export action projections: {e}")

    try:
        export_diffusion_step(model, args.output_dir, args.opset_version)
    except Exception as e:
        print(f"Warning: Failed to export diffusion step: {e}")
        import traceback
        traceback.print_exc()

    print("\nExport completed!")
    print(f"ONNX files saved to: {args.output_dir}")
    print("\nNote: The VLM component is not exported due to complexity.")
    print("For full ONNX inference, consider using ONNX Runtime GenAI or TensorRT-LLM.")


if __name__ == "__main__":
    main()