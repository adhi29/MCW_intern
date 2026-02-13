# SPDX-License-Identifier: Apache-2.0
"""
Export all 6 Alpamayo R1 ONNX models with FULLY STATIC shapes for Qualcomm NPU.

No dynamic axes — all tensor dimensions are fixed at compile time.
Exports in fp32 first; use convert_onnx_precision.py for fp16 conversion.

Fixed shapes (verified across 1181 clips):
  batch = 1
  num_images = 16 (4 cameras × 4 frames)
  total_patches = 11520
  merged_seq = 2880
  prefill_seq_len = 3006
  max_generation = 256
  max_total_seq = 3262
  action_len = 64

Usage:
    python export_static_fp16.py
    python export_static_fp16.py --output-dir ./onnx_models_static_fp16
    python export_static_fp16.py --export-only embed_tokens lm_head
"""

import os
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

# Import wrapper classes from existing export scripts
from export_vlm_head import EmbedTokensForONNX, LMHeadForONNX
from export_vlm_vision import VisionEncoderForONNX
from export_vlm_decoder import DecoderPrefillForONNX, DecoderDecodeForONNX
from export_expert_kvcache import FullDiffusionStepWithKVCache

# ============================================================
# Static shape constants
# ============================================================
BATCH = 1
NUM_IMAGES = 16
TOTAL_PATCHES = 11520
PATCH_DIM = 1536  # 3 * 2 * 16 * 16
MERGED_SEQ = 2880
PREFILL_SEQ_LEN = 3006
MAX_GENERATION = 256
MAX_TOTAL_SEQ = PREFILL_SEQ_LEN + MAX_GENERATION  # 3262
ACTION_LEN = 64
HIDDEN_SIZE = 4096
EXPERT_HIDDEN_SIZE = 2048
NUM_LAYERS = 36
NUM_KV_HEADS = 8
HEAD_DIM = 128
OPSET_VERSION = 17


def export_embed_tokens_static(model, output_dir):
    """Export vlm_embed_tokens with static shape (1, 3006)."""
    print("\n" + "=" * 60)
    print("Exporting vlm_embed_tokens.onnx (static)")
    print("=" * 60)

    wrapper = EmbedTokensForONNX(model.vlm.model.language_model.embed_tokens)
    wrapper = wrapper.float().eval()

    dummy_input_ids = torch.randint(0, 1000, (BATCH, PREFILL_SEQ_LEN), dtype=torch.long)
    output_path = os.path.join(output_dir, "common", "vlm_embed_tokens.onnx")

    with torch.no_grad():
        out = wrapper(dummy_input_ids)
        print(f"  Input:  input_ids {dummy_input_ids.shape}")
        print(f"  Output: embeddings {out.shape}")

        torch.onnx.export(
            wrapper,
            (dummy_input_ids,),
            output_path,
            input_names=["input_ids"],
            output_names=["embeddings"],
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
    print(f"  Saved: {output_path}")


def export_lm_head_static(model, output_dir):
    """Export vlm_lm_head with static shape (1, 1, 4096)."""
    print("\n" + "=" * 60)
    print("Exporting vlm_lm_head.onnx (static)")
    print("=" * 60)

    wrapper = LMHeadForONNX(model.vlm.lm_head)
    wrapper = wrapper.float().eval()

    dummy_hidden = torch.randn(BATCH, 1, HIDDEN_SIZE)
    output_path = os.path.join(output_dir, "common", "vlm_lm_head.onnx")

    with torch.no_grad():
        out = wrapper(dummy_hidden)
        print(f"  Input:  hidden_states {dummy_hidden.shape}")
        print(f"  Output: logits {out.shape}")

        torch.onnx.export(
            wrapper,
            (dummy_hidden,),
            output_path,
            input_names=["hidden_states"],
            output_names=["logits"],
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
    print(f"  Saved: {output_path}")


def export_vision_encoder_static(model, output_dir):
    """Export vlm_vision_encoder with static shape (11520, 1536)."""
    print("\n" + "=" * 60)
    print("Exporting vlm_vision_encoder.onnx (static)")
    print("=" * 60)

    wrapper = VisionEncoderForONNX(model.vlm.model.visual)
    wrapper = wrapper.float().eval()

    dummy_pixels = torch.randn(TOTAL_PATCHES, PATCH_DIM)
    dummy_grid = torch.tensor([[1, 20, 36]] * NUM_IMAGES, dtype=torch.long)
    output_path = os.path.join(output_dir, "common", "vlm_vision_encoder.onnx")

    with torch.no_grad():
        out = wrapper(dummy_pixels, dummy_grid)
        print(f"  Input:  pixel_values {dummy_pixels.shape}, grid_thw {dummy_grid.shape}")
        print(f"  Output: image_embeds {out[0].shape}, deepstack {out[1].shape}")

        torch.onnx.export(
            wrapper,
            (dummy_pixels, dummy_grid),
            output_path,
            input_names=["pixel_values", "grid_thw"],
            output_names=[
                "image_embeds",
                "deepstack_embed_0",
                "deepstack_embed_1",
                "deepstack_embed_2",
            ],
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
    print(f"  Saved: {output_path}")


def export_decoder_prefill_static(model, output_dir):
    """Export vlm_decoder_prefill with static shape (1, 3006, 4096)."""
    print("\n" + "=" * 60)
    print("Exporting vlm_decoder_prefill.onnx (static)")
    print("=" * 60)

    text_model = model.vlm.model.language_model
    wrapper = DecoderPrefillForONNX(text_model)
    wrapper = wrapper.float().eval()

    dummy_embeds = torch.randn(BATCH, PREFILL_SEQ_LEN, HIDDEN_SIZE)
    dummy_position_ids = torch.arange(PREFILL_SEQ_LEN).unsqueeze(0).unsqueeze(0).expand(3, BATCH, -1).long()
    dummy_attention_mask = torch.ones(BATCH, PREFILL_SEQ_LEN, dtype=torch.float32)
    dummy_ds0 = torch.randn(BATCH, PREFILL_SEQ_LEN, HIDDEN_SIZE) * 0.01
    dummy_ds1 = torch.randn(BATCH, PREFILL_SEQ_LEN, HIDDEN_SIZE) * 0.01
    dummy_ds2 = torch.randn(BATCH, PREFILL_SEQ_LEN, HIDDEN_SIZE) * 0.01

    output_path = os.path.join(output_dir, "decoder_prefill", "vlm_decoder_prefill.onnx")

    with torch.no_grad():
        print("  Testing forward pass...")
        out = wrapper(dummy_embeds, dummy_position_ids, dummy_attention_mask,
                      dummy_ds0, dummy_ds1, dummy_ds2)
        print(f"  Input:  inputs_embeds {dummy_embeds.shape}")
        print(f"  Output: hidden {out[0].shape}, past_keys {out[1].shape}, past_values {out[2].shape}")

        print("  Exporting to ONNX...")
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_position_ids, dummy_attention_mask,
             dummy_ds0, dummy_ds1, dummy_ds2),
            output_path,
            input_names=[
                "inputs_embeds", "position_ids", "attention_mask",
                "deepstack_full_0", "deepstack_full_1", "deepstack_full_2",
            ],
            output_names=["last_hidden_state", "past_keys", "past_values"],
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
    print(f"  Saved: {output_path}")


def export_decoder_decode_static(model, output_dir):
    """Export vlm_decoder_decode with STATIC KV cache at max size (3262)."""
    print("\n" + "=" * 60)
    print("Exporting vlm_decoder_decode.onnx (static, KV cache fixed at max)")
    print("=" * 60)

    text_model = model.vlm.model.language_model
    wrapper = DecoderDecodeForONNX(text_model)
    wrapper = wrapper.float().eval()

    # Fixed at max KV cache size
    past_seq_len = MAX_TOTAL_SEQ  # 3262

    dummy_embeds = torch.randn(BATCH, 1, HIDDEN_SIZE)
    dummy_position_ids = torch.tensor([[[past_seq_len]]], dtype=torch.long).expand(3, BATCH, 1)
    dummy_attention_mask = torch.zeros(BATCH, 1, 1, past_seq_len + 1, dtype=torch.float32)
    dummy_past_keys = torch.randn(NUM_LAYERS, BATCH, NUM_KV_HEADS, past_seq_len, HEAD_DIM)
    dummy_past_values = torch.randn(NUM_LAYERS, BATCH, NUM_KV_HEADS, past_seq_len, HEAD_DIM)

    output_path = os.path.join(output_dir, "decoder_decode", "vlm_decoder_decode.onnx")

    with torch.no_grad():
        print("  Testing forward pass...")
        out = wrapper(dummy_embeds, dummy_position_ids, dummy_attention_mask,
                      dummy_past_keys, dummy_past_values)
        print(f"  Input:  embeds {dummy_embeds.shape}, past_keys {dummy_past_keys.shape}")
        print(f"  Output: hidden {out[0].shape}, new_keys {out[1].shape}")

        print("  Exporting to ONNX...")
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_position_ids, dummy_attention_mask,
             dummy_past_keys, dummy_past_values),
            output_path,
            input_names=[
                "input_embeds", "position_ids", "attention_mask",
                "past_keys", "past_values",
            ],
            output_names=["hidden_state", "new_past_keys", "new_past_values"],
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
    print(f"  Saved: {output_path}")


def export_diffusion_step_static(model, output_dir):
    """Export diffusion_step_kvcache with static shapes."""
    print("\n" + "=" * 60)
    print("Exporting diffusion_step_kvcache.onnx (static)")
    print("=" * 60)

    diff_module = FullDiffusionStepWithKVCache(model)
    diff_module = diff_module.float().eval()

    context_len = MAX_TOTAL_SEQ  # 3262 — VLM's final KV cache length
    total_len = context_len + ACTION_LEN  # 3326

    dummy_actions = torch.randn(BATCH, ACTION_LEN, 2)
    dummy_timesteps = torch.rand(BATCH, 1)
    dummy_position_ids = torch.arange(ACTION_LEN).unsqueeze(0).unsqueeze(0).expand(3, BATCH, -1)
    dummy_mask = torch.zeros(BATCH, 1, ACTION_LEN, total_len)
    dummy_past_keys = torch.randn(NUM_LAYERS, BATCH, NUM_KV_HEADS, context_len, HEAD_DIM)
    dummy_past_values = torch.randn(NUM_LAYERS, BATCH, NUM_KV_HEADS, context_len, HEAD_DIM)

    output_path = os.path.join(output_dir, "expert", "diffusion_step_kvcache.onnx")

    with torch.no_grad():
        print("  Testing forward pass...")
        out = diff_module(dummy_actions, dummy_timesteps, dummy_position_ids,
                          dummy_mask, dummy_past_keys, dummy_past_values)
        print(f"  Input:  actions {dummy_actions.shape}, past_keys {dummy_past_keys.shape}")
        print(f"  Output: velocity {out.shape}")

        print("  Exporting to ONNX...")
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
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
    print(f"  Saved: {output_path}")


ALL_MODELS = {
    "embed_tokens": export_embed_tokens_static,
    "lm_head": export_lm_head_static,
    "vision_encoder": export_vision_encoder_static,
    "decoder_prefill": export_decoder_prefill_static,
    "decoder_decode": export_decoder_decode_static,
    "diffusion_step": export_diffusion_step_static,
}


def main():
    parser = argparse.ArgumentParser(description="Export Alpamayo R1 with static shapes for Qualcomm NPU")
    parser.add_argument("--model-path", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--output-dir", type=str, default="./onnx_models_static_fp16")
    parser.add_argument("--export-only", nargs="+", choices=list(ALL_MODELS.keys()),
                        help="Export only specific models")
    args = parser.parse_args()

    # Create output directories
    for subdir in ["common", "decoder_prefill", "decoder_decode", "expert"]:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    model = AlpamayoR1.from_pretrained(
        args.model_path, dtype=torch.float32, device_map="cpu"
    )
    model = model.float()
    model.eval()

    print(f"\nStatic shapes:")
    print(f"  batch={BATCH}, prefill_seq_len={PREFILL_SEQ_LEN}, max_total_seq={MAX_TOTAL_SEQ}")
    print(f"  num_images={NUM_IMAGES}, total_patches={TOTAL_PATCHES}, merged_seq={MERGED_SEQ}")
    print(f"  action_len={ACTION_LEN}, hidden_size={HIDDEN_SIZE}")

    models_to_export = args.export_only if args.export_only else list(ALL_MODELS.keys())

    for name in models_to_export:
        ALL_MODELS[name](model, args.output_dir)

    print("\n" + "=" * 60)
    print("All static exports complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    print("\nNext step: Convert to fp16:")
    print(f"  python convert_onnx_precision.py --input <model>.onnx --output <model>.onnx --precision fp16")


if __name__ == "__main__":
    main()
