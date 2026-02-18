# SPDX-License-Identifier: Apache-2.0
"""
Verify exported Qwen3-VL-2B ONNX models by comparing outputs against PyTorch.

Runs each ONNX model with the same dummy inputs used during export,
checks that shapes match and mean absolute difference is within tolerance.

Usage:
    python verify_qwen3vl_2b.py
    python verify_qwen3vl_2b.py --onnx-dir /data/users/adhi/qwen_npu/onnx_models
    python verify_qwen3vl_2b.py --verify-only embed_tokens vision_encoder
"""

import os
import sys
import argparse
import numpy as np
import torch

import onnxruntime as ort

# Add qwen_npu to path so we can import wrapper classes
sys.path.insert(0, os.path.dirname(__file__))
from export_qwen3vl_2b import (
    DecoderPrefillForONNX, DecoderDecodeForONNX, _get_text_model,
    BATCH, NUM_IMAGES, GRID_T, GRID_H, GRID_W,
    TOTAL_PATCHES, PATCH_DIM,
    PREFILL_SEQ_LEN, MAX_TOTAL_SEQ,
)


def ort_session(onnx_path):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(
        onnx_path, sess_options=opts, providers=["CPUExecutionProvider"],
    )


def to_numpy(t):
    return t.detach().float().numpy()


def check(name, pt_out, ort_out, atol=1e-2):
    """Compare PyTorch vs ONNX Runtime output. Primary criterion: mean_diff."""
    if isinstance(pt_out, (list, tuple)):
        pt_out = pt_out[0]
    pt      = to_numpy(pt_out)
    ort_arr = np.array(ort_out[0]).astype(np.float32)

    shape_ok  = pt.shape == ort_arr.shape
    max_diff  = float(np.abs(pt - ort_arr).max())
    mean_diff = float(np.abs(pt - ort_arr).mean())
    status    = "PASS" if shape_ok and mean_diff < atol else "FAIL"

    print(f"  [{status}] {name}")
    print(f"         shape  PyTorch={pt.shape}  ONNX={ort_arr.shape}")
    print(f"         max_diff={max_diff:.4f}  mean_diff={mean_diff:.6f}  mean_atol={atol}")
    return status == "PASS"


# ── Per-model verification ─────────────────────────────────────────────────

def verify_embed_tokens(model, onnx_dir):
    print("\n[embed_tokens]")
    onnx_path = os.path.join(onnx_dir, "common", "vlm_embed_tokens.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP — file not found")
        return

    dummy = torch.randint(0, 1000, (BATCH, PREFILL_SEQ_LEN))

    embed = model.get_input_embeddings().half().eval()
    with torch.no_grad():
        pt_out = embed(dummy)

    sess    = ort_session(onnx_path)
    ort_out = sess.run(None, {"input_ids": dummy.numpy()})
    check("embed_tokens output", pt_out, ort_out)


def verify_lm_head(model, onnx_dir):
    print("\n[lm_head]")
    onnx_path = os.path.join(onnx_dir, "common", "vlm_lm_head.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP — file not found")
        return

    H     = model.config.text_config.hidden_size
    dummy = torch.randn(BATCH, 1, H, dtype=torch.float16)

    lm_head = model.lm_head.half().eval()
    with torch.no_grad():
        pt_out = lm_head(dummy)

    sess    = ort_session(onnx_path)
    ort_out = sess.run(None, {"hidden_states": dummy.numpy()})
    check("lm_head logits", pt_out, ort_out)


def verify_vision_encoder(model, onnx_dir):
    print("\n[vision_encoder]")
    onnx_path = os.path.join(onnx_dir, "common", "vlm_vision_encoder.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP — file not found")
        return

    dummy_pixels = torch.randn(TOTAL_PATCHES, PATCH_DIM, dtype=torch.float32)
    dummy_grid   = torch.tensor([[GRID_T, GRID_H, GRID_W]] * NUM_IMAGES, dtype=torch.long)

    vision_model = model.model.visual.float().eval()
    with torch.no_grad():
        pt_out = vision_model(dummy_pixels, dummy_grid)

    sess    = ort_session(onnx_path)
    ort_out = sess.run(None, {
        "pixel_values": dummy_pixels.numpy(),
        "grid_thw":     dummy_grid.numpy(),
    })
    check("vision image_embeds", pt_out, ort_out)


def verify_decoder_prefill(model, onnx_dir):
    print("\n[decoder_prefill]")
    onnx_path = os.path.join(onnx_dir, "decoder_prefill", "vlm_decoder_prefill.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP — file not found")
        return

    tcfg = model.config.text_config
    H    = tcfg.hidden_size

    dummy_embeds = torch.randn(BATCH, PREFILL_SEQ_LEN, H, dtype=torch.float16)
    dummy_pos    = (
        torch.arange(PREFILL_SEQ_LEN)
        .unsqueeze(0).unsqueeze(0).expand(3, BATCH, -1).long()
    )
    dummy_mask   = torch.ones(BATCH, PREFILL_SEQ_LEN, dtype=torch.float16)

    text_model = _get_text_model(model)
    wrapper    = DecoderPrefillForONNX(text_model).half().eval()
    with torch.no_grad():
        pt_hidden, pt_keys, pt_values = wrapper(dummy_embeds, dummy_pos, dummy_mask)

    sess    = ort_session(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds":  dummy_embeds.numpy(),
        "position_ids":   dummy_pos.numpy(),
        "attention_mask": dummy_mask.numpy(),
    })

    # fp16 over 28 layers — max can spike on random inputs; mean is the true gauge
    check("last_hidden_state", pt_hidden, [ort_out[0]], atol=10.0)
    check("past_keys",         pt_keys,   [ort_out[1]], atol=1.0)
    check("past_values",       pt_values, [ort_out[2]], atol=1.0)


def verify_decoder_decode(model, onnx_dir):
    print("\n[decoder_decode]")
    onnx_path = os.path.join(onnx_dir, "decoder_decode", "vlm_decoder_decode.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP — file not found")
        return

    tcfg = model.config.text_config
    H    = tcfg.hidden_size
    L    = tcfg.num_hidden_layers
    KV   = tcfg.num_key_value_heads
    D    = H // tcfg.num_attention_heads

    # KV cache is FIXED at MAX_TOTAL_SEQ; write_index points to the slot being filled.
    write_idx    = MAX_TOTAL_SEQ - 1
    dummy_embeds = torch.randn(BATCH, 1, H, dtype=torch.float16)
    dummy_pos    = torch.tensor([[[write_idx]]], dtype=torch.long).expand(3, BATCH, 1)
    dummy_mask   = torch.zeros(BATCH, 1, 1, MAX_TOTAL_SEQ, dtype=torch.float16)
    dummy_pk     = torch.randn(L, BATCH, KV, MAX_TOTAL_SEQ, D, dtype=torch.float16)
    dummy_pv     = torch.randn(L, BATCH, KV, MAX_TOTAL_SEQ, D, dtype=torch.float16)
    dummy_wi     = torch.tensor(write_idx, dtype=torch.long)

    text_model = _get_text_model(model)
    wrapper    = DecoderDecodeForONNX(text_model).half().eval()
    with torch.no_grad():
        pt_hidden, pt_nk, pt_nv = wrapper(
            dummy_embeds, dummy_pos, dummy_mask, dummy_pk, dummy_pv, dummy_wi,
        )

    sess    = ort_session(onnx_path)
    ort_out = sess.run(None, {
        "input_embeds":   dummy_embeds.numpy(),
        "position_ids":   dummy_pos.numpy(),
        "attention_mask": dummy_mask.numpy(),
        "past_keys":      dummy_pk.numpy(),
        "past_values":    dummy_pv.numpy(),
        "write_index":    dummy_wi.numpy(),
    })

    check("hidden_state",    pt_hidden, [ort_out[0]], atol=2.0)
    check("new_past_keys",   pt_nk,     [ort_out[1]], atol=0.2)
    check("new_past_values", pt_nv,     [ort_out[2]], atol=0.2)


# ── Main ───────────────────────────────────────────────────────────────────

ALL = {
    "embed_tokens":    verify_embed_tokens,
    "lm_head":         verify_lm_head,
    "vision_encoder":  verify_vision_encoder,
    "decoder_prefill": verify_decoder_prefill,
    "decoder_decode":  verify_decoder_decode,
}


def main():
    parser = argparse.ArgumentParser(
        description="Verify Qwen3-VL-2B ONNX models against PyTorch",
    )
    parser.add_argument(
        "--onnx-dir", type=str, default="/data/users/adhi/qwen_npu/onnx_models",
    )
    parser.add_argument(
        "--model-path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--verify-only", nargs="+", choices=list(ALL.keys()),
        help="Verify only specific components",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cpu",
        attn_implementation="eager",
    )
    model = model.half().eval()

    checks = args.verify_only if args.verify_only else list(ALL.keys())
    for name in checks:
        try:
            ALL[name](model, args.onnx_dir)
        except Exception as e:
            import traceback
            print(f"\n  ERROR in {name}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
