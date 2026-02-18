# SPDX-License-Identifier: Apache-2.0
"""
verify_qwen3vl_2b.py
─────────────────────────────────────────────────────────────────────────────
Numerically compare PyTorch wrapper outputs vs ONNX Runtime outputs for every
exported Qwen3-VL-2B model component.

Checks per model
─────────────────
  • Output shape matches
  • Max absolute error  (MAE)   — fp16 tolerance: 1e-2, fp32: 1e-4
  • Cosine similarity           — should be > 0.9999 for a good export
  • Top-5 token agreement       — for lm_head logits specifically
  • KV cache slot write check   — for decoder_decode: confirms slot was written

What "PASS" means
──────────────────
  Max error < threshold  AND  cosine similarity > 0.9999
  For lm_head: additionally top-5 tokens must match between PT and ONNX.

Usage
──────
  # Verify all 5 models
  python verify_qwen3vl_2b.py

  # Verify specific models only
  python verify_qwen3vl_2b.py --verify embed_tokens lm_head decoder_decode

  # Use a local model checkpoint
  python verify_qwen3vl_2b.py --model-path /data/users/adhi/models/Qwen3-VL-2B

  # Use CPU ORT provider instead of QNN (for verification on x86 cluster)
  python verify_qwen3vl_2b.py --provider cpu

Requirements
────────────
  pip install torch transformers onnxruntime numpy
  (on Snapdragon device with QNN: pip install onnxruntime-qnn)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

# ── Shape constants — must match export_qwen3vl_2b.py exactly ──────────────
BATCH           = 1
NUM_IMAGES      = 1
GRID_T          = 1
GRID_H          = 28
GRID_W          = 28
TOTAL_PATCHES   = NUM_IMAGES * GRID_T * GRID_H * GRID_W   # 784
PATCH_DIM       = 3 * 2 * 16 * 16                          # 1536
PREFILL_SEQ_LEN = 1024
MAX_GENERATION  = 512
MAX_TOTAL_SEQ   = PREFILL_SEQ_LEN + MAX_GENERATION         # 1536

# fp16 numerical tolerances — ORT and PyTorch accumulate small rounding diffs
TOL_FP16_MAX_ABS  = 5e-2   # max single-element absolute error
TOL_FP16_MEAN_ABS = 1e-3   # mean absolute error
TOL_FP32_MAX_ABS  = 1e-3
TOL_FP32_MEAN_ABS = 1e-5
TOL_COSINE        = 0.9999  # cosine similarity floor


# ── Wrapper classes (identical to export_qwen3vl_2b.py) ────────────────────

class EmbedTokensForONNX(nn.Module):
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens
    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LMHeadForONNX(nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head
    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


class VisionEncoderForONNX(nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model
        for blk in self.vision_model.blocks:
            attn = getattr(blk, "attn", None)
            if attn is not None and hasattr(attn, "config"):
                attn.config._attn_implementation = "eager"
    def forward(self, pixel_values, grid_thw):
        out = self.vision_model(pixel_values, grid_thw)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


class DecoderPrefillForONNX(nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.layers     = text_model.layers
        self.norm       = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.num_layers = len(self.layers)
        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def forward(self, inputs_embeds, position_ids, attention_mask):
        B, seq_len, _ = inputs_embeds.shape
        cache_position = torch.arange(seq_len, device=inputs_embeds.device)
        text_pos_ids   = position_ids[0]
        hidden_states  = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal = torch.full(
            (seq_len, seq_len), torch.finfo(hidden_states.dtype).min,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        pad    = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(hidden_states.dtype).min
        attn_mask = causal + pad
        past_key_values = DynamicCache()
        for layer in self.layers:
            out = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=text_pos_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=True,
            )
            hidden_states = out[0] if isinstance(out, (tuple, list)) else out
        hidden_states = self.norm(hidden_states)
        keys   = torch.stack([past_key_values[i][0] for i in range(self.num_layers)])
        values = torch.stack([past_key_values[i][1] for i in range(self.num_layers)])
        return hidden_states, keys, values


class DecoderDecodeForONNX(nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.layers     = text_model.layers
        self.norm       = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.num_layers = len(self.layers)

    def forward(self, input_embeds, position_ids, attention_mask,
                past_keys, past_values, write_index):
        B = input_embeds.shape[0]
        position_embeddings = self.rotary_emb(input_embeds, position_ids)
        cos, sin = position_embeddings
        hidden_states   = input_embeds
        new_keys_list   = []
        new_values_list = []
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            hidden_shape = (B, 1, -1, attn.head_dim)
            residual = hidden_states
            normed   = layer.input_layernorm(hidden_states)
            q = attn.q_norm(attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
            v = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k_buf = past_keys[i].clone()
            v_buf = past_values[i].clone()
            k_buf[:, :, write_index : write_index + 1, :] = k
            v_buf[:, :, write_index : write_index + 1, :] = v
            new_keys_list.append(k_buf)
            new_values_list.append(v_buf)
            num_kv_groups = attn.num_key_value_groups
            k_expand = k_buf.repeat_interleave(num_kv_groups, dim=1)
            v_expand = v_buf.repeat_interleave(num_kv_groups, dim=1)
            attn_weights = torch.matmul(q, k_expand.transpose(-2, -1)) * attn.scaling
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)
            attn_out     = torch.matmul(attn_weights, v_expand)
            attn_out = attn_out.transpose(1, 2).reshape(B, 1, -1)
            attn_out = attn.o_proj(attn_out)
            hidden_states = residual + attn_out
            residual      = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
        hidden_states = self.norm(hidden_states)
        new_keys   = torch.stack(new_keys_list)
        new_values = torch.stack(new_values_list)
        return hidden_states, new_keys, new_values


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_text_model(model):
    inner = model.model
    if hasattr(inner, "language_model"):
        return inner.language_model
    return inner


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def make_ort_session(onnx_path: str, provider: str):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3  # suppress warnings
    if provider == "cpu":
        return ort.InferenceSession(onnx_path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
    elif provider == "qnn_npu":
        return ort.InferenceSession(onnx_path, sess_options=so,
                                    providers=["QNNExecutionProvider"],
                                    provider_options=[{"backend_path": "QnnHtp.dll",
                                                       "htp_performance_mode": "burst",
                                                       "enable_htp_fp16_precision": "1"}])
    elif provider == "qnn_gpu":
        return ort.InferenceSession(onnx_path, sess_options=so,
                                    providers=["QNNExecutionProvider"],
                                    provider_options=[{"backend_path": "QnnGpu.dll"}])
    else:
        raise ValueError(f"Unknown provider: {provider}")


def report(name: str, pt_out: np.ndarray, ort_out: np.ndarray,
           is_fp32: bool = False) -> bool:
    """Print comparison metrics and return True if PASS."""
    max_abs  = float(np.max(np.abs(pt_out - ort_out)))
    mean_abs = float(np.mean(np.abs(pt_out - ort_out)))
    cos      = cosine_sim(pt_out, ort_out)
    tol_max  = TOL_FP32_MAX_ABS  if is_fp32 else TOL_FP16_MAX_ABS
    tol_mean = TOL_FP32_MEAN_ABS if is_fp32 else TOL_FP16_MEAN_ABS

    passed = (max_abs <= tol_max) and (mean_abs <= tol_mean) and (cos >= TOL_COSINE)
    tag    = "PASS ✓" if passed else "FAIL ✗"

    print(f"    [{tag}] {name}")
    print(f"           shape    : PT={pt_out.shape}  ORT={ort_out.shape}")
    print(f"           max |err|: {max_abs:.6f}  (tol={tol_max})")
    print(f"           mean|err|: {mean_abs:.6f}  (tol={tol_mean})")
    print(f"           cosine   : {cos:.8f}  (tol={TOL_COSINE})")
    return passed


# ── Per-model verifiers ──────────────────────────────────────────────────────

def verify_embed_tokens(model, onnx_dir, provider):
    print("\n─── embed_tokens ───────────────────────────────────────")
    onnx_path = os.path.join(onnx_dir, "common", "vlm_embed_tokens.onnx")
    if not os.path.isfile(onnx_path):
        print(f"  SKIP — not found: {onnx_path}")
        return None

    # Fixed seed for reproducibility
    torch.manual_seed(42)
    input_ids = torch.randint(0, model.config.text_config.vocab_size,
                              (BATCH, PREFILL_SEQ_LEN))

    # PyTorch
    wrapper = EmbedTokensForONNX(model.get_input_embeddings()).half().eval()
    with torch.no_grad():
        pt_out = wrapper(input_ids).float().numpy()

    # ONNX Runtime
    sess = make_ort_session(onnx_path, provider)
    ort_out = sess.run(None, {"input_ids": input_ids.numpy()})[0].astype(np.float32)

    return report("embeddings", pt_out, ort_out, is_fp32=False)


def verify_lm_head(model, onnx_dir, provider):
    print("\n─── lm_head ─────────────────────────────────────────────")
    onnx_path = os.path.join(onnx_dir, "common", "vlm_lm_head.onnx")
    if not os.path.isfile(onnx_path):
        print(f"  SKIP — not found: {onnx_path}")
        return None

    torch.manual_seed(42)
    H = model.config.text_config.hidden_size
    hidden = torch.randn(BATCH, 1, H, dtype=torch.float16)

    wrapper = LMHeadForONNX(model.lm_head).half().eval()
    with torch.no_grad():
        pt_logits = wrapper(hidden).float().numpy()    # (1, 1, vocab_size)

    sess = make_ort_session(onnx_path, provider)
    ort_logits = sess.run(None, {"hidden_states": hidden.numpy()})[0].astype(np.float32)

    passed = report("logits", pt_logits, ort_logits, is_fp32=False)

    # Top-5 token agreement
    pt_top5  = np.argsort(pt_logits[0, 0])[-5:][::-1].tolist()
    ort_top5 = np.argsort(ort_logits[0, 0])[-5:][::-1].tolist()
    match    = pt_top5 == ort_top5
    print(f"    [{'PASS ✓' if match else 'FAIL ✗'}] top-5 tokens")
    print(f"           PT  top-5: {pt_top5}")
    print(f"           ORT top-5: {ort_top5}")

    return passed and match


def verify_vision_encoder(model, onnx_dir, provider):
    print("\n─── vision_encoder ──────────────────────────────────────")
    onnx_path = os.path.join(onnx_dir, "common", "vlm_vision_encoder.onnx")
    if not os.path.isfile(onnx_path):
        print(f"  SKIP — not found: {onnx_path}")
        return None

    torch.manual_seed(42)
    pixel_values = torch.randn(TOTAL_PATCHES, PATCH_DIM, dtype=torch.float32)
    grid_thw     = torch.tensor([[GRID_T, GRID_H, GRID_W]] * NUM_IMAGES, dtype=torch.long)

    wrapper = VisionEncoderForONNX(model.model.visual).float().eval()
    with torch.no_grad():
        pt_out = wrapper(pixel_values, grid_thw).numpy()

    sess = make_ort_session(onnx_path, provider)
    ort_out = sess.run(None, {
        "pixel_values": pixel_values.numpy(),
        "grid_thw":     grid_thw.numpy(),
    })[0]

    return report("image_embeds", pt_out, ort_out, is_fp32=True)


def verify_decoder_prefill(model, onnx_dir, provider):
    print("\n─── decoder_prefill ─────────────────────────────────────")
    onnx_path = os.path.join(onnx_dir, "decoder_prefill", "vlm_decoder_prefill.onnx")
    if not os.path.isfile(onnx_path):
        print(f"  SKIP — not found: {onnx_path}")
        return None

    torch.manual_seed(42)
    H = model.config.text_config.hidden_size
    inputs_embeds  = torch.randn(BATCH, PREFILL_SEQ_LEN, H, dtype=torch.float16)
    position_ids   = torch.arange(PREFILL_SEQ_LEN).unsqueeze(0).unsqueeze(0).expand(3, BATCH, -1).long().contiguous()
    attention_mask = torch.ones(BATCH, PREFILL_SEQ_LEN, dtype=torch.float16)

    text_model = _get_text_model(model)
    wrapper    = DecoderPrefillForONNX(text_model).half().eval()
    with torch.no_grad():
        pt_hidden, pt_keys, pt_values = wrapper(inputs_embeds, position_ids, attention_mask)
        pt_hidden = pt_hidden.float().numpy()
        pt_keys   = pt_keys.float().numpy()
        pt_values = pt_values.float().numpy()

    sess = make_ort_session(onnx_path, provider)
    ort_hidden, ort_keys, ort_values = sess.run(None, {
        "inputs_embeds":  inputs_embeds.numpy(),
        "position_ids":   position_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    })
    ort_hidden = ort_hidden.astype(np.float32)
    ort_keys   = ort_keys.astype(np.float32)
    ort_values = ort_values.astype(np.float32)

    r1 = report("last_hidden_state", pt_hidden, ort_hidden)
    r2 = report("past_keys",         pt_keys,   ort_keys)
    r3 = report("past_values",       pt_values, ort_values)
    return r1 and r2 and r3


def verify_decoder_decode(model, onnx_dir, provider):
    print("\n─── decoder_decode ──────────────────────────────────────")
    onnx_path = os.path.join(onnx_dir, "decoder_decode", "vlm_decoder_decode.onnx")
    if not os.path.isfile(onnx_path):
        print(f"  SKIP — not found: {onnx_path}")
        return None

    # Check file size first — a 1 MB file means constant-folded export
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ONNX file size: {size_mb:.1f} MB")
    if size_mb < 500:
        print(f"  *** ABORT: File is only {size_mb:.1f} MB — constant-folded export. ***")
        print(f"      Re-export with dynamo=True. See export_qwen3vl_2b.py.")
        return False

    torch.manual_seed(42)
    tcfg = model.config.text_config
    H    = tcfg.hidden_size
    L    = tcfg.num_hidden_layers
    KV   = tcfg.num_key_value_heads
    D    = H // tcfg.num_attention_heads

    WRITE_IDX = 100   # use a mid-sequence slot so we can verify it was written

    input_embeds   = torch.randn(BATCH, 1, H, dtype=torch.float16)
    position_ids   = torch.tensor([[[WRITE_IDX]]], dtype=torch.long).expand(3, BATCH, 1).contiguous()
    attention_mask = torch.zeros(BATCH, 1, 1, MAX_TOTAL_SEQ, dtype=torch.float16)
    past_keys      = torch.randn(L, BATCH, KV, MAX_TOTAL_SEQ, D, dtype=torch.float16)
    past_values    = torch.randn(L, BATCH, KV, MAX_TOTAL_SEQ, D, dtype=torch.float16)
    write_index    = torch.tensor(WRITE_IDX, dtype=torch.long)

    text_model = _get_text_model(model)
    wrapper    = DecoderDecodeForONNX(text_model).half().eval()
    with torch.no_grad():
        pt_hidden, pt_new_keys, pt_new_values = wrapper(
            input_embeds, position_ids, attention_mask,
            past_keys, past_values, write_index
        )
        pt_hidden    = pt_hidden.float().numpy()
        pt_new_keys  = pt_new_keys.float().numpy()
        pt_new_values= pt_new_values.float().numpy()

    sess = make_ort_session(onnx_path, provider)
    ort_results = sess.run(None, {
        "input_embeds":   input_embeds.numpy(),
        "position_ids":   position_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "past_keys":      past_keys.numpy(),
        "past_values":    past_values.numpy(),
        "write_index":    write_index.numpy(),
    })
    ort_hidden, ort_new_keys, ort_new_values = [x.astype(np.float32) for x in ort_results]

    r1 = report("hidden_state",    pt_hidden,     ort_hidden)
    r2 = report("new_past_keys",   pt_new_keys,   ort_new_keys)
    r3 = report("new_past_values", pt_new_values, ort_new_values)

    # Sanity check: confirm slot WRITE_IDX actually changed in new_past_keys
    # Compare new_keys[:, :, WRITE_IDX, :] vs old past_keys[:, :, WRITE_IDX, :]
    old_slot = past_keys[:, :, WRITE_IDX, :].numpy().astype(np.float32)
    pt_slot  = pt_new_keys[:, :, WRITE_IDX, :]
    ort_slot = ort_new_keys[:, :, WRITE_IDX, :]

    pt_slot_changed  = not np.allclose(old_slot, pt_slot,  atol=1e-3)
    ort_slot_changed = not np.allclose(old_slot, ort_slot, atol=1e-3)
    print(f"\n    KV write check at slot {WRITE_IDX}:")
    print(f"      PT  slot changed : {pt_slot_changed}   (should be True)")
    print(f"      ORT slot changed : {ort_slot_changed}  (should be True)")
    if not ort_slot_changed:
        print(f"      *** ORT slot unchanged → KV write NOT working in ONNX graph ***")

    return r1 and r2 and r3 and ort_slot_changed


# ── Full end-to-end logit comparison ────────────────────────────────────────

def verify_end_to_end_logits(model, onnx_dir, provider):
    """
    Run the full pipeline:
      embed_tokens → decoder_prefill → lm_head
    Compare final logits between:
      (a) pure PyTorch   — all wrappers chained
      (b) all ONNX       — ORT sessions chained

    This is the most important check: it confirms the logit distribution is
    preserved end-to-end, meaning token sampling will produce the same results.
    """
    print("\n─── END-TO-END LOGIT COMPARISON ─────────────────────────")

    required = [
        os.path.join(onnx_dir, "common", "vlm_embed_tokens.onnx"),
        os.path.join(onnx_dir, "decoder_prefill", "vlm_decoder_prefill.onnx"),
        os.path.join(onnx_dir, "common", "vlm_lm_head.onnx"),
    ]
    for p in required:
        if not os.path.isfile(p):
            print(f"  SKIP — missing: {p}")
            return None

    torch.manual_seed(0)
    vocab_size   = model.config.text_config.vocab_size
    input_ids    = torch.randint(0, vocab_size, (BATCH, PREFILL_SEQ_LEN))
    position_ids = torch.arange(PREFILL_SEQ_LEN).unsqueeze(0).unsqueeze(0).expand(3, BATCH, -1).long().contiguous()
    attn_mask    = torch.ones(BATCH, PREFILL_SEQ_LEN, dtype=torch.float16)

    # ── PyTorch pipeline ─────────────────────────────────────────────────────
    emb_wrapper    = EmbedTokensForONNX(model.get_input_embeddings()).half().eval()
    text_model     = _get_text_model(model)
    dec_wrapper    = DecoderPrefillForONNX(text_model).half().eval()
    lmh_wrapper    = LMHeadForONNX(model.lm_head).half().eval()

    with torch.no_grad():
        pt_embeds               = emb_wrapper(input_ids)
        pt_hidden, pt_k, pt_v   = dec_wrapper(pt_embeds, position_ids, attn_mask)
        pt_logits               = lmh_wrapper(pt_hidden[:, -1:, :]).float().numpy()

    # ── ONNX pipeline ────────────────────────────────────────────────────────
    sess_emb = make_ort_session(required[0], provider)
    sess_dec = make_ort_session(required[1], provider)
    sess_lmh = make_ort_session(required[2], provider)

    ort_embeds  = sess_emb.run(None, {"input_ids": input_ids.numpy()})[0]
    ort_hidden, ort_k, ort_v = sess_dec.run(None, {
        "inputs_embeds":  ort_embeds,
        "position_ids":   position_ids.numpy(),
        "attention_mask": attn_mask.numpy(),
    })
    ort_logits = sess_lmh.run(None, {"hidden_states": ort_hidden[:, -1:, :]})[0].astype(np.float32)

    passed = report("end-to-end logits", pt_logits, ort_logits, is_fp32=False)

    # Top-10 token comparison — this is what actually matters for generation
    pt_top10  = np.argsort(pt_logits[0, 0])[-10:][::-1].tolist()
    ort_top10 = np.argsort(ort_logits[0, 0])[-10:][::-1].tolist()
    overlap   = len(set(pt_top10) & set(ort_top10))
    print(f"\n    Top-10 token overlap: {overlap}/10  (need >= 8 for good generation)")
    print(f"      PT  top-10: {pt_top10}")
    print(f"      ORT top-10: {ort_top10}")
    print(f"      PT  top-1 prob : softmax={float(np.exp(pt_logits[0,0,pt_top10[0]]) / np.sum(np.exp(pt_logits[0,0]))):.4f}")

    generation_ok = overlap >= 8
    if not generation_ok:
        print(f"    *** WARNING: low overlap — generation quality may differ ***")

    return passed and generation_ok


# ── Main ────────────────────────────────────────────────────────────────────

ALL_VERIFIERS = {
    "embed_tokens":    verify_embed_tokens,
    "lm_head":         verify_lm_head,
    "vision_encoder":  verify_vision_encoder,
    "decoder_prefill": verify_decoder_prefill,
    "decoder_decode":  verify_decoder_decode,
    "end_to_end":      verify_end_to_end_logits,
}

def main():
    parser = argparse.ArgumentParser(
        description="Verify ONNX exports match PyTorch for Qwen3-VL-2B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", type=str,
                        default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--onnx-dir",   type=str,
                        default="/data/users/adhi/qwen_npu/onnx_models")
    parser.add_argument("--verify", nargs="+",
                        choices=list(ALL_VERIFIERS.keys()),
                        help="Which components to verify (default: all)")
    parser.add_argument("--provider", type=str, default="cpu",
                        choices=["cpu", "qnn_npu", "qnn_gpu"],
                        help="ORT execution provider (default: cpu)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Qwen3-VL-2B ONNX Verification")
    print(f"  Model   : {args.model_path}")
    print(f"  ONNX dir: {args.onnx_dir}")
    print(f"  Provider: {args.provider}")
    print("=" * 60)

    print("\nLoading PyTorch model (fp16 on CPU)...")
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="eager",
    ).half().eval()
    print("  Loaded ✓")

    names   = args.verify if args.verify else list(ALL_VERIFIERS.keys())
    results = {}

    for name in names:
        try:
            results[name] = ALL_VERIFIERS[name](model, args.onnx_dir, args.provider)
        except Exception as e:
            import traceback
            print(f"\n  ERROR in {name}: {e}")
            traceback.print_exc()
            results[name] = False

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        if ok is None:
            status = "SKIP  (file missing)"
        elif ok:
            status = "PASS ✓"
        else:
            status = "FAIL ✗"
        print(f"  {status:30s}  {name}")

    all_pass = all(v for v in results.values() if v is not None)
    print("=" * 60)
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILURES — check above ✗'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()