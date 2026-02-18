# SPDX-License-Identifier: Apache-2.0
"""
Export Qwen3-VL-2B-Instruct to 5 static-shape ONNX models for Qualcomm NPU.

Architecture (Qwen3-VL-2B-Instruct):
  hidden_size     = 2048
  num_layers      = 28
  num_attn_heads  = 16
  num_kv_heads    = 8
  head_dim        = 128

STATIC CACHE DESIGN
====================
The NPU requires every tensor shape known at compile time.
No DynamicCache. No Python objects. No dynamic shapes. No in-place ops.
No torch.arange inside forward(). Everything is a tensor operation.

decoder_prefill
  Inputs : inputs_embeds (1,1024,2048)  position_ids (3,1,1024)
           attention_mask (1,1024)
  Outputs: last_hidden_state (1,1024,2048)
           past_key_0 .. past_key_27     each (1,8,1024,128)
           past_value_0 .. past_value_27 each (1,8,1024,128)
  57 total outputs. No input KV -- first pass, cannot be constant-folded.

decoder_decode
  Inputs : input_embeds (1,1,2048)  position_ids (3,1,1)
           attention_mask (1,1,1,1536)  write_index () int64
           past_key_0 .. past_key_27     each (1,8,1536,128)
           past_value_0 .. past_value_27 each (1,8,1536,128)
  Outputs: hidden_state (1,1,2048)
           new_past_key_0 .. new_past_key_27     each (1,8,1536,128)
           new_past_value_0 .. new_past_value_27 each (1,8,1536,128)
  61 inputs, 57 outputs.

  KV write: torch.scatter(dim=2) = ONNX ScatterElements, native on QNN HTP.

  Why 28 separate inputs (not one stacked tensor):
    TorchScript unrolls for-loops. stacked[i] with integer i becomes a
    constant slice baked into the ONNX graph. do_constant_folding then
    removes the whole compute subgraph -> 1 MB file.
    Named ONNX inputs are ALWAYS symbolic -- never constant-folded.

Both decoders: TorchScript export (dynamo=False), do_constant_folding=True.
"""

import os
import argparse
import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

# ── Constants ────────────────────────────────────────────────────────────────
BATCH           = 1
NUM_IMAGES      = 1
GRID_T, GRID_H, GRID_W = 1, 28, 28
SPATIAL_MERGE   = 2
TOTAL_PATCHES   = NUM_IMAGES * GRID_T * GRID_H * GRID_W   # 784
PATCH_DIM       = 3 * 2 * 16 * 16                          # 1536
PREFILL_SEQ_LEN = 1024
MAX_GENERATION  = 512
MAX_TOTAL_SEQ   = PREFILL_SEQ_LEN + MAX_GENERATION         # 1536
NUM_LAYERS      = 28
NUM_KV_HEADS    = 8
NUM_HEADS       = 16
HEAD_DIM        = 128
HIDDEN_SIZE     = 2048
OPSET_VERSION   = 17


# ── Simple wrappers ──────────────────────────────────────────────────────────

class EmbedTokensForONNX(nn.Module):
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class LMHeadForONNX(nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


class VisionEncoderForONNX(nn.Module):
    """fp32 only -- ONNX Range op does not support fp16."""
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model
        for blk in self.vision_model.blocks:
            attn = getattr(blk, "attn", None)
            if attn is not None and hasattr(attn, "config"):
                attn.config._attn_implementation = "eager"
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(pixel_values, grid_thw)
        return out[0] if isinstance(out, (tuple, list)) else out


# ── decoder_prefill ──────────────────────────────────────────────────────────

class DecoderPrefillForONNX(nn.Module):
    """
    Prefill: PREFILL_SEQ_LEN tokens in, static KV cache out.

    No input KV cache -- this is the first pass.
    All 28 K tensors and 28 V tensors are pure graph outputs computed from
    the input embeddings via learned weight matrices. They CANNOT be
    constant-folded because they are outputs that flow to downstream nodes.

    Zero Python objects, zero DynamicCache, zero dynamic shapes.
    """

    def __init__(self, text_model):
        super().__init__()
        self.layers     = text_model.layers
        self.norm       = text_model.norm
        self.rotary_emb = text_model.rotary_emb

    def forward(
        self,
        inputs_embeds:  torch.Tensor,   # (B, S, H)
        position_ids:   torch.Tensor,   # (3, B, S) int64
        attention_mask: torch.Tensor,   # (B, S)    fp16  1=keep 0=pad
    ):
        B, S, _ = inputs_embeds.shape

        # MRoPE cos/sin for full sequence
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)

        # Static causal + padding mask: (B, 1, S, S)
        causal = torch.full(
            (S, S), torch.finfo(inputs_embeds.dtype).min,
            dtype=inputs_embeds.dtype, device=inputs_embeds.device,
        )
        causal    = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)  # (1,1,S,S)
        pad       = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(inputs_embeds.dtype).min
        attn_bias = causal + pad                                               # (B,1,S,S)

        hidden   = inputs_embeds
        out_keys = []
        out_vals = []

        nh   = NUM_HEADS
        nkv  = NUM_KV_HEADS
        nkvg = NUM_HEADS // NUM_KV_HEADS
        hd   = HEAD_DIM
        scaling = HEAD_DIM ** -0.5

        for layer in self.layers:
            attn = layer.self_attn

            residual = hidden
            x        = layer.input_layernorm(hidden)

            # QKV -- (B, heads, S, hd)
            q = attn.q_norm(attn.q_proj(x).view(B, S, nh,  hd).transpose(1, 2))
            k = attn.k_norm(attn.k_proj(x).view(B, S, nkv, hd).transpose(1, 2))
            v =             attn.v_proj(x).view(B, S, nkv, hd).transpose(1, 2)

            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Accumulate KV -- these become live graph outputs, not constants
            out_keys.append(k)   # (B, nkv, S, hd)
            out_vals.append(v)   # (B, nkv, S, hd)

            # GQA
            k_e = k.repeat_interleave(nkvg, dim=1)
            v_e = v.repeat_interleave(nkvg, dim=1)

            # Attention
            w = torch.matmul(q, k_e.transpose(-2, -1)) * scaling
            w = w + attn_bias
            w = torch.softmax(w.float(), dim=-1).to(q.dtype)
            o = torch.matmul(w, v_e)

            o      = o.transpose(1, 2).reshape(B, S, nh * hd)
            o      = attn.o_proj(o)
            hidden = residual + o

            residual = hidden
            hidden   = layer.post_attention_layernorm(hidden)
            hidden   = layer.mlp(hidden)
            hidden   = residual + hidden

        hidden = self.norm(hidden)

        # Returns: (last_hidden, k_0, k_1, ..., k_27, v_0, v_1, ..., v_27)
        return tuple([hidden] + out_keys + out_vals)


# ── decoder_decode ───────────────────────────────────────────────────────────

class DecoderDecodeForONNX(nn.Module):
    """
    Decode: 1 token in, updated static KV cache out.

    Receives 28 + 28 = 56 separate per-layer KV tensors as individual inputs.
    Returns 28 + 28 = 56 updated per-layer KV tensors as individual outputs.

    KV write: torch.scatter(dim=2)
      Maps to ONNX ScatterElements -- supported natively by QNN HTP.
      Functional (returns new tensor). TorchScript-traceable with runtime index.

    Why *kv_cache varargs (not a stacked input tensor):
      A stacked (28, B, KV, T, D) tensor indexed with stacked[i] inside a
      for-loop causes TorchScript to constant-fold every slice to the dummy
      value used during tracing. Result: 1 MB ONNX file.

      Named ONNX graph inputs are ALWAYS symbolic. Passing 28 separate named
      tensors guarantees each one is a live runtime value in the graph.
    """

    def __init__(self, text_model, num_layers, num_kv_heads, max_total_seq, head_dim):
        super().__init__()
        self.layers        = text_model.layers
        self.norm          = text_model.norm
        self.rotary_emb    = text_model.rotary_emb
        self.num_layers    = num_layers
        self.num_kv_heads  = num_kv_heads
        self.max_total_seq = max_total_seq
        self.head_dim      = head_dim

    def forward(
        self,
        input_embeds:   torch.Tensor,   # (B, 1, H)
        position_ids:   torch.Tensor,   # (3, B, 1) int64
        attention_mask: torch.Tensor,   # (B, 1, 1, T) fp16
        write_index:    torch.Tensor,   # () scalar int64
        *kv_cache,                      # k_0..k_27 then v_0..v_27
                                        # each (B, KV_HEADS, T, HEAD_DIM)
    ):
        past_keys   = list(kv_cache[:self.num_layers])
        past_values = list(kv_cache[self.num_layers:])

        B = input_embeds.shape[0]

        # MRoPE for single new token
        cos, sin = self.rotary_emb(input_embeds, position_ids)

        # Scatter index: (B, KV_HEADS, 1, HEAD_DIM) -- all static except slot
        idx = (write_index
               .view(1, 1, 1, 1)
               .expand(B, self.num_kv_heads, 1, self.head_dim))

        hidden   = input_embeds
        new_keys = []
        new_vals = []

        nh   = NUM_HEADS
        nkv  = NUM_KV_HEADS
        nkvg = NUM_HEADS // NUM_KV_HEADS
        hd   = HEAD_DIM
        scaling = HEAD_DIM ** -0.5

        for i, layer in enumerate(self.layers):
            attn = layer.self_attn

            residual = hidden
            x        = layer.input_layernorm(hidden)

            # QKV -- (B, heads, 1, hd)
            q = attn.q_norm(attn.q_proj(x).view(B, 1, nh,  hd).transpose(1, 2))
            k = attn.k_norm(attn.k_proj(x).view(B, 1, nkv, hd).transpose(1, 2))
            v =             attn.v_proj(x).view(B, 1, nkv, hd).transpose(1, 2)

            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Static KV cache write via scatter
            # scatter(dim, index, src): index shape == src shape == (B,nkv,1,hd)
            # output shape == self (cache buffer) == (B,nkv,T,hd)
            # Maps to ONNX ScatterElements -- QNN HTP native op
            k_new = past_keys[i].scatter(2, idx, k)
            v_new = past_values[i].scatter(2, idx, v)
            new_keys.append(k_new)
            new_vals.append(v_new)

            # GQA
            k_e = k_new.repeat_interleave(nkvg, dim=1)
            v_e = v_new.repeat_interleave(nkvg, dim=1)

            # Attention over full static KV buffer -- (B, nh, 1, T)
            w = torch.matmul(q, k_e.transpose(-2, -1)) * scaling
            w = w + attention_mask
            w = torch.softmax(w.float(), dim=-1).to(q.dtype)
            o = torch.matmul(w, v_e)

            o      = o.transpose(1, 2).reshape(B, 1, nh * hd)
            o      = attn.o_proj(o)
            hidden = residual + o

            residual = hidden
            hidden   = layer.post_attention_layernorm(hidden)
            hidden   = layer.mlp(hidden)
            hidden   = residual + hidden

        hidden = self.norm(hidden)

        # Returns: (hidden, k_0_new, ..., k_27_new, v_0_new, ..., v_27_new)
        return tuple([hidden] + new_keys + new_vals)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_text_model(model):
    inner = model.model
    return inner.language_model if hasattr(inner, "language_model") else inner


def _check_size(path, min_mb, label):
    mb = os.path.getsize(path) / 1024 / 1024
    if min_mb > 0 and mb < min_mb:
        print(f"  *** WARNING {label}: {mb:.1f} MB -- expected >= {min_mb:.0f} MB ***")
    else:
        print(f"  {label}: {mb:.0f} MB  OK")


# ── Export functions ─────────────────────────────────────────────────────────

def export_embed_tokens(model, cfg, output_dir):
    print("\n[embed_tokens]")
    wrapper  = EmbedTokensForONNX(model.get_input_embeddings()).half().eval()
    dummy    = torch.randint(0, cfg.text_config.vocab_size, (BATCH, PREFILL_SEQ_LEN))
    out_path = os.path.join(output_dir, "common", "vlm_embed_tokens.onnx")
    with torch.no_grad():
        torch.onnx.export(wrapper, (dummy,), out_path,
            input_names=["input_ids"], output_names=["embeddings"],
            opset_version=OPSET_VERSION, do_constant_folding=True)
    _check_size(out_path, 0, "embed_tokens")


def export_lm_head(model, cfg, output_dir):
    print("\n[lm_head]")
    wrapper  = LMHeadForONNX(model.lm_head).half().eval()
    dummy    = torch.randn(BATCH, 1, HIDDEN_SIZE, dtype=torch.float16)
    out_path = os.path.join(output_dir, "common", "vlm_lm_head.onnx")
    with torch.no_grad():
        torch.onnx.export(wrapper, (dummy,), out_path,
            input_names=["hidden_states"], output_names=["logits"],
            opset_version=OPSET_VERSION, do_constant_folding=True)
    _check_size(out_path, 0, "lm_head")


def export_vision_encoder(model, cfg, output_dir):
    print("\n[vision_encoder]")
    wrapper      = VisionEncoderForONNX(model.model.visual).float().eval()
    dummy_pixels = torch.randn(TOTAL_PATCHES, PATCH_DIM, dtype=torch.float32)
    dummy_grid   = torch.tensor([[GRID_T, GRID_H, GRID_W]] * NUM_IMAGES, dtype=torch.long)
    out_path     = os.path.join(output_dir, "common", "vlm_vision_encoder.onnx")
    with torch.no_grad():
        out = wrapper(dummy_pixels, dummy_grid)
        print(f"  image_embeds: {out.shape}")
        torch.onnx.export(wrapper, (dummy_pixels, dummy_grid), out_path,
            input_names=["pixel_values", "grid_thw"],
            output_names=["image_embeds"],
            opset_version=OPSET_VERSION, do_constant_folding=True)
    _check_size(out_path, 0, "vision_encoder")


def export_decoder_prefill(model, cfg, output_dir):
    print("\n[decoder_prefill]")
    wrapper = DecoderPrefillForONNX(_get_text_model(model)).half().eval()

    dummy_embeds = torch.randn(BATCH, PREFILL_SEQ_LEN, HIDDEN_SIZE, dtype=torch.float16)
    dummy_pos    = (torch.arange(PREFILL_SEQ_LEN)
                    .unsqueeze(0).unsqueeze(0).expand(3, BATCH, -1)
                    .long().contiguous())
    dummy_mask   = torch.ones(BATCH, PREFILL_SEQ_LEN, dtype=torch.float16)
    out_path     = os.path.join(output_dir, "decoder_prefill", "vlm_decoder_prefill.onnx")

    print(f"  inputs_embeds : {dummy_embeds.shape}  {dummy_embeds.dtype}")
    print(f"  position_ids  : {dummy_pos.shape}  {dummy_pos.dtype}")
    print(f"  attention_mask: {dummy_mask.shape}  {dummy_mask.dtype}")

    with torch.no_grad():
        outs = wrapper(dummy_embeds, dummy_pos, dummy_mask)

    assert len(outs) == 1 + 2 * NUM_LAYERS, f"Got {len(outs)} outputs, expected {1+2*NUM_LAYERS}"
    print(f"  -> last_hidden_state : {outs[0].shape}")
    print(f"  -> past_key_0        : {outs[1].shape}  (x{NUM_LAYERS})")
    print(f"  -> past_value_0      : {outs[NUM_LAYERS+1].shape}  (x{NUM_LAYERS})")

    out_names = (["last_hidden_state"]
                 + [f"past_key_{i}"   for i in range(NUM_LAYERS)]
                 + [f"past_value_{i}" for i in range(NUM_LAYERS)])

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_pos, dummy_mask),
            out_path,
            input_names=["inputs_embeds", "position_ids", "attention_mask"],
            output_names=out_names,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
    _check_size(out_path, 500, "decoder_prefill")


def export_decoder_decode(model, cfg, output_dir):
    print("\n[decoder_decode]")
    wrapper = DecoderDecodeForONNX(
        _get_text_model(model),
        num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS,
        max_total_seq=MAX_TOTAL_SEQ, head_dim=HEAD_DIM,
    ).half().eval()

    dummy_embeds    = torch.randn(BATCH, 1, HIDDEN_SIZE, dtype=torch.float16)
    dummy_pos       = (torch.tensor([[[MAX_TOTAL_SEQ - 1]]], dtype=torch.long)
                       .expand(3, BATCH, 1).contiguous())
    dummy_mask      = torch.zeros(BATCH, 1, 1, MAX_TOTAL_SEQ, dtype=torch.float16)
    dummy_write_idx = torch.tensor(MAX_TOTAL_SEQ - 1, dtype=torch.long)
    dummy_ks = [torch.randn(BATCH, NUM_KV_HEADS, MAX_TOTAL_SEQ, HEAD_DIM,
                            dtype=torch.float16) for _ in range(NUM_LAYERS)]
    dummy_vs = [torch.randn(BATCH, NUM_KV_HEADS, MAX_TOTAL_SEQ, HEAD_DIM,
                            dtype=torch.float16) for _ in range(NUM_LAYERS)]
    out_path = os.path.join(output_dir, "decoder_decode", "vlm_decoder_decode.onnx")

    print(f"  input_embeds  : {dummy_embeds.shape}  {dummy_embeds.dtype}")
    print(f"  attention_mask: {dummy_mask.shape}  {dummy_mask.dtype}")
    print(f"  past_key_i    : {dummy_ks[0].shape}  (x{NUM_LAYERS})")
    print(f"  write_index   : {dummy_write_idx.shape}  {dummy_write_idx.dtype}  (scalar)")

    args = (dummy_embeds, dummy_pos, dummy_mask, dummy_write_idx, *dummy_ks, *dummy_vs)

    with torch.no_grad():
        outs = wrapper(*args)

    assert len(outs) == 1 + 2 * NUM_LAYERS, f"Got {len(outs)} outputs, expected {1+2*NUM_LAYERS}"
    print(f"  -> hidden_state      : {outs[0].shape}")
    print(f"  -> new_past_key_0    : {outs[1].shape}  (x{NUM_LAYERS})")
    print(f"  -> new_past_value_0  : {outs[NUM_LAYERS+1].shape}  (x{NUM_LAYERS})")

    in_names  = (["input_embeds", "position_ids", "attention_mask", "write_index"]
                 + [f"past_key_{i}"   for i in range(NUM_LAYERS)]
                 + [f"past_value_{i}" for i in range(NUM_LAYERS)])
    out_names = (["hidden_state"]
                 + [f"new_past_key_{i}"   for i in range(NUM_LAYERS)]
                 + [f"new_past_value_{i}" for i in range(NUM_LAYERS)])

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            out_path,
            input_names=in_names,
            output_names=out_names,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
    _check_size(out_path, 500, "decoder_decode")


# ── Main ─────────────────────────────────────────────────────────────────────

ALL_EXPORTS = {
    "embed_tokens":    export_embed_tokens,
    "lm_head":         export_lm_head,
    "vision_encoder":  export_vision_encoder,
    "decoder_prefill": export_decoder_prefill,
    "decoder_decode":  export_decoder_decode,
}


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3-VL-2B-Instruct to static-shape ONNX for Qualcomm NPU")
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", default="/data/users/adhi/qwen_npu/onnx_models")
    parser.add_argument("--export-only", nargs="+", choices=list(ALL_EXPORTS.keys()))
    args = parser.parse_args()

    for d in ["common", "decoder_prefill", "decoder_decode"]:
        os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

    print(f"Loading {args.model_path}  (fp16 CPU, ~8 GB RAM)")
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="cpu", attn_implementation="eager",
    ).half().eval()
    tc = model.config.text_config
    print(f"  hidden={tc.hidden_size}  layers={tc.num_hidden_layers}"
          f"  kv_heads={tc.num_key_value_heads}"
          f"  head_dim={tc.hidden_size//tc.num_attention_heads}")

    for name in (args.export_only or list(ALL_EXPORTS.keys())):
        try:
            ALL_EXPORTS[name](model, model.config, args.output_dir)
        except Exception as e:
            import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("Export complete!")
    print(f"Output dir: {args.output_dir}")
    print("\nStatic cache I/O layout:")
    print("  decoder_prefill  OUT: last_hidden_state, past_key_{{0..27}}, past_value_{{0..27}}")
    print("  decoder_decode    IN: input_embeds, position_ids, attention_mask, write_index,")
    print("                        past_key_{{0..27}}, past_value_{{0..27}}")
    print("                   OUT: hidden_state, new_past_key_{{0..27}}, new_past_value_{{0..27}}")
    print("="*60)


if __name__ == "__main__":
    main()