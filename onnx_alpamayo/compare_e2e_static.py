# SPDX-License-Identifier: Apache-2.0
"""
End-to-end comparison: Original Alpamayo PyTorch vs Static FP16 ONNX pipeline.

Handles:
  - fp16 model I/O (inputs/outputs are float16)
  - Static KV cache for decoder_decode (padded to 3262)
  - Static shapes throughout (no dynamic axes)

Usage:
    python compare_e2e_static.py
    python compare_e2e_static.py --onnx-dir ./onnx_models_static_fp16
"""

import sys
import copy
import argparse
import torch
import numpy as np
import onnxruntime as ort
import einops
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from alpamayo_r1.models.token_utils import (
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)

# Static shape constants (must match export_static_fp16.py)
PREFILL_SEQ_LEN = 3006
MAX_TOTAL_SEQ = 3262  # 3006 + 256
ACTION_LEN = 64
NUM_LAYERS = 36
NUM_KV_HEADS = 8
HEAD_DIM = 128


class OnnxSession:
    """Generic ONNX Runtime session wrapper."""

    def __init__(self, onnx_path, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        # Use basic optimizations only — advanced fusions (SimplifiedLayerNormFusion)
        # conflict with fp16 converter's inserted cast nodes
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        self.output_names = [out.name for out in self.sess.get_outputs()]

    def run(self, feeds):
        return self.sess.run(None, feeds)


class StaticOnnxVLMComponents:
    """Holds all ONNX sessions for the static fp16 VLM pipeline."""

    def __init__(self, onnx_base, providers=None):
        print("Loading static fp16 ONNX sessions...")

        common_dir = str(Path(onnx_base) / "common")
        prefill_dir = str(Path(onnx_base) / "decoder_prefill")
        decode_dir = str(Path(onnx_base) / "decoder_decode")
        expert_dir = str(Path(onnx_base) / "expert")

        self.embed_tokens = OnnxSession(
            str(Path(common_dir) / "vlm_embed_tokens.onnx"), providers=providers
        )
        print("  embed_tokens loaded")
        self.vision_encoder = OnnxSession(
            str(Path(common_dir) / "vlm_vision_encoder.onnx"), providers=providers
        )
        print("  vision_encoder loaded")
        self.decoder_prefill = OnnxSession(
            str(Path(prefill_dir) / "vlm_decoder_prefill.onnx"), providers=providers
        )
        print("  decoder_prefill loaded")
        self.decoder_decode = OnnxSession(
            str(Path(decode_dir) / "vlm_decoder_decode.onnx"), providers=providers
        )
        print("  decoder_decode loaded")
        self.lm_head = OnnxSession(
            str(Path(common_dir) / "vlm_lm_head.onnx"), providers=providers
        )
        print("  lm_head loaded")
        self.diffusion_step = OnnxSession(
            str(Path(expert_dir) / "diffusion_step_kvcache.onnx"), providers=providers
        )
        print("  diffusion_step loaded")
        print("All static fp16 ONNX sessions loaded.")


def get_rope_index_python(
    input_ids,
    image_grid_thw,
    attention_mask,
    image_token_id,
    vision_start_token_id,
    spatial_merge_size,
):
    """Python reimplementation of Qwen3VLModel.get_rope_index()."""
    B, seq_len = input_ids.shape
    device = input_ids.device

    if image_grid_thw is not None:
        position_ids = torch.ones(3, B, seq_len, dtype=torch.long, device=device)
        image_index = 0
        mrope_position_deltas = []

        for i in range(B):
            ids = input_ids[i]
            if attention_mask is not None:
                ids = ids[attention_mask[i] == 1]

            input_tokens = ids.tolist()
            vision_start_indices = (ids == vision_start_token_id).nonzero(
                as_tuple=True
            )[0]
            image_nums = 0
            for vi in vision_start_indices:
                if vi + 1 < len(ids) and ids[vi + 1] == image_token_id:
                    image_nums += 1

            llm_pos_ids_list = []
            st = 0
            remain_images = image_nums

            for _ in range(image_nums):
                if image_token_id in input_tokens[st:] and remain_images > 0:
                    ed = input_tokens.index(image_token_id, st)
                else:
                    break

                t = image_grid_thw[image_index][0].item()
                h = image_grid_thw[image_index][1].item()
                w = image_grid_thw[image_index][2].item()
                image_index += 1
                remain_images -= 1

                llm_grid_t = t
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size

                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                    + st_idx
                )

                t_index = (
                    torch.arange(llm_grid_t, device=device)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h, device=device)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w, device=device)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                    + st_idx
                )

            if llm_pos_ids_list:
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                if attention_mask is not None:
                    position_ids[:, i, attention_mask[i] == 1] = llm_positions.to(
                        device
                    )
                else:
                    position_ids[:, i, :] = llm_positions.to(device)
                mrope_position_deltas.append(llm_positions.max() + 1 - seq_len)

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=device, dtype=torch.long
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            max_pos = position_ids.max(0)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_pos + 1 - seq_len
        else:
            position_ids = (
                torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, B, -1)
            )
            mrope_position_deltas = torch.zeros(B, 1, device=device, dtype=torch.long)
        return position_ids, mrope_position_deltas


def top_p_sampling(logits, temperature=0.6, top_p=0.98, greedy=False):
    """Apply temperature and top-p (nucleus) sampling, or greedy (argmax) decoding."""
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
    return next_token.view(logits.shape[0], -1)


def to_fp16_np(tensor):
    """Convert torch tensor to fp16 numpy array for ONNX input."""
    return tensor.detach().float().cpu().numpy().astype(np.float16)


def from_fp16_np(np_array, device="cpu"):
    """Convert fp16 numpy output to float32 torch tensor."""
    return torch.from_numpy(np_array.astype(np.float32)).to(device=device)


def pad_kv_cache(past_keys, past_values, target_seq_len):
    """
    Pad KV cache to target_seq_len along the sequence dimension (dim=3).

    Args:
        past_keys: (num_layers, B, num_kv_heads, actual_seq_len, head_dim)
        past_values: same shape
        target_seq_len: target sequence length to pad to

    Returns:
        padded_keys, padded_values: (num_layers, B, num_kv_heads, target_seq_len, head_dim)
    """
    actual_seq = past_keys.shape[3]
    if actual_seq >= target_seq_len:
        return (
            past_keys[:, :, :, :target_seq_len, :],
            past_values[:, :, :, :target_seq_len, :],
        )

    pad_len = target_seq_len - actual_seq
    pad_shape = list(past_keys.shape)
    pad_shape[3] = pad_len

    pad_k = torch.zeros(pad_shape, dtype=past_keys.dtype, device=past_keys.device)
    pad_v = torch.zeros(pad_shape, dtype=past_values.dtype, device=past_values.device)

    return torch.cat([past_keys, pad_k], dim=3), torch.cat([past_values, pad_v], dim=3)


def static_fp16_onnx_inference(
    model,
    data,
    onnx_components,
    top_p=0.98,
    temperature=0.6,
    max_generation_length=256,
    return_extra=True,
    greedy=False,
):
    """
    Full static fp16 ONNX VLM + diffusion inference pipeline.

    Key differences from dynamic inference:
    - All ONNX I/O is float16
    - decoder_decode uses fixed KV cache size (MAX_TOTAL_SEQ=3262)
    - KV cache is padded with zeros, attention mask controls valid positions
    """
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    B = ego_history_xyz.shape[0]
    tokenized_data = copy.deepcopy(data["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")

    # Fuse trajectory tokens (PyTorch)
    traj_data_vlm = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)
    device = input_ids.device

    pixel_values = tokenized_data.get("pixel_values", None)
    image_grid_thw = tokenized_data.get("image_grid_thw", None)

    # 1. Embed tokens (ONNX, fp16)
    print("  Step 1: Embed tokens...")
    input_ids_np = input_ids.detach().long().cpu().numpy()
    embed_result = onnx_components.embed_tokens.run({"input_ids": input_ids_np})
    inputs_embeds = from_fp16_np(embed_result[0], device=device)

    # 2. Vision encoder (ONNX, fp16) + merge into embeddings
    print("  Step 2: Vision encoder...")
    deepstack_embeds = None
    visual_pos_mask = None

    if pixel_values is not None:
        pixel_np = to_fp16_np(pixel_values)

        # grid_thw was constant-folded away in fp16 conversion (static shapes)
        vision_feeds = {"pixel_values": pixel_np}
        vision_result = onnx_components.vision_encoder.run(vision_feeds)
        image_embeds = from_fp16_np(vision_result[0], device=device)
        ds0 = from_fp16_np(vision_result[1], device=device)
        ds1 = from_fp16_np(vision_result[2], device=device)
        ds2 = from_fp16_np(vision_result[3], device=device)
        deepstack_embeds = [ds0, ds1, ds2]

        # Merge visual embeddings into text embeddings
        image_token_id = model.vlm.config.image_token_id
        special_image_mask = input_ids == image_token_id
        special_image_mask_3d = special_image_mask.unsqueeze(-1).expand_as(
            inputs_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            special_image_mask_3d, image_embeds
        )
        visual_pos_mask = special_image_mask

    # 3. Compute position_ids (Python)
    print("  Step 3: Compute position IDs...")
    image_token_id = model.vlm.config.image_token_id
    vision_start_token_id = model.vlm.config.vision_start_token_id
    spatial_merge_size = model.vlm.config.vision_config.spatial_merge_size

    attention_mask_1d = torch.ones(
        B, input_ids.shape[1], dtype=torch.long, device=device
    )

    position_ids, rope_deltas = get_rope_index_python(
        input_ids,
        image_grid_thw,
        attention_mask_1d,
        image_token_id,
        vision_start_token_id,
        spatial_merge_size,
    )

    # 4. Decoder prefill (ONNX, fp16)
    print("  Step 4: Decoder prefill...")
    seq_len = inputs_embeds.shape[1]
    hidden_size = inputs_embeds.shape[2]

    # Build pre-expanded DeepStack tensors
    if deepstack_embeds is not None and visual_pos_mask is not None:
        ds0, ds1, ds2 = deepstack_embeds
        ds_full_0 = torch.zeros(
            B, seq_len, hidden_size, dtype=torch.float32, device=device
        )
        ds_full_1 = torch.zeros(
            B, seq_len, hidden_size, dtype=torch.float32, device=device
        )
        ds_full_2 = torch.zeros(
            B, seq_len, hidden_size, dtype=torch.float32, device=device
        )
        for b in range(B):
            vis_indices = visual_pos_mask[b].nonzero(as_tuple=True)[0]
            num_vis = min(vis_indices.shape[0], ds0.shape[0])
            if num_vis > 0:
                ds_full_0[b, vis_indices[:num_vis], :] = ds0[:num_vis].float()
                ds_full_1[b, vis_indices[:num_vis], :] = ds1[:num_vis].float()
                ds_full_2[b, vis_indices[:num_vis], :] = ds2[:num_vis].float()
    else:
        ds_full_0 = torch.zeros(
            B, seq_len, hidden_size, dtype=torch.float32, device=device
        )
        ds_full_1 = torch.zeros(
            B, seq_len, hidden_size, dtype=torch.float32, device=device
        )
        ds_full_2 = torch.zeros(
            B, seq_len, hidden_size, dtype=torch.float32, device=device
        )

    prefill_feeds = {
        "inputs_embeds": to_fp16_np(inputs_embeds),
        "position_ids": position_ids.detach().long().cpu().numpy(),
        "attention_mask": to_fp16_np(attention_mask_1d.float()),
        "deepstack_full_0": to_fp16_np(ds_full_0),
        "deepstack_full_1": to_fp16_np(ds_full_1),
        "deepstack_full_2": to_fp16_np(ds_full_2),
    }

    prefill_result = onnx_components.decoder_prefill.run(prefill_feeds)
    # Keep KV cache as numpy fp16 throughout — avoids 480MB torch↔numpy conversion per step
    last_hidden_np = prefill_result[0]   # (1, 3006, 4096) fp16
    real_past_keys_np = prefill_result[1]  # (36, 1, 8, 3006, 128) fp16
    real_past_values_np = prefill_result[2]

    print(f"    Prefill KV cache shape: {real_past_keys_np.shape}")

    # 5. LM head -> first token logits
    print("  Step 5: LM head (first token)...")
    last_pos_hidden_np = last_hidden_np[:, -1:, :]
    logits_result = onnx_components.lm_head.run({"hidden_states": last_pos_hidden_np})
    logits = from_fp16_np(logits_result[0], device=device)

    # 6. Autoregressive decode loop with STATIC KV cache
    print("  Step 6: Autoregressive decode loop (static KV cache)...")
    eos_token_id = model.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )
    pad_token_id = model.tokenizer.pad_token_id

    generated_ids = [input_ids]
    actual_past_len = real_past_keys_np.shape[3]  # starts at PREFILL_SEQ_LEN (3006)

    import time
    for step in range(max_generation_length):
        t0 = time.time()
        # Sample next token
        next_token = top_p_sampling(
            logits[:, -1, :], temperature=temperature, top_p=top_p, greedy=greedy
        )
        generated_ids.append(next_token)

        if (next_token == eos_token_id).all():
            print(f"    EOS at step {step}")
            break

        # Embed next token (PyTorch — ONNX embed_tokens has static shape 3006)
        with torch.no_grad():
            next_embed_np = model.vlm.model.language_model.embed_tokens(
                next_token.to(device)
            ).float().cpu().numpy().astype(np.float16)

        # Position IDs for this decode step
        cur_pos = actual_past_len + 1
        delta = rope_deltas
        decode_pos = torch.tensor([[[cur_pos]]], dtype=torch.long, device=device)
        decode_pos = decode_pos + delta.unsqueeze(0)
        decode_position_ids_np = decode_pos.expand(3, B, 1).detach().cpu().numpy()

        # Pad KV cache to MAX_TOTAL_SEQ (3262) using numpy (no torch conversion)
        actual_seq = real_past_keys_np.shape[3]
        if actual_seq < MAX_TOTAL_SEQ:
            pad_len = MAX_TOTAL_SEQ - actual_seq
            pad_shape_k = list(real_past_keys_np.shape)
            pad_shape_k[3] = pad_len
            padded_keys_np = np.concatenate(
                [real_past_keys_np, np.zeros(pad_shape_k, dtype=np.float16)], axis=3
            )
            padded_values_np = np.concatenate(
                [real_past_values_np, np.zeros(pad_shape_k, dtype=np.float16)], axis=3
            )
        else:
            padded_keys_np = real_past_keys_np[:, :, :, :MAX_TOTAL_SEQ, :]
            padded_values_np = real_past_values_np[:, :, :, :MAX_TOTAL_SEQ, :]

        # Attention mask in fp16: 0 for valid, -65504 (fp16 min) for padded.
        # Layout of the 3263 attention positions after the model appends the new token KV:
        #   [0 .. actual_past_len-1]  = real past KV     → unmask
        #   [actual_past_len .. 3261] = zero-padded KV   → mask
        #   [3262 = MAX_TOTAL_SEQ]    = new token's own KV (self-attention) → unmask
        total_mask_len = MAX_TOTAL_SEQ + 1  # 3263
        decode_attn_mask_np = np.full(
            (B, 1, 1, total_mask_len), np.finfo(np.float16).min, dtype=np.float16
        )
        decode_attn_mask_np[:, :, :, :actual_past_len] = 0.0       # real past (no +1)
        decode_attn_mask_np[:, :, :, MAX_TOTAL_SEQ] = 0.0          # new token self-attn

        # Run decode step — all numpy fp16, no conversion overhead
        decode_feeds = {
            "input_embeds": next_embed_np,
            "position_ids": decode_position_ids_np,
            "attention_mask": decode_attn_mask_np,
            "past_keys": padded_keys_np,
            "past_values": padded_values_np,
        }

        decode_result = onnx_components.decoder_decode.run(decode_feeds)
        hidden_state_np = decode_result[0]
        new_past_keys_np = decode_result[1]   # (36, 1, 8, 3263, 128) fp16
        new_past_values_np = decode_result[2]

        # Append new token KV (at position MAX_TOTAL_SEQ) to real cache
        real_past_keys_np = np.concatenate(
            [real_past_keys_np, new_past_keys_np[:, :, :, MAX_TOTAL_SEQ:MAX_TOTAL_SEQ+1, :]], axis=3
        )
        real_past_values_np = np.concatenate(
            [real_past_values_np, new_past_values_np[:, :, :, MAX_TOTAL_SEQ:MAX_TOTAL_SEQ+1, :]], axis=3
        )
        actual_past_len += 1

        elapsed = time.time() - t0
        print(f"    Step {step}, KV len: {actual_past_len}, time: {elapsed:.1f}s")

        # LM head (fp16)
        logits_result = onnx_components.lm_head.run({"hidden_states": hidden_state_np})
        logits = from_fp16_np(logits_result[0], device=device)

    # Concatenate all generated token IDs
    all_sequences = torch.cat(generated_ids, dim=1)
    all_sequences = replace_padding_after_eos(
        token_ids=all_sequences,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    # Free VLM ONNX sessions AND PyTorch VLM weights before diffusion
    import gc
    del onnx_components.embed_tokens
    del onnx_components.vision_encoder
    del onnx_components.decoder_prefill
    del onnx_components.decoder_decode
    del onnx_components.lm_head
    del generated_ids
    # Delete the heavy VLM transformer weights (~20 GB on CPU).
    # We only need model.action_space, model.diffusion, and model.tokenizer below.
    if hasattr(model, "vlm"):
        del model.vlm
    gc.collect()
    print("  Freed VLM ONNX sessions + PyTorch VLM weights to reclaim RAM for diffusion.")

    # 7. Prepare for diffusion
    print("  Step 7: Diffusion loop...")
    b_star = all_sequences.shape[0]
    traj_future_start_mask = all_sequences == eos_token_id
    has_traj_future_start = traj_future_start_mask.any(dim=1)
    traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
    last_token_positions = torch.full(
        (b_star,), all_sequences.shape[1] - 1, device=device
    )
    valid_token_pos_id = torch.where(
        has_traj_future_start, traj_future_start_positions, last_token_positions
    )
    offset = valid_token_pos_id + 1

    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    diff_position_ids = torch.arange(n_diffusion_tokens, device=device)
    diff_position_ids = einops.repeat(diff_position_ids, "l -> 3 b l", b=b_star).clone()
    delta_val = rope_deltas + offset[:, None]
    diff_position_ids += delta_val.to(diff_position_ids.device)

    # Pad KV cache to MAX_TOTAL_SEQ for diffusion step (numpy fp16)
    actual_seq = real_past_keys_np.shape[3]
    if actual_seq < MAX_TOTAL_SEQ:
        pad_len = MAX_TOTAL_SEQ - actual_seq
        pad_shape = list(real_past_keys_np.shape)
        pad_shape[3] = pad_len
        diff_past_keys_np = np.concatenate(
            [real_past_keys_np, np.zeros(pad_shape, dtype=np.float16)], axis=3
        )
        diff_past_values_np = np.concatenate(
            [real_past_values_np, np.zeros(pad_shape, dtype=np.float16)], axis=3
        )
    else:
        diff_past_keys_np = real_past_keys_np[:, :, :, :MAX_TOTAL_SEQ, :]
        diff_past_values_np = real_past_values_np[:, :, :, :MAX_TOTAL_SEQ, :]

    # Diffusion attention mask: (B, 1, 64, 3326)
    # Layout of the 3326 attention positions:
    #   [0 .. actual_seq-1]          = real past text KV  → unmask
    #   [actual_seq .. 3261]         = zero-padded text KV → mask
    #   [3262=MAX_TOTAL_SEQ .. 3325] = diffusion tokens' own KV (full self-attn) → unmask
    diff_total_len = MAX_TOTAL_SEQ + ACTION_LEN  # 3326
    diff_attention_mask_np = np.full(
        (b_star, 1, n_diffusion_tokens, diff_total_len), np.finfo(np.float16).min, dtype=np.float16
    )
    diff_attention_mask_np[:, :, :, :actual_seq] = 0.0          # real past text KV
    diff_attention_mask_np[:, :, :, MAX_TOTAL_SEQ:] = 0.0       # diffusion self-attention

    # 8. Diffusion Euler loop (ONNX, fp16)
    x_dims = model.action_space.get_action_space_dims()
    # Use CPU randn with fixed seed for reproducibility across runs.
    # (caller should set torch.manual_seed before calling this function)
    x = torch.randn(B, *x_dims, device="cpu").to(device)

    time_steps = torch.linspace(
        0.0, 1.0, model.diffusion.num_inference_steps + 1, device=device
    )
    n_dim = len(x_dims)

    for i in range(model.diffusion.num_inference_steps):
        dt = time_steps[i + 1] - time_steps[i]
        dt = dt.view(1, *[1] * n_dim).expand(B, *[1] * n_dim)
        t_start = time_steps[i].view(1, *[1] * n_dim).expand(B, *[1] * n_dim)

        t_onnx = (
            t_start[:, :1].view(B, 1) if t_start.dim() > 1 else t_start.unsqueeze(-1)
        )

        diff_feeds = {
            "noisy_actions": to_fp16_np(x),
            "timesteps": to_fp16_np(t_onnx),
            "position_ids": diff_position_ids.detach().long().cpu().numpy(),
            "attention_mask": diff_attention_mask_np,
            "past_keys": diff_past_keys_np,
            "past_values": diff_past_values_np,
        }

        result = onnx_components.diffusion_step.run(diff_feeds)
        v = from_fp16_np(result[0], device=device)
        v = v.view(B, *x_dims)
        x = x + dt * v

    sampled_action = x

    # 9. Action to trajectory (PyTorch)
    print("  Step 8: Action to trajectory...")
    pred_xyz, pred_rot = model.action_space.action_to_traj(
        sampled_action, ego_history_xyz[:, -1], ego_history_rot[:, -1]
    )

    pred_xyz = pred_xyz.unsqueeze(1).unsqueeze(1)
    pred_rot = pred_rot.unsqueeze(1).unsqueeze(1)

    result = (pred_xyz, pred_rot)
    if return_extra:
        extra = extract_text_tokens(model.tokenizer, all_sequences)
        result = (pred_xyz, pred_rot, extra)

    return result


def rotate_90cc(xy):
    """Rotate (x, y) trajectory 90° CCW for display: (x,y) → (-y, x)."""
    return np.stack([-xy[1], xy[0]], axis=0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Compare PyTorch vs Static FP16 ONNX inference"
    )
    parser.add_argument("--onnx-dir", type=str, default="./onnx_models_static_fp16")
    parser.add_argument(
        "--clip-id", type=str, default="030c760c-ae38-49aa-9ad8-f5650a545d26"
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip PyTorch inference, only run ONNX",
    )
    parser.add_argument(
        "--save-dir", type=str, default="/data/users/adhi",
        help="Directory to save comparison plot and results",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature (default 0.6, matches test_inference.py).",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.98,
        help="Top-p nucleus sampling threshold (default 0.98).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("END-TO-END COMPARISON: PyTorch vs Static FP16 ONNX")
    print("=" * 80)

    # --- Load dataset ---
    print(f"\nLoading dataset for clip_id: {args.clip_id}...")
    data = load_physical_aiavdataset(args.clip_id, t0_us=5_100_000)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    # --- Load model ---
    print("Loading PyTorch model (bfloat16, CUDA)...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()

    # Save a copy of model_inputs for ONNX (PyTorch may consume it in-place)
    onnx_model_inputs = copy.deepcopy(model_inputs)

    # ==========================================
    # 1. Run ORIGINAL PyTorch model (exact test_inference.py)
    # ==========================================
    pt_min_ade = None
    pt_coc = "N/A"
    pt_meta = "N/A"
    pt_pred_xyz = None

    if not args.skip_pytorch:
        print("\n" + "-" * 60)
        print(f"Running ORIGINAL PyTorch model (seed=42, T={args.temperature}, top_p={args.top_p})...")
        print("-" * 60)

        torch.cuda.manual_seed_all(42)
        torch.manual_seed(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pt_pred_xyz, _, pt_extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=args.top_p,
                temperature=args.temperature,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )

        pt_coc = pt_extra.get("cot", [["N/A"]])[0][0][0] if "cot" in pt_extra else "N/A"
        pt_meta = (
            pt_extra.get("meta_action", [["N/A"]])[0][0][0]
            if "meta_action" in pt_extra
            else "N/A"
        )
        # minADE (matching test_inference.py)
        pt_pred_xy_all = pt_pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        pt_min_ade = np.linalg.norm(pt_pred_xy_all - gt_xy[None, ...], axis=1).mean(-1).min()

        print(f"\n  CoC: {pt_coc[:200]}...")
        print(f"  Meta-action: {pt_meta}")
        print(f"  minADE: {pt_min_ade:.4f} meters")

    # ==========================================
    # 2. Run Static FP16 ONNX pipeline
    # ==========================================
    print("\n" + "-" * 60)
    print("Running STATIC FP16 ONNX pipeline...")
    print("-" * 60)

    # Move PyTorch model to CPU to free GPU memory
    print("  Moving PyTorch model to CPU...")
    model = model.cpu()
    onnx_model_inputs = helper.to_device(onnx_model_inputs, "cpu")
    torch.cuda.empty_cache()

    onnx_components = StaticOnnxVLMComponents(
        onnx_base=args.onnx_dir,
        providers=["CPUExecutionProvider"],
    )

    torch.manual_seed(42)  # seed CPU RNG for diffusion randn (matches PyTorch run)
    with torch.no_grad():
        onnx_pred_xyz, _, onnx_extra = static_fp16_onnx_inference(
            model=model,
            data=onnx_model_inputs,
            onnx_components=onnx_components,
            top_p=args.top_p,
            temperature=args.temperature,
            max_generation_length=256,
            return_extra=True,
        )

    onnx_coc = onnx_extra.get("cot", ["N/A"])[0] if "cot" in onnx_extra else "N/A"
    onnx_meta = (
        onnx_extra.get("meta_action", ["N/A"])[0]
        if "meta_action" in onnx_extra
        else "N/A"
    )
    # minADE
    onnx_pred_xy_all = onnx_pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    onnx_min_ade = np.linalg.norm(onnx_pred_xy_all - gt_xy[None, ...], axis=1).mean(-1).min()

    print(f"\n  CoC: {onnx_coc[:200]}...")
    print(f"  Meta-action: {onnx_meta}")
    print(f"  minADE: {onnx_min_ade:.4f} meters")

    # ==========================================
    # 3. Comparison summary
    # ==========================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'':>25s} {'PyTorch':>15s}  {'Static FP16 ONNX':>18s}")
    if pt_min_ade is not None:
        print(f"{'minADE (meters)':>25s} {pt_min_ade:>15.4f}  {onnx_min_ade:>18.4f}")
    else:
        print(f"{'minADE (meters)':>25s} {'(skipped)':>15s}  {onnx_min_ade:>18.4f}")
    print()

    import textwrap
    def _fmt_coc(text):
        if not text or text == "N/A":
            return "  (none generated)"
        return "\n".join("  " + line for line in textwrap.wrap(text.strip(), width=74))

    if not args.skip_pytorch:
        print("--- PyTorch CoC ---")
        print(_fmt_coc(pt_coc))
        print()

    print("--- Static FP16 ONNX CoC ---")
    print(_fmt_coc(onnx_coc))

    # ==========================================
    # 4. Save trajectory plot
    # ==========================================
    import os
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"compare_{args.clip_id[:8]}.png")

    fig, ax = plt.subplots(figsize=(8, 8))

    gt_rot = rotate_90cc(gt_xy)
    ax.plot(*gt_rot, "r-o", linewidth=2, markersize=4, label="Ground Truth", zorder=3)

    # ONNX trajectories
    for i in range(onnx_pred_xy_all.shape[0]):
        xy_rot = rotate_90cc(onnx_pred_xy_all[i])
        ax.plot(*xy_rot, "b-", linewidth=1.5, alpha=0.8,
                label="Static FP16 ONNX" if i == 0 else "_nolegend_")

    # PyTorch trajectories (if available)
    if pt_pred_xyz is not None:
        pt_pred_xy_all_np = pt_pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        for i in range(pt_pred_xy_all_np.shape[0]):
            xy_rot = rotate_90cc(pt_pred_xy_all_np[i])
            ax.plot(*xy_rot, "g--", linewidth=1.5, alpha=0.8,
                    label="PyTorch (bf16)" if i == 0 else "_nolegend_")

    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    ax.set_title(
        f"Trajectory Comparison [T={args.temperature}, top_p={args.top_p}]\nClip: {args.clip_id[:8]}...\n"
        f"ONNX minADE: {onnx_min_ade:.3f}m"
        + (f"  |  PyTorch minADE: {pt_min_ade:.3f}m" if pt_min_ade is not None else "")
    )
    ax.legend(loc="upper left")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Add CoC text boxes (wrapped)
    onnx_coc_wrapped = "\n".join(textwrap.wrap(onnx_coc.strip()[:200], width=55)) if onnx_coc and onnx_coc != "N/A" else "(none)"
    coc_text = f"ONNX CoC:\n{onnx_coc_wrapped}"
    if not args.skip_pytorch and pt_coc and pt_coc != "N/A":
        pt_coc_wrapped = "\n".join(textwrap.wrap(pt_coc.strip()[:200], width=55))
        coc_text = f"PyTorch CoC:\n{pt_coc_wrapped}\n\nONNX CoC:\n{onnx_coc_wrapped}"
    ax.text(0.02, 0.02, coc_text, transform=ax.transAxes, fontsize=6.5,
            verticalalignment="bottom", bbox=dict(boxstyle="round", alpha=0.3, facecolor="white"))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to: {save_path}")

    print("\n" + "=" * 80)
    print("NOTE: Text outputs differ due to stochastic sampling + fp16 precision.")
    print("Both should produce valid CoC text and comparable trajectory predictions.")
    print("=" * 80)


if __name__ == "__main__":
    main()