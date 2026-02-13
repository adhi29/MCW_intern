# SPDX-License-Identifier: Apache-2.0
"""
Full ONNX VLM inference pipeline.

Chains all ONNX-exported VLM components:
  1. vlm_embed_tokens.onnx — token embedding lookup
  2. vlm_vision_encoder.onnx — image patch processing
  3. vlm_decoder_prefill.onnx — prefill with DeepStack + KV-cache
  4. vlm_decoder_decode.onnx — autoregressive single-token decode
  5. vlm_lm_head.onnx — logits projection
  6. diffusion_step_kvcache.onnx — diffusion denoising step

Python orchestration handles:
  - get_rope_index (position_ids computation)
  - Visual embedding merging (masked_scatter)
  - Token sampling (top-p, temperature)
  - Diffusion Euler loop
  - action_to_traj kinematic conversion
"""

import os
import sys
import copy
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
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from full_onnx_inference import OnnxDiffusionStep, extract_kv_cache_tensors

ONNX_BASE = os.path.join(os.path.dirname(__file__), "onnx_models_vlm_clean")
ONNX_COMMON = os.path.join(ONNX_BASE, "common")
ONNX_PREFILL = os.path.join(ONNX_BASE, "decoder_prefill")
ONNX_DECODE = os.path.join(ONNX_BASE, "decoder_decode")
ONNX_EXPERT = os.path.join(ONNX_BASE, "expert")


class OnnxSession:
    """Generic ONNX Runtime session wrapper."""

    def __init__(self, onnx_path, providers=None):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        self.output_names = [out.name for out in self.sess.get_outputs()]

    def run(self, feeds):
        return self.sess.run(None, feeds)


class OnnxVLMComponents:
    """Holds all ONNX sessions for the VLM pipeline."""

    def __init__(self, onnx_base=None, providers=None):
        print("Loading ONNX VLM sessions...")

        common_dir = os.path.join(onnx_base, "common") if onnx_base else ONNX_COMMON
        prefill_dir = os.path.join(onnx_base, "decoder_prefill") if onnx_base else ONNX_PREFILL
        decode_dir = os.path.join(onnx_base, "decoder_decode") if onnx_base else ONNX_DECODE
        expert_dir = ONNX_EXPERT

        self.embed_tokens = OnnxSession(
            os.path.join(common_dir, "vlm_embed_tokens.onnx"), providers=providers
        )
        self.vision_encoder = OnnxSession(
            os.path.join(common_dir, "vlm_vision_encoder.onnx"), providers=providers
        )
        self.decoder_prefill = OnnxSession(
            os.path.join(prefill_dir, "vlm_decoder_prefill.onnx"), providers=providers
        )
        self.decoder_decode = OnnxSession(
            os.path.join(decode_dir, "vlm_decoder_decode.onnx"), providers=providers
        )
        self.lm_head = OnnxSession(
            os.path.join(common_dir, "vlm_lm_head.onnx"), providers=providers
        )
        self.diffusion_step = OnnxDiffusionStep(
            os.path.join(expert_dir, "diffusion_step_kvcache.onnx"), providers=providers
        )
        print("All ONNX sessions loaded.")


def get_rope_index_python(input_ids, image_grid_thw, attention_mask,
                           image_token_id, vision_start_token_id,
                           spatial_merge_size):
    """
    Python reimplementation of Qwen3VLModel.get_rope_index().

    Computes 3D MRoPE position_ids and rope_deltas from input_ids
    and image grid dimensions.

    Args:
        input_ids: (B, seq_len) int64
        image_grid_thw: (num_images, 3) int64
        attention_mask: (B, seq_len) int/float — 1 for real tokens, 0 for padding
        image_token_id: int — token ID for image placeholder
        vision_start_token_id: int — token ID for vision start marker
        spatial_merge_size: int — spatial merge factor (typically 2)

    Returns:
        position_ids: (3, B, seq_len) int64
        mrope_position_deltas: (B, 1) int64
    """
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

            # Count images
            vision_start_indices = (ids == vision_start_token_id).nonzero(as_tuple=True)[0]
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
                    torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx
                )

                # 3D grid indices for visual tokens
                t_index = torch.arange(llm_grid_t, device=device).view(-1, 1).expand(
                    -1, llm_grid_h * llm_grid_w
                ).flatten()
                h_index = torch.arange(llm_grid_h, device=device).view(1, -1, 1).expand(
                    llm_grid_t, -1, llm_grid_w
                ).flatten()
                w_index = torch.arange(llm_grid_w, device=device).view(1, 1, -1).expand(
                    llm_grid_t, llm_grid_h, -1
                ).flatten()
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # Remaining text after last image
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx
                )

            if llm_pos_ids_list:
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                if attention_mask is not None:
                    position_ids[:, i, attention_mask[i] == 1] = llm_positions.to(device)
                else:
                    position_ids[:, i, :] = llm_positions.to(device)
                mrope_position_deltas.append(llm_positions.max() + 1 - seq_len)

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=device, dtype=torch.long
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        # No visual tokens — simple sequential positions
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            max_pos = position_ids.max(0)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_pos + 1 - seq_len
        else:
            position_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, B, -1)
            mrope_position_deltas = torch.zeros(B, 1, device=device, dtype=torch.long)

        return position_ids, mrope_position_deltas


def top_p_sampling(logits, temperature=0.6, top_p=0.98):
    """Apply temperature and top-p (nucleus) sampling."""
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above top_p
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


def full_vlm_onnx_inference(
    model,
    data,
    onnx_components,
    top_p=0.98,
    temperature=0.6,
    max_generation_length=256,
    return_extra=True,
):
    """
    Full ONNX VLM + diffusion inference pipeline.

    Steps:
    1. Embed tokens (ONNX)
    2. Vision encoder (ONNX)
    3. Merge visual into text embeddings (Python)
    4. Compute position_ids (Python: get_rope_index)
    5. Decoder prefill (ONNX)
    6. LM head (ONNX) → first logits
    7. Autoregressive decode loop (ONNX: decode + lm_head)
    8. Extract KV-cache for expert
    9. Diffusion loop (ONNX: diffusion_step_kvcache)
    10. action_to_traj (PyTorch)
    """
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    B = ego_history_xyz.shape[0]
    tokenized_data = copy.deepcopy(data["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")

    # Fuse trajectory tokens (still PyTorch — simple embedding replacement)
    traj_data_vlm = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)
    device = input_ids.device

    # Get image data
    pixel_values = tokenized_data.get("pixel_values", None)
    image_grid_thw = tokenized_data.get("image_grid_thw", None)

    # 1. Embed tokens (ONNX)
    input_ids_np = input_ids.detach().long().cpu().numpy()
    embed_result = onnx_components.embed_tokens.run({"input_ids": input_ids_np})
    inputs_embeds = torch.from_numpy(embed_result[0]).to(device=device, dtype=torch.float32)

    # 2. Vision encoder (ONNX) + merge into embeddings
    deepstack_embeds = None
    visual_pos_mask = None

    if pixel_values is not None:
        pixel_np = pixel_values.detach().float().cpu().numpy()
        grid_np = image_grid_thw.detach().long().cpu().numpy()

        vision_input_names = onnx_components.vision_encoder.input_names
        vision_feeds = {vision_input_names[0]: pixel_np}
        if len(vision_input_names) > 1:
            vision_feeds[vision_input_names[1]] = grid_np
        vision_result = onnx_components.vision_encoder.run(vision_feeds)
        image_embeds = torch.from_numpy(vision_result[0]).to(device=device, dtype=inputs_embeds.dtype)
        ds0 = torch.from_numpy(vision_result[1]).to(device=device, dtype=inputs_embeds.dtype)
        ds1 = torch.from_numpy(vision_result[2]).to(device=device, dtype=inputs_embeds.dtype)
        ds2 = torch.from_numpy(vision_result[3]).to(device=device, dtype=inputs_embeds.dtype)
        deepstack_embeds = [ds0, ds1, ds2]

        # Create image mask and merge
        image_token_id = model.vlm.config.image_token_id
        special_image_mask = (input_ids == image_token_id)
        special_image_mask_3d = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask_3d, image_embeds)

        visual_pos_mask = special_image_mask  # (B, seq_len) bool

    # 3. Compute position_ids (Python)
    image_token_id = model.vlm.config.image_token_id
    vision_start_token_id = model.vlm.config.vision_start_token_id
    spatial_merge_size = model.vlm.config.vision_config.spatial_merge_size

    attention_mask_1d = torch.ones(B, input_ids.shape[1], dtype=torch.long, device=device)

    position_ids, rope_deltas = get_rope_index_python(
        input_ids, image_grid_thw, attention_mask_1d,
        image_token_id, vision_start_token_id, spatial_merge_size
    )

    # 4. Decoder prefill (ONNX)
    # Build pre-expanded DeepStack tensors (B, seq_len, hidden_size) with zeros at non-visual positions
    seq_len = inputs_embeds.shape[1]
    hidden_size = inputs_embeds.shape[2]

    if deepstack_embeds is not None and visual_pos_mask is not None:
        ds0, ds1, ds2 = deepstack_embeds
        # Scatter deepstack embeds (merged_seq, hidden_size) into full-size tensors
        ds_full_0 = torch.zeros(B, seq_len, hidden_size, dtype=torch.float32, device=device)
        ds_full_1 = torch.zeros(B, seq_len, hidden_size, dtype=torch.float32, device=device)
        ds_full_2 = torch.zeros(B, seq_len, hidden_size, dtype=torch.float32, device=device)
        for b in range(B):
            vis_indices = visual_pos_mask[b].nonzero(as_tuple=True)[0]
            num_vis = min(vis_indices.shape[0], ds0.shape[0])
            if num_vis > 0:
                ds_full_0[b, vis_indices[:num_vis], :] = ds0[:num_vis].float()
                ds_full_1[b, vis_indices[:num_vis], :] = ds1[:num_vis].float()
                ds_full_2[b, vis_indices[:num_vis], :] = ds2[:num_vis].float()
    else:
        ds_full_0 = torch.zeros(B, seq_len, hidden_size, dtype=torch.float32, device=device)
        ds_full_1 = torch.zeros(B, seq_len, hidden_size, dtype=torch.float32, device=device)
        ds_full_2 = torch.zeros(B, seq_len, hidden_size, dtype=torch.float32, device=device)

    prefill_feeds = {
        "inputs_embeds": inputs_embeds.detach().float().cpu().numpy(),
        "position_ids": position_ids.detach().long().cpu().numpy(),
        "attention_mask": attention_mask_1d.detach().float().cpu().numpy(),
        "deepstack_full_0": ds_full_0.detach().float().cpu().numpy(),
        "deepstack_full_1": ds_full_1.detach().float().cpu().numpy(),
        "deepstack_full_2": ds_full_2.detach().float().cpu().numpy(),
    }

    prefill_result = onnx_components.decoder_prefill.run(prefill_feeds)
    last_hidden = torch.from_numpy(prefill_result[0]).to(device=device)
    past_keys = torch.from_numpy(prefill_result[1]).to(device=device)
    past_values = torch.from_numpy(prefill_result[2]).to(device=device)

    # 5. LM head → first token logits
    # Get logits for last position
    last_pos_hidden = last_hidden[:, -1:, :]
    lm_feeds = {"hidden_states": last_pos_hidden.detach().float().cpu().numpy()}
    logits_result = onnx_components.lm_head.run(lm_feeds)
    logits = torch.from_numpy(logits_result[0]).to(device=device)

    # 6. Autoregressive decode loop
    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    pad_token_id = model.tokenizer.pad_token_id

    generated_ids = [input_ids]
    past_seq_len = past_keys.shape[3]

    for step in range(max_generation_length):
        # Sample next token
        next_token = top_p_sampling(logits[:, -1, :], temperature=temperature, top_p=top_p)
        generated_ids.append(next_token)

        # Check for EOS
        if (next_token == eos_token_id).all():
            break

        # Embed next token
        next_embed_result = onnx_components.embed_tokens.run({
            "input_ids": next_token.detach().long().cpu().numpy()
        })
        next_embed = torch.from_numpy(next_embed_result[0]).to(device=device, dtype=torch.float32)

        # Position IDs for this decode step
        cur_pos = past_seq_len + step + 1
        delta = rope_deltas
        decode_pos = torch.tensor([[[cur_pos]]], dtype=torch.long, device=device)
        decode_pos = decode_pos + delta.unsqueeze(0)
        decode_position_ids = decode_pos.expand(3, B, 1)

        # Attention mask: (B, 1, 1, total_len)
        total_len = past_keys.shape[3] + 1
        decode_attn_mask = torch.zeros(B, 1, 1, total_len, dtype=torch.float32, device=device)

        # Run decode step
        decode_feeds = {
            "input_embeds": next_embed.detach().float().cpu().numpy(),
            "position_ids": decode_position_ids.detach().long().cpu().numpy(),
            "attention_mask": decode_attn_mask.detach().float().cpu().numpy(),
            "past_keys": past_keys.detach().float().cpu().numpy(),
            "past_values": past_values.detach().float().cpu().numpy(),
        }

        decode_result = onnx_components.decoder_decode.run(decode_feeds)
        hidden_state = torch.from_numpy(decode_result[0]).to(device=device)
        past_keys = torch.from_numpy(decode_result[1]).to(device=device)
        past_values = torch.from_numpy(decode_result[2]).to(device=device)

        # LM head
        lm_feeds = {"hidden_states": hidden_state.detach().float().cpu().numpy()}
        logits_result = onnx_components.lm_head.run(lm_feeds)
        logits = torch.from_numpy(logits_result[0]).to(device=device)

    # Concatenate all generated token IDs
    all_sequences = torch.cat(generated_ids, dim=1)
    all_sequences = replace_padding_after_eos(
        token_ids=all_sequences,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    # 7. Prepare for diffusion
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

    prefill_seq_len = past_keys.shape[3]
    diff_attention_mask = torch.zeros(
        (b_star, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
        dtype=torch.float32, device=device,
    )
    for i in range(b_star):
        diff_attention_mask[i, :, :, offset[i]:-n_diffusion_tokens] = torch.finfo(
            torch.float32
        ).min

    # 8. Diffusion loop (ONNX)
    x_dims = model.action_space.get_action_space_dims()
    x = torch.randn(B, *x_dims, device=device)

    time_steps = torch.linspace(0.0, 1.0, model.diffusion.num_inference_steps + 1, device=device)
    n_dim = len(x_dims)

    for i in range(model.diffusion.num_inference_steps):
        dt = time_steps[i + 1] - time_steps[i]
        dt = dt.view(1, *[1] * n_dim).expand(B, *[1] * n_dim)
        t_start = time_steps[i].view(1, *[1] * n_dim).expand(B, *[1] * n_dim)

        t_onnx = t_start[:, :1].view(B, 1) if t_start.dim() > 1 else t_start.unsqueeze(-1)

        v = onnx_components.diffusion_step(
            x, t_onnx, diff_position_ids, diff_attention_mask, past_keys, past_values
        )
        v = v.view(B, *x_dims)
        x = x + dt * v

    sampled_action = x

    # 9. Action to trajectory (PyTorch)
    pred_xyz, pred_rot = model.action_space.action_to_traj(
        sampled_action, ego_history_xyz[:, -1], ego_history_rot[:, -1]
    )

    pred_xyz = pred_xyz.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, ...)
    pred_rot = pred_rot.unsqueeze(1).unsqueeze(1)

    result = (pred_xyz, pred_rot)
    if return_extra:
        extra = extract_text_tokens(model.tokenizer, all_sequences)
        result = (pred_xyz, pred_rot, extra)

    return result


def main():
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    print(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    print("Dataset loaded.")
    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    print("Loading PyTorch model (for tokenizer, action_space, etc.)...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16
    ).to("cuda")
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

    # Load all ONNX components
    onnx_components = OnnxVLMComponents()

    # Run full ONNX inference
    torch.cuda.manual_seed_all(42)
    print("\nRunning full ONNX VLM inference...")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = full_vlm_onnx_inference(
            model=model,
            data=model_inputs,
            onnx_components=onnx_components,
            top_p=0.98,
            temperature=0.6,
            max_generation_length=256,
            return_extra=True,
        )

    print("\nChain-of-Causation:\n", extra.get("cot", ["N/A"])[0])

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, 0, :, :2].T
    ade = np.linalg.norm(pred_xy - gt_xy, axis=0).mean()
    print("ADE:", ade, "meters")


if __name__ == "__main__":
    main()
