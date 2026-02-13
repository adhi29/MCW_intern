# SPDX-License-Identifier: Apache-2.0
"""
Export the Qwen3-VL vision encoder to ONNX.

The vision encoder processes image patches through 27 ViT blocks,
producing visual embeddings (4096-dim) and DeepStack features
for injection into the text decoder at layers [8, 16, 24].
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class VisionEncoderForONNX(nn.Module):
    """
    Wraps Qwen3VLVisionModel for ONNX export.

    Uses eager attention (no Flash Attention / cu_seqlens) and
    processes a single image at a time for simplicity.
    """

    def __init__(self, vision_model):
        super().__init__()
        self.patch_embed = vision_model.patch_embed
        self.pos_embed = vision_model.pos_embed
        self.num_grid_per_side = vision_model.num_grid_per_side
        self.rotary_pos_emb = vision_model.rotary_pos_emb
        self.blocks = vision_model.blocks
        self.merger = vision_model.merger
        self.spatial_merge_size = vision_model.spatial_merge_size
        self.deepstack_visual_indexes = vision_model.deepstack_visual_indexes
        self.deepstack_merger_list = vision_model.deepstack_merger_list

        # Force eager attention for all blocks
        for block in self.blocks:
            block.attn.config._attn_implementation = "eager"

    def forward(
        self,
        pixel_values: torch.Tensor,    # (seq_len, patch_dim)
        grid_thw: torch.Tensor,        # (num_images, 3) int64
    ):
        """
        Process image patches through vision encoder.

        Returns:
            image_embeds: (merged_seq, 4096)
            deepstack_embed_0: (merged_seq, 4096) - from layer 8
            deepstack_embed_1: (merged_seq, 4096) - from layer 16
            deepstack_embed_2: (merged_seq, 4096) - from layer 24
        """
        hidden_states = self.patch_embed(pixel_values)

        # Position embeddings via bilinear interpolation
        pos_embeds = self._fast_pos_embed(hidden_states, grid_thw)
        hidden_states = hidden_states + pos_embeds

        # Rotary position embeddings
        rotary_emb = self._rot_pos_emb(grid_thw)
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_emb = rotary_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_emb, rotary_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Compute cu_seqlens for block-wise attention
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        # Run through vision blocks
        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                ds_feat = self.deepstack_merger_list[idx](hidden_states)
                deepstack_features.append(ds_feat)

        # Final merger
        image_embeds = self.merger(hidden_states)

        # Pad deepstack to always return 3 features
        while len(deepstack_features) < 3:
            deepstack_features.append(torch.zeros_like(image_embeds))

        return image_embeds, deepstack_features[0], deepstack_features[1], deepstack_features[2]

    def _fast_pos_embed(self, hidden_states, grid_thw):
        """Simplified position embedding computation."""
        grid_hs = grid_thw[:, 1]
        grid_ws = grid_thw[:, 2]
        grid_ts = grid_thw[:, 0]
        merge_size = self.spatial_merge_size

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(h.item()),
                                    device=hidden_states.device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(w.item()),
                                    device=hidden_states.device)

            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        device = self.pos_embed.weight.device
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [int(h.item()) * int(w.item()) for h, w in zip(grid_hs, grid_ws)]
        )

        result = []
        for pos_emb, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            t, h, w = int(t.item()), int(h.item()), int(w.item())
            pos_emb = pos_emb.repeat(t, 1)
            pos_emb = (
                pos_emb.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result.append(pos_emb)

        return torch.cat(result)

    def _rot_pos_emb(self, grid_thw):
        """Compute rotary position embeddings for vision."""
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            num_frames = int(num_frames.item())
            height = int(height.item())
            width = int(width.item())
            merged_h = height // merge_size
            merged_w = width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)


def export_vision_encoder(model: AlpamayoR1, output_dir: str, opset_version: int = 17):
    """Export the vision encoder to ONNX."""
    print("Exporting VLM vision encoder...")

    vision_model = model.vlm.model.visual
    wrapper = VisionEncoderForONNX(vision_model)
    wrapper = wrapper.float()
    wrapper.eval()

    # Dummy inputs matching real Alpamayo data: 16 camera images
    # Each image: grid_thw = (1, 20, 36), total_patches per image = 20*36 = 720
    # 16 images × 720 = 11520 total patches
    # patch_dim = 3 * 2 * 16 * 16 = 1536
    num_images = 16
    grid_t, grid_h, grid_w = 1, 20, 36
    total_patches = num_images * grid_t * grid_h * grid_w
    patch_dim = 3 * 2 * 16 * 16  # in_channels * temporal_patch_size * patch_size * patch_size

    dummy_pixels = torch.randn(total_patches, patch_dim)
    dummy_grid = torch.tensor([[grid_t, grid_h, grid_w]] * num_images, dtype=torch.long)

    output_path = os.path.join(output_dir, "vlm_vision_encoder.onnx")

    with torch.no_grad():
        # Test forward first
        print("  Testing forward pass...")
        out = wrapper(dummy_pixels, dummy_grid)
        print(f"  image_embeds shape: {out[0].shape}")
        print(f"  deepstack_0 shape: {out[1].shape}")

        print("  Exporting to ONNX...")
        try:
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
                dynamic_axes={
                    "pixel_values": {0: "total_patches"},
                    "grid_thw": {0: "num_images"},
                    "image_embeds": {0: "merged_seq"},
                    "deepstack_embed_0": {0: "merged_seq"},
                    "deepstack_embed_1": {0: "merged_seq"},
                    "deepstack_embed_2": {0: "merged_seq"},
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
    parser = argparse.ArgumentParser(description="Export Qwen3-VL vision encoder to ONNX")
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

    export_vision_encoder(model, args.output_dir, args.opset_version)
    print("\nDone!")


if __name__ == "__main__":
    main()
