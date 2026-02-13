"""Save camera frames from a clip as a grid image for visual inspection."""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, "/data/users/adhi/alpamayo/src")
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
t0_us = 5_100_000
save_dir = "/data/users/adhi"

print(f"Loading clip {clip_id} at t0={t0_us}...")
data = load_physical_aiavdataset(clip_id, t0_us=t0_us)

# image_frames: (num_timesteps, num_cameras, C, H, W) in [0,1] float
frames = data["image_frames"]  # e.g. (4, 3, 3, H, W)
print(f"image_frames shape: {frames.shape}")

# Flatten to (T*cams, C, H, W), convert to HWC numpy
flat = frames.flatten(0, 1)  # (N, C, H, W)
imgs = flat.permute(0, 2, 3, 1).numpy()  # (N, H, W, 3) — already uint8 [0,255]

N, H, W, _ = imgs.shape
cols = 4
rows = (N + cols - 1) // cols
grid = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
for i, img in enumerate(imgs):
    r, c = divmod(i, cols)
    grid[r*H:(r+1)*H, c*W:(c+1)*W] = img

# Save as PNG
try:
    from PIL import Image
    out_path = os.path.join(save_dir, f"clip_frames_{clip_id[:8]}.png")
    Image.fromarray(grid).save(out_path)
    print(f"Saved {N} frames ({rows}x{cols} grid) to {out_path}")
except ImportError:
    import cv2
    out_path = os.path.join(save_dir, f"clip_frames_{clip_id[:8]}.png")
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Saved {N} frames ({rows}x{cols} grid) to {out_path}")
