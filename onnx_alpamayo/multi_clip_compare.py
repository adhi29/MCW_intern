"""
Run compare_e2e_static.py across multiple clips (ONNX only by default)
to see whether CoC and ADE are consistent across different scenes.

Usage:
    python multi_clip_compare.py --onnx-dir ./onnx_models_static_fp16 --n-clips 5
    python multi_clip_compare.py --onnx-dir ./onnx_models_static_fp16 --n-clips 5 --run-pytorch
"""
import argparse
import subprocess
import sys
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=str, default="./onnx_models_static_fp16")
    parser.add_argument("--n-clips", type=int, default=5, help="Number of clips to test")
    parser.add_argument("--save-dir", type=str, default="/data/users/adhi")
    parser.add_argument("--run-pytorch", action="store_true",
                        help="Also run PyTorch (much slower, needs GPU)")
    args = parser.parse_args()

    # Load clip IDs from parquet
    parquet_path = "/data/users/adhi/alpamayo/notebooks/clip_ids.parquet"
    clip_ids = pd.read_parquet(parquet_path)["clip_id"].tolist()

    # Pick a spread: default clip + samples from beginning, middle, end
    selected = ["030c760c-ae38-49aa-9ad8-f5650a545d26"]  # known clip first
    step = max(1, len(clip_ids) // (args.n_clips - 1)) if args.n_clips > 1 else 1
    for i in range(0, len(clip_ids), step):
        cid = clip_ids[i]
        if cid not in selected:
            selected.append(cid)
        if len(selected) >= args.n_clips:
            break

    print(f"Running {len(selected)} clips...")
    print("=" * 80)

    results = []
    for idx, clip_id in enumerate(selected):
        print(f"\n[{idx+1}/{len(selected)}] Clip: {clip_id}")
        print("-" * 60)

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "compare_e2e_static.py"),
            "--onnx-dir", args.onnx_dir,
            "--clip-id", clip_id,
            "--save-dir", args.save_dir,
        ]
        if not args.run_pytorch:
            cmd.append("--skip-pytorch")

        result = subprocess.run(cmd, capture_output=False, text=True)
        results.append((clip_id, result.returncode))

    print("\n" + "=" * 80)
    print("MULTI-CLIP SUMMARY")
    print("=" * 80)
    for clip_id, rc in results:
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {clip_id[:8]}...  {status}")

if __name__ == "__main__":
    main()
