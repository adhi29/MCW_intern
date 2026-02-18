# SPDX-License-Identifier: Apache-2.0
"""
Benchmark Qwen3-VL-2B ONNX models on Qualcomm NPU and iGPU.

Run this script on the ARM64 Snapdragon X Plus device after copying the
onnx_models/ folder from the export machine.

Runs all 5 models on NPU (Hexagon / QnnHtp.dll) first, then iGPU (Adreno / QnnGpu.dll).

Requirements (ARM64 device):
    pip install onnxruntime-qnn numpy

Usage:
    python benchmark_npu.py
    python benchmark_npu.py --onnx-dir ./onnx_models
    python benchmark_npu.py --npu-only       # skip iGPU run
    python benchmark_npu.py --model vision_encoder decoder_prefill
"""

import argparse
import onnxruntime as ort
import numpy as np
import time

# ── Change to your actual ONNX models directory ────────────────────────────
ONNX_DIR = "./onnx_models"

# ── Static shape constants — Qwen3-VL-2B (must match export_qwen3vl_2b.py) ─
BATCH           = 1
PREFILL_SEQ_LEN = 1024
MAX_TOTAL_SEQ   = 1536
HIDDEN          = 2048   # text hidden_size
LAYERS          = 28     # num_hidden_layers
KV_HEADS        = 8      # num_key_value_heads
HEAD_DIM        = 128    # hidden_size / num_attention_heads = 2048/16
TOTAL_PATCHES   = 784    # 1 * 1 * 28 * 28
PATCH_DIM       = 1536   # 3 * 2 * 16 * 16
NUM_IMAGES      = 1
GRID_T          = 1
GRID_H          = 28
GRID_W          = 28


def make_feeds(onnx_dir):
    return {
        "embed_tokens": {
            "path": f"{onnx_dir}/common/vlm_embed_tokens.onnx",
            "feed": {
                "input_ids": np.random.randint(
                    0, 1000, (BATCH, PREFILL_SEQ_LEN), dtype=np.int64,
                ),
            },
            "num_runs": 100,
        },
        "lm_head": {
            "path": f"{onnx_dir}/common/vlm_lm_head.onnx",
            "feed": {
                "hidden_states": np.random.randn(BATCH, 1, HIDDEN).astype(np.float16),
            },
            "num_runs": 100,
        },
        "vision_encoder": {
            "path": f"{onnx_dir}/common/vlm_vision_encoder.onnx",
            "feed": {
                "pixel_values": np.random.randn(TOTAL_PATCHES, PATCH_DIM).astype(np.float32),
                "grid_thw":     np.array([[GRID_T, GRID_H, GRID_W]] * NUM_IMAGES, dtype=np.int64),
            },
            "num_runs": 20,
        },
        "decoder_prefill": {
            "path": f"{onnx_dir}/decoder_prefill/vlm_decoder_prefill.onnx",
            "feed": {
                "inputs_embeds":  np.random.randn(BATCH, PREFILL_SEQ_LEN, HIDDEN).astype(np.float16),
                "position_ids":   np.tile(
                    np.arange(PREFILL_SEQ_LEN, dtype=np.int64), (3, BATCH, 1),
                ),
                "attention_mask": np.ones((BATCH, PREFILL_SEQ_LEN), dtype=np.float16),
            },
            "num_runs": 10,
        },
        "decoder_decode": {
            "path": f"{onnx_dir}/decoder_decode/vlm_decoder_decode.onnx",
            "feed": {
                "input_embeds":   np.random.randn(BATCH, 1, HIDDEN).astype(np.float16),
                "position_ids":   np.full((3, BATCH, 1), MAX_TOTAL_SEQ, dtype=np.int64),
                "attention_mask": np.zeros((BATCH, 1, 1, MAX_TOTAL_SEQ + 1), dtype=np.float16),
                "past_keys":      np.random.randn(
                    LAYERS, BATCH, KV_HEADS, MAX_TOTAL_SEQ, HEAD_DIM,
                ).astype(np.float16),
                "past_values":    np.random.randn(
                    LAYERS, BATCH, KV_HEADS, MAX_TOTAL_SEQ, HEAD_DIM,
                ).astype(np.float16),
            },
            "num_runs": 50,
        },
    }


def run_benchmark(model_name, onnx_path, input_feed, backend_dll, backend_name, num_runs):
    print(f"\n  Model   : {model_name}")
    print(f"  Backend : {backend_name}")

    session_options = ort.SessionOptions()

    try:
        session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=["QNNExecutionProvider"],
            provider_options=[{
                "backend_path": backend_dll,
                "profiling_level": "basic",
            }],
        )

        # Warm up
        print("  Warming up (5 runs)...")
        for _ in range(5):
            session.run(None, input_feed)

        # Benchmark
        print(f"  Running {num_runs} iterations...")
        latencies = []
        output = session.run(None, input_feed)   # initial run to get output shapes
        for _ in range(num_runs):
            start  = time.perf_counter()
            output = session.run(None, input_feed)
            end    = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg = np.mean(latencies)
        mn  = np.min(latencies)
        mx  = np.max(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        fps = 1000 / avg

        print(f"\n  Results — {model_name} on {backend_name}")
        print(f"    Avg: {avg:.2f} ms   Min: {mn:.2f} ms   Max: {mx:.2f} ms")
        print(f"    P50: {p50:.2f} ms   P95: {p95:.2f} ms   FPS: {fps:.2f}")
        print(f"    Output shapes: {[np.array(o).shape for o in output]}")

    except Exception as e:
        print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-VL-2B ONNX models on Qualcomm NPU/iGPU",
    )
    parser.add_argument(
        "--onnx-dir", type=str, default=ONNX_DIR,
    )
    parser.add_argument(
        "--model", nargs="+",
        choices=["embed_tokens", "lm_head", "vision_encoder",
                 "decoder_prefill", "decoder_decode"],
        help="Run only specific models (default: all)",
    )
    parser.add_argument(
        "--npu-only", action="store_true", help="Skip iGPU benchmarks",
    )
    args = parser.parse_args()

    MODELS = make_feeds(args.onnx_dir)
    model_names = args.model if args.model else list(MODELS.keys())

    # ── NPU (Hexagon / QnnHtp.dll) ─────────────────────────────────────────
    print("=" * 60)
    print("Benchmarking on NPU (Hexagon — QnnHtp.dll)")
    print("=" * 60)
    for name in model_names:
        cfg = MODELS[name]
        run_benchmark(
            name, cfg["path"], cfg["feed"],
            "QnnHtp.dll", "NPU (Hexagon)", cfg["num_runs"],
        )

    if not args.npu_only:
        # ── iGPU (Adreno / QnnGpu.dll) ─────────────────────────────────────
        print("\n" + "=" * 60)
        print("Benchmarking on iGPU (Adreno — QnnGpu.dll)")
        print("=" * 60)
        # vision_encoder uses Split op not supported on QnnGpu in QAIRT 2.43
        gpu_skip = {"vision_encoder"}
        for name in model_names:
            if name in gpu_skip:
                print(f"\n  [{name}]  SKIP on iGPU — Split op unsupported by QnnGpu.dll")
                continue
            cfg = MODELS[name]
            run_benchmark(
                name, cfg["path"], cfg["feed"],
                "QnnGpu.dll", "iGPU (Adreno)", cfg["num_runs"],
            )

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
