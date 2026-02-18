# Qwen3-VL-2B NPU Export and Benchmarking

Export **Qwen3-VL-2B-Instruct** to static-shape ONNX models and benchmark them on a Qualcomm Snapdragon NPU (Hexagon HTP) and iGPU (Adreno).

---

## Overview

The model is split into 5 ONNX components so each can be compiled and accelerated independently on the NPU:

| Component | File | Description |
|---|---|---|
| `embed_tokens` | `common/vlm_embed_tokens.onnx` | Token ID → embedding lookup |
| `lm_head` | `common/vlm_lm_head.onnx` | Hidden state → vocabulary logits |
| `vision_encoder` | `common/vlm_vision_encoder.onnx` | Image patches → image embeddings (fp32) |
| `decoder_prefill` | `decoder_prefill/vlm_decoder_prefill.onnx` | Full-sequence prefill with static KV cache output |
| `decoder_decode` | `decoder_decode/vlm_decoder_decode.onnx` | Single-token decode with KV cache read/write |

### Static Cache Design

The NPU (QNN HTP backend) requires all tensor shapes to be known at compile time. The decoder uses a **static KV cache** with fixed size `(1, 8, 1536, 128)` per layer:

- **Prefill** (`PREFILL_SEQ_LEN = 1024`): Writes keys/values for the full prompt into the cache.
- **Decode** (`MAX_GENERATION = 512`): Uses `ScatterElements` (`torch.scatter`) to write one new token's KV slot at `write_index` into the static cache buffer per step.

KV tensors are passed as **28 separate named ONNX inputs/outputs** (not stacked) to prevent TorchScript constant-folding.

---

## Files

| File | Purpose |
|---|---|
| `export_qwen3vl_2b.py` | Export all 5 model components to ONNX |
| `run_e2e_baseline.py` | fp32/fp16 end-to-end accuracy baseline |
| `verify.py` | Quick shape + mean-error ONNX verification |
| `verify_qwen3vl_2b.py` | Detailed numerical verification (cosine sim, top-5 tokens, KV write check) |
| `benchmark_npu.py` | Latency benchmark on NPU (QnnHtp) and iGPU (QnnGpu) |
| `unsuport_ops.py` | List all ONNX operators in a model (for NPU support checking) |

---

## Requirements

**Export machine (x86 Linux, ~8 GB RAM):**
```bash
pip install torch transformers qwen-vl-utils onnx
```

**Snapdragon ARM64 device (benchmark/verify on NPU):**
```bash
pip install onnxruntime-qnn numpy
```

---

## Usage

### Step 1 — Baseline accuracy check
```bash
python run_e2e_baseline.py
# Runs fp32 and fp16 PyTorch inference, saves reference.json
```

### Step 2 — Export to ONNX
```bash
python export_qwen3vl_2b.py --model-path Qwen/Qwen3-VL-2B-Instruct --output-dir ./onnx_models

# Export specific components only
python export_qwen3vl_2b.py --export-only embed_tokens lm_head
```

Output layout:
```
onnx_models/
  common/
    vlm_embed_tokens.onnx
    vlm_lm_head.onnx
    vlm_vision_encoder.onnx
  decoder_prefill/
    vlm_decoder_prefill.onnx
  decoder_decode/
    vlm_decoder_decode.onnx
```

### Step 3 — Verify exports (CPU, x86)
```bash
# Quick check
python verify.py --onnx-dir ./onnx_models

# Detailed check with cosine similarity and top-5 token agreement
python verify_qwen3vl_2b.py --onnx-dir ./onnx_models --provider cpu
```

### Step 4 — Benchmark on Snapdragon device
Copy `onnx_models/` to the ARM64 device, then:
```bash
python benchmark_npu.py --onnx-dir ./onnx_models

# NPU only
python benchmark_npu.py --npu-only

# Specific models only
python benchmark_npu.py --model vision_encoder decoder_prefill
```

### Check ONNX operators (NPU support)
```bash
python unsuport_ops.py
# Edit model_path inside the script to point to the desired .onnx file
```

---

## Model Architecture (Qwen3-VL-2B-Instruct)

| Parameter | Value |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `vocab_size` | 151936 |

### Decoder I/O shapes

**decoder_prefill** — 3 inputs, 57 outputs:
```
IN:  inputs_embeds  (1, 1024, 2048)
     position_ids   (3, 1, 1024)  int64
     attention_mask (1, 1024)     fp16

OUT: last_hidden_state            (1, 1024, 2048)
     past_key_{0..27}             (1, 8, 1024, 128)  x28
     past_value_{0..27}           (1, 8, 1024, 128)  x28
```

**decoder_decode** — 61 inputs, 57 outputs:
```
IN:  input_embeds                 (1, 1, 2048)
     position_ids                 (3, 1, 1)     int64
     attention_mask               (1, 1, 1, 1536) fp16
     write_index                  ()            int64  scalar
     past_key_{0..27}             (1, 8, 1536, 128) x28
     past_value_{0..27}           (1, 8, 1536, 128) x28

OUT: hidden_state                 (1, 1, 2048)
     new_past_key_{0..27}         (1, 8, 1536, 128) x28
     new_past_value_{0..27}       (1, 8, 1536, 128) x28
```

---

## Notes

- `vision_encoder` is exported in **fp32** — the ONNX `Range` op does not support fp16.
- `vision_encoder` is skipped on iGPU (`QnnGpu.dll`) benchmarks — the `Split` op is unsupported in QAIRT 2.43.
- Both decoders use **TorchScript export** (`dynamo=False`) with `do_constant_folding=True`.
- A `decoder_decode.onnx` smaller than ~500 MB indicates the KV cache was constant-folded — re-export if this happens.
