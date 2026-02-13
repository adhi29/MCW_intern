# Alpamayo R1 ONNX Conversion

This directory contains scripts to convert the [NVIDIA Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B) autonomous driving model from PyTorch to ONNX format.

## Model Architecture

Alpamayo R1 is a Vision-Language-Action (VLA) model for autonomous driving. It combines:

```
                        ┌─────────────────────────────────────────────────────────┐
                        │                   Alpamayo R1 Pipeline                  │
                        │                                                         │
  16 Camera Images ──►  │  ┌──────────────┐    ┌──────────────────┐               │
                        │  │ Vision Encoder│───►│  Text Decoder    │               │
  Text Prompt ────────► │  │ (ViT + RoPE + │    │  (36 Qwen3 layers│               │
                        │  │  DeepStack)   │    │  + KV-cache)     │               │
                        │  └──────────────┘    └────────┬─────────┘               │
                        │                               │                         │
                        │                    ┌──────────▼─────────┐               │
                        │                    │  Expert Decoder     │               │
                        │                    │  (36 layers +       │──► Trajectory │
                        │                    │   cross-attn to VLM │    Prediction │
                        │                    │   KV-cache)         │               │
                        │                    └──────────┬─────────┘               │
                        │                               │                         │
                        │                    ┌──────────▼─────────┐               │
                        │                    │  Flow Matching      │               │
                        │                    │  Diffusion (10 step)│               │
                        │                    └────────────────────┘               │
                        └─────────────────────────────────────────────────────────┘
```

### Key Architectural Details

| Property | Value |
|----------|-------|
| VLM Base | Qwen3-VL-8B |
| Hidden Size | 4096 |
| Decoder Layers | 36 |
| Attention Heads | 32 (8 KV heads, GQA) |
| Head Dimension | 128 |
| Vocab Size | 155,697 |
| Expert Hidden Size | 2048 |
| Expert Layers | 36 |
| Position Encoding | 3D MRoPE (sections: [24, 20, 20]) |
| DeepStack Layers | 3 (from ViT blocks 8, 16, 24 → decoder layers 0, 1, 2) |
| Diffusion Steps | 10 (Euler integration) |
| Spatial Merge Size | 2 |

## ONNX Conversion Strategy

The model is split into **8 ONNX components** across 3 phases. Each component is independently exportable, verifiable, and can run on different hardware.

### Why Split?

1. **Memory**: The full model is ~20GB — splitting allows loading components independently
2. **Flexibility**: Different components can run on different devices (GPU/CPU/NPU)
3. **Debuggability**: Each component can be tested and verified against PyTorch independently
4. **ONNX Tracing Limitations**: Complex control flow (autoregressive loops, diffusion scheduling) cannot be traced — these remain as Python orchestration

## ONNX Components

### Phase 1: Action Projections

| File | ONNX Model | Description |
|------|------------|-------------|
| `export_alpamayo.py` | `action_in_proj.onnx` | Encodes raw action vectors into diffusion token space |
| `export_alpamayo.py` | `action_out_proj.onnx` | Decodes diffusion output back to action space |

### Phase 2: Expert Model

| File | ONNX Model | Description |
|------|------------|-------------|
| `export_expert_kvcache.py` | `diffusion_step_kvcache.onnx` | Expert decoder (36 layers) with cross-attention to VLM KV-cache. Single diffusion denoising step. |

### Phase 3: VLM (Qwen3-VL)

| File | ONNX Model | Location | Description |
|------|------------|----------|-------------|
| `export_vlm_head.py` | `vlm_embed_tokens.onnx` | `common/` | Token embedding lookup (vocab → hidden) |
| `export_vlm_head.py` | `vlm_lm_head.onnx` | `common/` | Logits projection (hidden → vocab) |
| `export_vlm_vision.py` | `vlm_vision_encoder.onnx` | `common/` | ViT encoder + DeepStack feature extraction |
| `export_vlm_decoder.py` | `vlm_decoder_prefill.onnx` | `decoder_prefill/` | 36-layer decoder prefill with DeepStack injection |
| `export_vlm_decoder.py` | `vlm_decoder_decode.onnx` | `decoder_decode/` | 36-layer decoder single-token decode with KV-cache |

> **Note:** Decoder prefill and decode are in separate directories because `torch.onnx.export` creates external data files with auto-generated names (`onnx::MatMul_XXXXX`) that would conflict if both models shared the same directory.

## Directory Structure

```
onnx_export/
├── README.md
│
├── # ─── Export Scripts ───
├── export_alpamayo.py           # Phase 1: action_in_proj, action_out_proj
├── export_expert_kvcache.py     # Phase 2: expert diffusion step
├── export_vlm_head.py           # Phase 3: embed_tokens, lm_head
├── export_vlm_vision.py         # Phase 3: vision encoder + DeepStack
├── export_vlm_decoder.py        # Phase 3: decoder prefill + decode
├── export_all_vlm.sh            # Convenience: export all Phase 3 components
│
├── # ─── Inference Pipelines ───
├── hybrid_inference.py          # Phase 1: hybrid PyTorch + ONNX inference
├── full_onnx_inference.py       # Phase 2: full ONNX (expert + action projs)
├── vlm_onnx_inference.py        # Phase 3: full end-to-end ONNX pipeline
│
├── # ─── Verification Scripts ───
├── verify_components.py         # Phase 1: compare action proj ONNX vs PyTorch
├── compare_results.py           # Phase 1: compare hybrid vs pure PyTorch results
├── compare_full_onnx.py         # Phase 2: compare expert ONNX vs PyTorch
├── compare_vlm_onnx.py          # Phase 3: compare all VLM ONNX vs PyTorch
│
├── # ─── Utilities ───
├── convert_onnx_precision.py    # Convert fp32 ONNX models to bf16 or fp16
├── extract_hf_models.py         # Extract sub-models from HuggingFace checkpoint
├── test_onnx_pipeline.py        # Quick sanity tests
│
├── # ─── ONNX Model Outputs ───
├── onnx_models/                 # Phase 1 & 2 ONNX models
│   ├── action_in_proj.onnx
│   ├── action_out_proj.onnx
│   ├── diffusion_step_kvcache.onnx
│   └── diffusion_step.onnx
│
└── onnx_models_vlm_clean/       # Phase 3 VLM ONNX models
    ├── common/
    │   ├── vlm_embed_tokens.onnx    + external data files
    │   ├── vlm_lm_head.onnx        + external data files
    │   └── vlm_vision_encoder.onnx  + external data files
    ├── decoder_prefill/
    │   └── vlm_decoder_prefill.onnx + external data files
    └── decoder_decode/
        └── vlm_decoder_decode.onnx  + external data files
```

## How to Run

### Prerequisites

```bash
# Activate the virtual environment
source /data/users/adhi/alpamayo/ar1_venv/bin/activate

# Required packages (already installed in ar1_venv)
# torch, onnx, onnxruntime, transformers, einops
```

### Step 1: Export ONNX Models

```bash
cd /data/users/adhi/alpamayo/onnx_export

# Phase 1: Action projections
python export_alpamayo.py

# Phase 2: Expert diffusion step with KV-cache
python export_expert_kvcache.py

# Phase 3: VLM components (to separate directories)
python export_vlm_head.py --output-dir ./onnx_models_vlm_clean/common
python export_vlm_vision.py --output-dir ./onnx_models_vlm_clean/common
python export_vlm_decoder.py --prefill-only --output-dir ./onnx_models_vlm_clean/decoder_prefill
python export_vlm_decoder.py --decode-only --output-dir ./onnx_models_vlm_clean/decoder_decode
```

### Step 2: Verify ONNX Models

```bash
# Phase 1: Verify action projections
python verify_components.py

# Phase 2: Verify expert model
python compare_full_onnx.py

# Phase 3: Verify all VLM components
python compare_vlm_onnx.py
```

### Step 3: Convert to bfloat16 (Optional)

```bash
python convert_onnx_precision.py --input ./onnx_models_vlm_clean/common/vlm_embed_tokens.onnx --precision bf16
python convert_onnx_precision.py --input ./onnx_models_vlm_clean/common/vlm_lm_head.onnx --precision bf16
python convert_onnx_precision.py --input ./onnx_models_vlm_clean/common/vlm_vision_encoder.onnx --precision bf16
python convert_onnx_precision.py --input ./onnx_models_vlm_clean/decoder_prefill/vlm_decoder_prefill.onnx --precision bf16
python convert_onnx_precision.py --input ./onnx_models_vlm_clean/decoder_decode/vlm_decoder_decode.onnx --precision bf16
```

### Step 4: Run Full ONNX Inference

```bash
# Full end-to-end ONNX pipeline (requires GPU)
python vlm_onnx_inference.py
```

## Verification Results

All components verified with negligible numerical differences (float32):

| Component | Max Diff | Mean Diff | Relative Diff | Status |
|-----------|----------|-----------|---------------|--------|
| embed_tokens | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| lm_head | 2.74e-06 | 2.68e-07 | 4.29e-07 | PASS |
| vision_encoder (image_embeds) | 5.48e-05 | 2.84e-07 | 2.79e-06 | PASS |
| vision_encoder (deepstack_0) | 3.99e-06 | 2.72e-07 | 2.53e-06 | PASS |
| vision_encoder (deepstack_1) | 7.84e-06 | 3.78e-07 | 2.53e-06 | PASS |
| vision_encoder (deepstack_2) | 1.29e-05 | 1.95e-07 | 1.84e-06 | PASS |
| decoder_prefill (hidden) | 6.94e-04 | 9.44e-06 | 1.22e-05 | PASS |
| decoder_prefill (keys) | 2.21e-04 | 2.52e-06 | 2.66e-06 | PASS |
| decoder_prefill (values) | 8.05e-04 | 3.95e-06 | 9.05e-06 | PASS |
| decoder_decode (hidden) | 2.29e-05 | 3.78e-07 | 3.32e-07 | PASS |
| decoder_decode (keys) | 3.91e-05 | 2.85e-08 | 1.78e-07 | PASS |
| decoder_decode (values) | 2.10e-05 | 3.33e-08 | 3.50e-07 | PASS |

## Is the ONNX Architecture the Same as Alpamayo?

**Yes — the ONNX models preserve the exact same architecture.** Every layer, every weight, every operation is faithfully replicated. The differences are:

### What is identical:
- All 36 Qwen3-VL decoder layers (self-attention + MLP + RMSNorm)
- All 36 Expert decoder layers (self-attention + cross-attention + MLP)
- Vision encoder (27 ViT blocks + spatial merger + DeepStack mergers)
- Token embeddings and LM head projections
- KV-cache cross-attention between Expert and VLM
- DeepStack feature injection at decoder layers 0, 1, 2
- Flow matching diffusion denoising

### What is handled differently (by design):
- **KV-cache**: PyTorch uses `DynamicCache` objects; ONNX uses flat tensors `(num_layers, B, num_kv_heads, seq_len, head_dim)` that are reconstructed into `DynamicCache` inside the wrapper
- **DeepStack injection**: PyTorch uses boolean indexing (`hidden_states[mask, :]`); ONNX uses pre-expanded full-size tensors with zeros at non-visual positions (element-wise add)
- **Attention**: PyTorch uses SDPA (Scaled Dot Product Attention); ONNX uses eager math attention (equivalent computation, different kernel)

### What remains in Python orchestration:
These are non-neural-network operations with no learned weights:
- `get_rope_index()` — 3D MRoPE position ID computation
- `masked_scatter` — merging visual embeddings into text sequence
- Token sampling (top-p / temperature)
- Diffusion Euler loop scheduling
- `action_to_traj()` — kinematic conversion (rotation + translation)

## Known Limitations

1. **Vision encoder grid dimensions are baked in**: The vision encoder is exported with 16 images of grid `(1, 20, 36)` (the standard Alpamayo 16-camera setup). Different image configurations require re-exporting the vision encoder with matching grid dimensions.

2. **External data storage**: Large models (decoder prefill/decode) use ONNX external data storage. The `.onnx` file contains only the graph; weights are in separate files in the same directory.

3. **bfloat16 runtime**: While the conversion to bf16 reduces model size by 2x, ONNX Runtime's bf16 support varies by platform. GPU execution providers generally support bf16; CPU providers may need float32 I/O with bf16 weights.
