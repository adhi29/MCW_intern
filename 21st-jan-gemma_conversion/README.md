# Gemma Model ONNX Conversion

This folder contains scripts for converting Google's Gemma language models to ONNX format with static key-value cache support.

## 📁 Files

### 1. `build_gemma.py`
A custom ONNX graph builder for Gemma 3 models that manually constructs ONNX IR instead of using `torch.onnx.export`.

**Key Features:**
- Manual ONNX graph construction using `onnx_ir` library
- Support for Gemma 3 architecture (`Gemma3ForCausalLM`)
- Custom RoPE (Rotary Position Embedding) implementation
- Static KV cache management
- Multiple precision support: `fp32`, `fp16`, `q4`, `q4f16`
- INT4 quantization capability

**Main Components:**
- `Gemma3Model`: Core model builder class
- `RopeCacheConfig`: Configuration for rotary embeddings
- Layer builders: embedding, attention, MLP, layer normalization
- Model export and quantization utilities

**Supported Models:**
- Gemma 3 270M (optimized for edge/browser deployment)

### 2. `gemma_onnx_conversion.py`
A comprehensive conversion script that exports Gemma models to ONNX format with separate prefill and decode phases.

**Key Features:**
- Loads Gemma models from Hugging Face
- Exports separate ONNX models for prefill and decode phases
- Static shape optimization for inference
- Comparison with PyTorch reference outputs
- Performance benchmarking
- Detailed logging and validation

**Main Components:**
- `GemmaPrefillWrapper`: Handles initial token processing
- `GemmaDecodeWrapper`: Handles autoregressive generation with KV cache
- `GemmaONNXConverter`: Main conversion orchestrator
- Validation and benchmarking utilities

**Output:**
- `gemma_prefill.onnx`: Prefill phase model
- `gemma_decode.onnx`: Decode phase model
- Comparison reports and benchmark results

## 🚀 Usage

### Prerequisites

```bash
pip install torch transformers onnx onnxruntime onnx_ir numpy
```

### Using build_gemma.py

```bash
python build_gemma.py \
    --model_name google/gemma-3-270m-it \
    --output ./gemma3_onnx \
    -p fp32 fp16 q4 q4f16
```

**Arguments:**
- `--model_name`: Hugging Face model identifier
- `--output`: Output directory for ONNX models
- `-p`: Precision options (can specify multiple)

### Using gemma_onnx_conversion.py

```bash
python gemma_onnx_conversion.py \
    --model_name google/gemma-1.1-2b-it \
    --hf_token YOUR_HUGGINGFACE_TOKEN \
    --output_dir ./gemma_onnx \
    --batch_size 1 \
    --seq_length 128 \
    --max_length 512
```

**Arguments:**
- `--model_name`: Hugging Face model identifier
- `--hf_token`: Hugging Face authentication token
- `--output_dir`: Directory to save ONNX models
- `--batch_size`: Batch size for static shapes (default: 1)
- `--seq_length`: Input sequence length (default: 128)
- `--max_length`: Maximum KV cache length (default: 512)
- `--compare`: Compare ONNX outputs with PyTorch reference
- `--benchmark`: Run performance benchmarks

## 🔍 Technical Details

### Architecture Differences

**Gemma 3 270M:**
- Specifically designed for ONNX export
- Smaller model (270M parameters)
- Optimized for edge/browser deployment
- Compatible with `build_gemma.py`

**Gemma 1.1 2B:**
- General-purpose language model
- Larger model (2B parameters)
- More complex architecture
- Requires custom adaptation for ONNX

### ONNX Conversion Approach

Both scripts use different strategies:

1. **build_gemma.py**: Manual ONNX IR construction
   - Builds graph node-by-node
   - Full control over operations
   - Optimized for specific architecture

2. **gemma_onnx_conversion.py**: PyTorch export wrapper
   - Uses `torch.onnx.export`
   - Wrapper classes for static shapes
   - Separate prefill/decode models

### Static KV Cache

The conversion implements static key-value cache for efficient autoregressive generation:

- **Prefill Phase**: Processes initial prompt, generates KV cache
- **Decode Phase**: Uses cached keys/values for token-by-token generation
- **Benefits**: Reduced computation, faster inference

## 📊 Performance Considerations

- **Quantization**: INT4 quantization can reduce model size by ~4x
- **Static Shapes**: Enables better optimization in ONNX Runtime
- **Separate Phases**: Allows independent optimization of prefill/decode
- **Browser Deployment**: Gemma 3 270M can run in browsers via Transformers.js

## ⚠️ Important Notes

1. **Model Compatibility**: `build_gemma.py` only works with Gemma 3 models
2. **Token Requirements**: Gemma models require Hugging Face authentication
3. **Memory Usage**: 2B models require significant RAM/VRAM
4. **Validation**: Always compare ONNX outputs with PyTorch reference

## 🔗 Related Resources

- [Gemma Models on Hugging Face](https://huggingface.co/google)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Transformers.js](https://huggingface.co/docs/transformers.js)

## 📝 Project Context

This work is part of exploring ONNX conversion strategies for Gemma language models, comparing different approaches and evaluating their effectiveness for deployment scenarios.

**Date**: January 21, 2026  
**Focus**: ONNX conversion with static KV cache for efficient inference
