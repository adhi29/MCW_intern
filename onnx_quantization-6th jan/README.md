# ONNX Model Quantization - Inception V3

This project demonstrates model quantization techniques for the Inception V3 architecture using ONNX Runtime. The implementation includes conversion from FP32 (full precision) to FP16 (half precision) and INT8 (8-bit integer) quantization, with comprehensive accuracy and performance benchmarking.

## Overview

Model quantization reduces model size and improves inference speed while maintaining acceptable accuracy. This project explores:

- **FP32 → FP16 Conversion**: Reduces model size by ~50% with minimal accuracy loss
- **FP32 → INT8 Quantization**: Reduces model size by ~75% using static quantization with calibration

## Accuracy Results

Evaluation performed on 50,000 ImageNet validation images:

| Model Type | Top-1 Accuracy | Top-5 Accuracy | Inference Time | Speed-up |
|------------|---------------|----------------|----------------|----------|
| **FP32** (Baseline) | 77.29% | 93.45% | 1,869.71s | 1.00x |
| **FP16** | 77.29% | 93.45% | 1,847.66s | 1.01x |
| **INT8** | 76.95% | 93.29% | 1,819.68s | 1.03x |

### Key Findings

- **FP16**: Maintains identical accuracy to FP32 with slight performance improvement
- **INT8**: Minimal accuracy degradation (0.34% Top-1, 0.16% Top-5) with 3% speed improvement
- **Model Size Reduction**:
  - FP32: 95.3 MB
  - FP16: 47.7 MB (50% reduction)
  - INT8: 24.1 MB (75% reduction)

## Project Structure

```
onnx_quantization/
├── int8.py                 # INT8 static quantization with calibration
├── onnx-fp16.py           # FP16 conversion script
├── onnx_conversion.py     # PyTorch to ONNX conversion
├── model_load.py          # Model loading utilities
├── fp32-fp16.py           # FP32/FP16 comparison
├── onnx-int8.ipynb        # Jupyter notebook for experimentation
└── README.md              # This file
```

## Dependencies

```bash
pip install torch torchvision
pip install onnx onnxruntime
pip install onnxconverter-common
pip install numpy
```

## Usage

### 1. Convert PyTorch Model to ONNX (FP32)

```bash
python onnx_conversion.py
```

This converts the PyTorch Inception V3 model to ONNX format.

### 2. Convert FP32 to FP16

```bash
python onnx-fp16.py
```

Converts the FP32 ONNX model to FP16, reducing model size by approximately 50%.

### 3. Quantize to INT8

```bash
python int8.py
```

Performs static quantization to INT8 using 1,000 calibration images. This script:
- Loads the ImageNet validation dataset
- Uses 1,000 images for calibration
- Quantizes the model to INT8
- Evaluates accuracy on all 50,000 validation images
- Compares FP32 vs INT8 performance

## Dataset

The project uses the **ImageNet ILSVRC2012 validation set** (50,000 images) for evaluation. The dataset should be organized as:

```
ILSVRC2012_img_val/
├── n01440764/
├── n01443537/
└── ...
```

> **Note**: The dataset is not included in this repository due to size constraints. Download from [ImageNet](https://image-net.org/).

## Preprocessing

Images are preprocessed using standard Inception V3 transformations:
- Resize to 342×342
- Center crop to 299×299
- Normalize with ImageNet mean and std

## Results Summary

### Accuracy vs Performance Trade-off

The quantization experiments demonstrate that:

1. **FP16 is a safe choice**: No accuracy loss with similar or better performance
2. **INT8 offers best compression**: 75% size reduction with only 0.34% accuracy drop
3. **Calibration is effective**: Using just 1,000 images for calibration maintains high accuracy

### When to Use Each Format

- **FP32**: Maximum accuracy, larger model size
- **FP16**: Best balance for GPU inference, no accuracy loss
- **INT8**: Edge devices and mobile deployment, minimal accuracy impact

## Technical Details

### Static Quantization Process

The INT8 quantization uses ONNX Runtime's static quantization with:
- **Calibration method**: MinMax
- **Calibration dataset**: 1,000 ImageNet validation images
- **Quantization type**: Per-tensor symmetric quantization
- **Weight type**: QInt8
- **Activation type**: QInt8

### Inference Configuration

- **Batch size**: 16
- **Runtime**: ONNX Runtime
- **Hardware**: CPU inference (results may vary on GPU/specialized hardware)

## License

This project is for educational and research purposes.

## Acknowledgments

- Model: Inception V3 from torchvision
- Dataset: ImageNet ILSVRC2012
- Framework: ONNX Runtime for quantization and inference
