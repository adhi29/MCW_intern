# PyTorch INT8 Quantization - 5th Jan

Static quantization implementation for InceptionV3 using PyTorch's quantization API.

## What's here

- `int8.py` - Main script for static INT8 quantization

## What it does

Takes a pretrained InceptionV3 model and converts it to INT8 using post-training static quantization.

### Steps:
1. Loads FP32 InceptionV3 (pretrained)
2. Wraps model with QuantStub/DeQuantStub
3. Fuses Conv+BatchNorm layers
4. Inserts observers
5. Calibrates on 100 images
6. Converts to INT8
7. Runs inference comparison (FP32 vs INT8)

## Requirements

```bash
torch
torchvision
```

Dataset: ImageNet validation set (ILSVRC2012)

## Running

```bash
python int8.py
```

Make sure you have the ImageNet validation dataset at:
`/Users/adhi/Desktop/Multicoreware/quantization/ILSVRC2012_img_val`

## Key features

- Uses QNNPACK backend (optimized for ARM/mobile)
- Static quantization (requires calibration)
- Handles InceptionV3's auxiliary logits
- Compares Top-1 and Top-5 accuracy
- Measures inference time speedup

## Output

Saves quantized model as `inceptionv3_int8_qnnpack.pth`

Prints:
- Top-1 and Top-5 accuracy for both FP32 and INT8
- Inference time comparison
- Speedup factor
