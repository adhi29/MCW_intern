# InceptionV3 Experiments - 2nd Jan

Playing around with InceptionV3 on ImageNet validation set. Tried different precision formats and ran some benchmarks.

## Files

**Dataset prep:**
- `class_organize.py` - organizes ImageNet val images into class folders (needed for PyTorch's ImageFolder)
- `imagenet_class_index.json` - class mappings
- `val_labels.txt` - ground truth labels for 50k validation images

**Inference scripts:**
- `load_inference.py` - basic FP32 inference
- `torchfp16.py` - FP16 inference (auto-detects GPU and converts model)

**Saved models:**
- `inceptionv3_int8.pth` - quantized model (first attempt)
- `inceptionv3_int8_qnnpack.pth` - quantized with QNNPACK backend

## Setup

Need ImageNet validation set extracted somewhere. The scripts point to `/Users/adhi/Desktop/Multicoreware/quantization/ILSVRC2012_img_val` but you can change that.

First run `class_organize.py` to sort images into folders by class. Takes a few minutes.

## Running inference

FP32:
```bash
python load_inference.py
```

FP16 (if you have GPU):
```bash
python torchfp16.py
```

Both scripts show Top-1/Top-5 accuracy and timing info. Expected accuracy is around 77% Top-1 and 93% Top-5 for the pretrained model.

## Notes

- FP16 only helps if you have a GPU (CUDA or MPS)
- The quantized models are from static quantization experiments
- Batch size is 32, can tweak if needed
