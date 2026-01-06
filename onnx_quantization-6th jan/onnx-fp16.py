import onnx
from onnxconverter_common import float16
import os

model_fp32 = onnx.load("inception_v3.onnx")
model_fp16 = float16.convert_float_to_float16(model_fp32)
onnx.save(model_fp16, "inception_v3_fp16.onnx")

# Compare file sizes
fp32_size = os.path.getsize("inception_v3.onnx") / (1024**2)
fp16_size = os.path.getsize("inception_v3_fp16.onnx") / (1024**2)

print("ONNX Model Size Comparison")
print(f"FP32 (original): {fp32_size:.2f} MB")
print(f"FP16 (converted): {fp16_size:.2f} MB")
print(f"Size reduction:  {fp32_size - fp16_size:.2f} MB")
print(f"Compression ratio: {fp32_size / fp16_size:.2f}x")
print(f"Space saved: {((fp32_size - fp16_size) / fp32_size * 100):.1f}%")