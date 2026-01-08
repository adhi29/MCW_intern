import torch
import torch.nn as nn
import torch.quantization as quant
import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.models import Inception_V3_Weights
from torch.utils.data import DataLoader, Subset
import time
import copy
import sys


print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Supported quantized engines:", torch.backends.quantized.supported_engines)


torch.backends.quantized.engine = "qnnpack"
print("Using quantized engine:", torch.backends.quantized.engine)

device = torch.device("cpu")


# LOAD FP32 MODEL

print("\n[1] Loading FP32 InceptionV3...")

model_fp32 = models.inception_v3(
    weights=Inception_V3_Weights.DEFAULT,
    aux_logits=True
)

# 🔥 CRITICAL FIX FOR QUANTIZATION
model_fp32.transform_input = False

model_fp32.eval()
model_fp32.to(device)

print("✓ FP32 model loaded")


# QUANTIZATION WRAPPER

class QuantizedInception(nn.Module):
    def __init__(self, fp32_model):
        super().__init__()
        self.quant = quant.QuantStub()
        self.model = fp32_model
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)

        # Handle aux logits
        if isinstance(x, tuple):
            x = x[0]

        x = self.dequant(x)
        return x


# PREPROCESSING (DO ALL NORMALIZATION HERE)

print("\n[2] Preparing preprocessing...")

preprocess = transforms.Compose([
    transforms.Resize(342),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

print("✓ Preprocessing ready")


# DATASET

print("\n[3] Loading dataset...")

val_dataset = datasets.ImageFolder(
    root="/Users/adhi/Desktop/Multicoreware/quantization/ILSVRC2012_img_val",
    transform=preprocess
)

# 100 images → calibration
calibration_dataset = Subset(val_dataset, list(range(100)))

# 1000 images → inference
inference_dataset = Subset(val_dataset, list(range(1000)))

calibration_loader = DataLoader(calibration_dataset, batch_size=16, shuffle=False)
inference_loader   = DataLoader(inference_dataset, batch_size=16, shuffle=False)

print("✓ Dataset loaded")
print("  Calibration images:", len(calibration_dataset))
print("  Inference images:", len(inference_dataset))

# PREPARE MODEL FOR STATIC QUANTIZATION
print("\n[4] Preparing quantized model...")

model_int8 = QuantizedInception(copy.deepcopy(model_fp32)).cpu()
model_int8.eval()

# SAFE FUSION (CONV + BN)
print("\n[5] Fusing layers...")

def fuse_recursive(module):
    for _, child in module.named_children():
        if hasattr(child, "conv") and hasattr(child, "bn"):
            try:
                quant.fuse_modules(child, ["conv", "bn"], inplace=True)
            except Exception:
                pass
        fuse_recursive(child)

fuse_recursive(model_int8)

print("✓ Fusion complete")

# QCONFIG + OBSERVERS
print("\n[6] Inserting observers...")

model_int8.qconfig = quant.get_default_qconfig("qnnpack")
quant.prepare(model_int8, inplace=True)

print("✓ Observers inserted")

    # CALIBRATION (100 IMAGES)
print("\n[7] Running calibration...")

with torch.no_grad():
    for images, _ in calibration_loader:
        model_int8(images.cpu())

print("✓ Calibration complete")

# CONVERT TO INT8
print("\n[8] Converting to INT8...")

quant.convert(model_int8, inplace=True)

print("✓ INT8 model ready")

# SAVE MODEL
print("\n[9] Saving INT8 model...")

torch.save(model_int8.state_dict(), "inceptionv3_int8_qnnpack.pth")

print("✓ Model saved")

# INFERENCE FUNCTION
def run_inference(model, loader, tag):
    model.eval()
    correct1 = correct5 = total = 0
    start = time.time()

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.cpu())

            _, pred1 = outputs.max(1)
            _, pred5 = outputs.topk(5, 1)

            correct1 += pred1.eq(labels).sum().item()
            correct5 += pred5.eq(labels.view(-1, 1)).sum().item()
            total += labels.size(0)

    elapsed = time.time() - start
    print(f"{tag} | Top-1: {100*correct1/total:.2f}% | "
          f"Top-5: {100*correct5/total:.2f}% | "
          f"Time: {elapsed:.2f}s")

    return elapsed

# RUN FP32 vs INT8
print("\n" + "=" * 60)
print("EVALUATION (10000 images)")
print("=" * 60)

print("\n[10] Running FP32 inference...")
fp32_time = run_inference(model_fp32, inference_loader, "FP32")

print("\n[11] Running INT8 inference...")
int8_time = run_inference(model_int8, inference_loader, "INT8")

print("\nSpeed-up:", f"{fp32_time / int8_time:.2f}x")
print("\nDONE ✅")
