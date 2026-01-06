import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import time


class InceptionCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_dataset, batch_size=16):
        self.loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
        self.iterator = iter(self.loader)
        self.input_name = "input"
        
    def get_next(self):
        try:
            images, _ = next(self.iterator)
            return {self.input_name: images.numpy()}
        except StopIteration:
            return None


# Preprocessing
print("\n[1] Preparing preprocessing...")

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

# Load dataset
print("\n[2] Loading dataset...")

val_dataset = datasets.ImageFolder(
    root="/Users/adhi/Desktop/Multicoreware/onnx_quantization/ILSVRC2012_img_val",
    transform=preprocess
)

calibration_dataset = Subset(val_dataset, list(range(1000)))
inference_dataset = Subset(val_dataset, list(range(50000)))

print("✓ Dataset loaded")
print("  Calibration images:", len(calibration_dataset))
print("  Inference images:", len(inference_dataset))

# Static quantization to INT8
print("\n[3] Running static quantization to INT8...")

calibration_reader = InceptionCalibrationDataReader(calibration_dataset)

quantize_static(
    model_input="inception_v3.onnx",
    model_output="inception_v3_int8.onnx",
    calibration_data_reader=calibration_reader,
    quant_format=QuantType.QInt8,
    per_channel=False,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)

print("✓ INT8 model saved: inception_v3_int8.onnx")


def run_inference(model_path, loader, tag):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    correct1 = correct5 = total = 0
    start = time.time()
    
    for images, labels in loader:
        outputs = session.run(None, {input_name: images.numpy()})
        logits = outputs[0]
        
        pred1 = np.argmax(logits, axis=1)
        pred5 = np.argsort(logits, axis=1)[:, -5:]
        
        correct1 += (pred1 == labels.numpy()).sum()
        correct5 += sum([label in pred5[i] for i, label in enumerate(labels.numpy())])
        total += labels.size(0)
    
    elapsed = time.time() - start
    print(f"{tag} | Top-1: {100*correct1/total:.2f}% | "
          f"Top-5: {100*correct5/total:.2f}% | "
          f"Time: {elapsed:.2f}s")
    
    return elapsed


# Evaluation
print("EVALUATION (50000 images)")

inference_loader = DataLoader(inference_dataset, batch_size=16, shuffle=False)

print("\n[4] Running FP32 inference...")
fp32_time = run_inference("inception_v3.onnx", inference_loader, "FP32")

print("\n[5] Running INT8 inference...")
int8_time = run_inference("inception_v3_int8.onnx", inference_loader, "INT8")

print("\nSpeed-up:", f"{fp32_time / int8_time:.2f}x")
