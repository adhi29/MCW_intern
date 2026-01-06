import torch
import torch.onnx
import torchvision.models as models

# Create the Inception v3 model architecture
model = models.inception_v3(weights=None, init_weights=False)

# Load the saved weights
state_dict = torch.load('inception_v3.pth', map_location='cpu')
model.load_state_dict(state_dict)

# Set to evaluation mode
model.eval()

# Verify model loaded correctly
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Create a dummy input tensor (Inception v3 expects 299x299 images)
dummy_input = torch.randn(1, 3, 299, 299)

# Use the legacy ONNX exporter
with torch.no_grad():
    # Disable the new dynamo-based exporter
    torch.onnx.export(
        model,
        dummy_input,
        "inception_v3.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        # Force legacy exporter
        dynamo=False
    )

# Verify the exported file
import os
file_size = os.path.getsize("inception_v3.onnx") / (1024**2)
print(f"\nONNX model exported successfully!")
print(f"File size: {file_size:.2f} MB")

if file_size < 50:
    print("WARNING: File size seems too small. Weights may not have been exported.")
else:
    print("File size looks correct!")