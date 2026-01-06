import torch
import torchvision.models as models

# Load pretrained Inception v3 model
model = models.inception_v3(pretrained=True)
torch.save(model.state_dict(), "inception_v3.pth")

# Set model to evaluation mode
model.eval()

# Create a dummy input tensor (batch_size=1, 3 channels, 299x299)
dummy_input = torch.randn(1, 3, 299, 299)

# Disable gradient calculation for inference
with torch.no_grad():
    output = model(dummy_input)

print("Model loaded successfully")
print("Output shape:", output.shape)



