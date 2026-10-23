
import torch
import torchvision

# Select the available device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Create a sample image batch
images = torch.randn(4, 3, 224, 224).to(device)

# Apply a simple convolution
model = torch.nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1
).to(device)

output = model(images)

print("Device:", device)
print("Input shape:", images.shape)
print("Output shape:", output.shape)
print("PyTorch:", torch.__version__)
print("Environment check passed!")
