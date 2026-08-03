import torch
import torch.nn as nn
from torchvision import models

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create EfficientNet-B0 architecture
model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

# Replace classifier (2 classes: Human and Generated)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

# Load trained weights
model.load_state_dict(
    torch.load(
        "models/finalmodel.pth",
        map_location=device
    )
)

# Move model to device
model = model.to(device)

# Set model to evaluation mode
model.eval()

print("EfficientNet-B0 model loaded successfully.")
