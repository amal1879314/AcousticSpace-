import torch
import torch.nn as nn
from torchvision import models

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create ResNet18 architecture
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace final layer (2 classes: Human and Generated)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Load trained weights
model.load_state_dict(
    torch.load(
        "models/resnet18_baseline-50% (1).pth",
        map_location=device
    )
)

# Move model to device
model = model.to(device)

# Set model to evaluation mode
model.eval()

print("✅ ResNet18 model loaded successfully!")
