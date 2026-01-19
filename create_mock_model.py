"""
Create a mock Vision Transformer model for testing the application.
This is NOT a trained model - it will give random predictions.
Replace with your actual trained model for production use.
"""

import torch
from torchvision import models
import json

# Load configuration
with open('models/model_config.json', 'r') as f:
    config = json.load(f)

with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

# Create model architecture
print("Creating Vision Transformer model...")
model = models.vit_b_16(pretrained=False)
num_classes = len(class_names)
model.heads = torch.nn.Linear(model.heads.head.in_features, num_classes)

# Initialize with random weights (for testing only)
print(f"Initializing model with {num_classes} classes: {class_names}")

# Save the model
model_path = 'models/cattle_disease_vit_model.pth'
torch.save(model.state_dict(), model_path)
print(f"✅ Mock model saved to {model_path}")
print(f"⚠️  WARNING: This is a MOCK model with random weights!")
print(f"⚠️  It will give random predictions - NOT accurate!")
print(f"⚠️  Replace with your trained model for production use.")
