# Cattle Disease Detection - Model Placeholder

⚠️ **IMPORTANT**: You need to add your trained model file here!

## Required Files

1. **cattle_disease_vit_model.pth** - Your trained Vision Transformer model weights
2. **class_names.json** - Already provided (update with your classes)
3. **model_config.json** - Already provided (update with your config)

## How to Add Your Model

### Option 1: Direct Upload
1. Place your trained `cattle_disease_vit_model.pth` file in this directory
2. Update `class_names.json` with your disease classes
3. Update `model_config.json` with your model configuration

### Option 2: Download from Cloud
If your model is stored in cloud storage:

```bash
# Example: Download from Google Drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=YOUR_FILE_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=YOUR_FILE_ID" -O cattle_disease_vit_model.pth && rm -rf /tmp/cookies.txt
```

### Option 3: Train Your Own Model
If you don't have a trained model yet, here's a basic training script:

```python
import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader

# Load Vision Transformer
model = models.vit_b_16(pretrained=True)
num_classes = 4  # Update based on your classes
model.heads = torch.nn.Linear(model.heads.head.in_features, num_classes)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your dataset
train_dataset = datasets.ImageFolder('path/to/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'cattle_disease_vit_model.pth')
print('Model saved successfully!')
```

## Model Configuration

### class_names.json
```json
["Healthy", "Foot-and-Mouth Disease", "Lumpy Skin Disease", "Mastitis"]
```
Update this with your actual disease classes.

### model_config.json
```json
{
    "image_size": 224,
    "model_type": "vit_b_16",
    "num_classes": 4,
    "pretrained": false
}
```
Update this with your model specifications.

## Model Requirements

- **Architecture**: Vision Transformer (ViT-B/16)
- **Input Size**: 224x224 pixels
- **Format**: PyTorch state dict (.pth)
- **Classes**: Must match class_names.json

## Testing Your Model

After adding your model, test it:

```python
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = models.vit_b_16(pretrained=False)
model.heads = torch.nn.Linear(model.heads.head.in_features, 4)
model.load_state_dict(torch.load('cattle_disease_vit_model.pth'))
model.eval()

# Test with an image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('test_image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1)
    print(f'Predicted class: {prediction.item()}')
```

## Need Help?

- Check the main README.md for more information
- Open an issue on GitHub
- Contact: support@cattlehealth.ai

---

**Note**: The application will not work without a trained model file. Make sure to add your model before running the application.
