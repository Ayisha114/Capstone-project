import json
import os
import torch
from torchvision import models

MODEL_IN = os.path.join('models', 'cattle_disease_vit_model.pth')
MODEL_OUT = os.path.join('models', 'cattle_disease_vit_model_int8.pth')

with open(os.path.join('models', 'model_config.json'), 'r') as f:
    model_config = json.load(f)

with open(os.path.join('models', 'class_names.json'), 'r') as f:
    class_names = json.load(f)

if not os.path.exists(MODEL_IN):
    raise FileNotFoundError(f'Model file not found: {MODEL_IN}')

state_dict = torch.load(MODEL_IN, map_location='cpu')
num_classes = len(class_names)

is_transformers = any(key.startswith('vit.') for key in state_dict.keys())

if is_transformers:
    from transformers import ViTConfig, ViTForImageClassification

    config = ViTConfig(
        num_labels=num_classes,
        image_size=model_config.get('image_size', 224),
        num_channels=3
    )
    model = ViTForImageClassification(config)
else:
    model = models.vit_b_16(weights=None)
    model.heads = torch.nn.Linear(model.heads.head.in_features, num_classes)

model.load_state_dict(state_dict, strict=True)
model.eval()

# Dynamic quantization to reduce memory on CPU
qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
torch.save(qmodel.state_dict(), MODEL_OUT)

print(f'Wrote quantized model: {MODEL_OUT}')
