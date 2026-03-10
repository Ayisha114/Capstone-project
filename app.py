import os
import json
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
from PIL import Image
import hashlib
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import threading

# Keep thread usage low to reduce memory/CPU spikes on small instances
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

torch = None
transforms = None
models = None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load translations
with open('translations.json', 'r', encoding='utf-8') as f:
    translations = json.load(f)

# Load model configuration
with open('models/model_config.json', 'r') as f:
    model_config = json.load(f)

with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

# Treatment recommendations database
TREATMENT_RECOMMENDATIONS = {
    "Healthy": {
        "en": "Your cattle is healthy! Continue regular care and monitoring.",
        "hi": "à¤†à¤ªà¤•à¤¾ à¤ªà¤¶à¥ à¤¸à¥à¤µà¤¸à¥à¤¥ à¤¹à¥ˆ! à¤¨à¤¿à¤¯à¤®à¤¿à¤¤ à¤¦à¥‡à¤–à¤­à¤¾à¤² à¤”à¤° à¤¨à¤¿à¤—à¤°à¤¾à¤¨à¥€ à¤œà¤¾à¤°à¥€ à¤°à¤–à¥‡à¤‚à¥¤",
        "ta": "à®‰à®™à¯à®•à®³à¯ à®•à®¾à®²à¯à®¨à®Ÿà¯ˆ à®†à®°à¯‹à®•à¯à®•à®¿à®¯à®®à®¾à®• à®‰à®³à¯à®³à®¤à¯! à®µà®´à®•à¯à®•à®®à®¾à®© à®ªà®°à®¾à®®à®°à®¿à®ªà¯à®ªà¯ à®®à®±à¯à®±à¯à®®à¯ à®•à®£à¯à®•à®¾à®£à®¿à®ªà¯à®ªà¯ˆà®¤à¯ à®¤à¯Šà®Ÿà®°à®µà¯à®®à¯.",
        "kn": "à²¨à²¿à²®à³à²® à²œà²¾à²¨à³à²µà²¾à²°à³ à²†à²°à³‹à²—à³à²¯à²•à²°à²µà²¾à²—à²¿à²¦à³†! à²¨à²¿à²¯à²®à²¿à²¤ à²†à²°à³ˆà²•à³† à²®à²¤à³à²¤à³ à²®à³‡à²²à³à²µà²¿à²šà²¾à²°à²£à³†à²¯à²¨à³à²¨à³ à²®à³à²‚à²¦à³à²µà²°à²¿à²¸à²¿."
    },
    "Diseased": {
        "en": "Your cattle shows signs of illness. Isolate the animal, monitor symptoms, and contact a veterinarian for diagnosis and treatment.",
        "hi": "à¤†à¤ªà¤•à¥‡ à¤ªà¤¶à¥ à¤®à¥‡à¤‚ à¤¬à¥€à¤®à¤¾à¤°à¥€ à¤•à¥‡ à¤²à¤•à¥à¤·à¤£ à¤¹à¥ˆà¤‚à¥¤ à¤ªà¤¶à¥ à¤•à¥‹ à¤…à¤²à¤— à¤•à¤°à¥‡à¤‚, à¤²à¤•à¥à¤·à¤£à¥‹à¤‚ à¤•à¥€ à¤¨à¤¿à¤—à¤°à¤¾à¤¨à¥€ à¤•à¤°à¥‡à¤‚ à¤”à¤° à¤‰à¤ªà¤šà¤¾à¤° à¤•à¥‡ à¤²à¤¿à¤ à¤ªà¤¶à¥ à¤šà¤¿à¤•à¤¿à¤¤à¥à¤¸à¤• à¤¸à¥‡ à¤¸à¤‚à¤ªà¤°à¥à¤• à¤•à¤°à¥‡à¤‚à¥¤",
        "ta": "à®‰à®™à¯à®•à®³à¯ à®•à®¾à®²à¯à®¨à®Ÿà¯ˆà®¯à®¿à®²à¯ à®¨à¯‹à®¯à¯ à®…à®±à®¿à®•à¯à®±à®¿à®•à®³à¯ à®‰à®³à¯à®³à®©. à®•à®¾à®²à¯à®¨à®Ÿà¯ˆà®¯à¯ˆ à®¤à®©à®¿à®®à¯ˆà®ªà¯à®ªà®Ÿà¯à®¤à¯à®¤à®¿ à®…à®±à®¿à®•à¯à®±à®¿à®•à®³à¯ˆ à®•à®£à¯à®•à®¾à®£à®¿à®¤à¯à®¤à¯ à®µà®¿à®²à®™à¯à®•à®¿à®¯à®²à¯ à®®à®°à¯à®¤à¯à®¤à¯à®µà®°à¯ˆ à®¤à¯Šà®Ÿà®°à¯à®ªà¯ à®•à¯Šà®³à¯à®³à®µà¯à®®à¯.",
        "kn": "à²¨à²¿à²®à³à²® à²œà²¾à²¨à³à²µà²¾à²°à²¿à²¨à²²à³à²²à²¿ à²°à³‹à²— à²²à²•à³à²·à²£à²—à²³à³ à²•à²‚à²¡à³à²¬à²°à³à²¤à³à²¤à²µà³†. à²œà²¾à²¨à³à²µà²¾à²°à²¨à³à²¨à³ à²ªà³à²°à²¤à³à²¯à³‡à²•à²¿à²¸à²¿, à²²à²•à³à²·à²£à²—à²³à²¨à³à²¨à³ à²—à²®à²¨à²¿à²¸à²¿ à²®à²¤à³à²¤à³ à²ªà²¶à³à²µà³ˆà²¦à³à²¯à²°à²¨à³à²¨à³ à²¸à²‚à²ªà²°à³à²•à²¿à²¸à²¿."
    },
    "Foot-and-Mouth Disease": {
        "en": "URGENT: Isolate immediately. Contact veterinarian. Provide soft feed and clean water. Disinfect area.",
        "hi": "à¤¤à¤¤à¥à¤•à¤¾à¤²: à¤¤à¥à¤°à¤‚à¤¤ à¤…à¤²à¤— à¤•à¤°à¥‡à¤‚à¥¤ à¤ªà¤¶à¥ à¤šà¤¿à¤•à¤¿à¤¤à¥à¤¸à¤• à¤¸à¥‡ à¤¸à¤‚à¤ªà¤°à¥à¤• à¤•à¤°à¥‡à¤‚à¥¤ à¤¨à¤°à¤® à¤šà¤¾à¤°à¤¾ à¤”à¤° à¤¸à¤¾à¤« à¤ªà¤¾à¤¨à¥€ à¤¦à¥‡à¤‚à¥¤ à¤•à¥à¤·à¥‡à¤¤à¥à¤° à¤•à¥‹ à¤•à¥€à¤Ÿà¤¾à¤£à¥à¤°à¤¹à¤¿à¤¤ à¤•à¤°à¥‡à¤‚à¥¤",
        "ta": "à®…à®µà®šà®°à®®à¯: à®‰à®Ÿà®©à®Ÿà®¿à®¯à®¾à®• à®¤à®©à®¿à®®à¯ˆà®ªà¯à®ªà®Ÿà¯à®¤à¯à®¤à®µà¯à®®à¯. à®•à®¾à®²à¯à®¨à®Ÿà¯ˆ à®®à®°à¯à®¤à¯à®¤à¯à®µà®°à¯ˆ à®¤à¯Šà®Ÿà®°à¯à®ªà¯ à®•à¯Šà®³à¯à®³à®µà¯à®®à¯. à®®à¯†à®©à¯à®®à¯ˆà®¯à®¾à®© à®‰à®£à®µà¯ à®®à®±à¯à®±à¯à®®à¯ à®šà¯à®¤à¯à®¤à®®à®¾à®© à®¤à®£à¯à®£à¯€à®°à¯ à®µà®´à®™à¯à®•à®µà¯à®®à¯.",
        "kn": "à²¤à³à²°à³à²¤à³: à²¤à²•à³à²·à²£ à²ªà³à²°à²¤à³à²¯à³‡à²•à²¿à²¸à²¿. à²ªà²¶à³à²µà³ˆà²¦à³à²¯à²°à²¨à³à²¨à³ à²¸à²‚à²ªà²°à³à²•à²¿à²¸à²¿. à²®à³ƒà²¦à³à²µà²¾à²¦ à²†à²¹à²¾à²° à²®à²¤à³à²¤à³ à²¶à³à²¦à³à²§ à²¨à³€à²°à³ à²’à²¦à²—à²¿à²¸à²¿."
    },
    
}

# Precaution recommendations database
PRECAUTION_RECOMMENDATIONS = {
    "Healthy": {
        "en": "Maintain hygiene, balanced nutrition, regular vaccinations, and routine health checks.",
        "hi": "à¤¸à¥à¤µà¤šà¥à¤›à¤¤à¤¾, à¤¸à¤‚à¤¤à¥à¤²à¤¿à¤¤ à¤ªà¥‹à¤·à¤£, à¤¨à¤¿à¤¯à¤®à¤¿à¤¤ à¤Ÿà¥€à¤•à¤¾à¤•à¤°à¤£ à¤”à¤° à¤¨à¤¿à¤¯à¤®à¤¿à¤¤ à¤¸à¥à¤µà¤¾à¤¸à¥à¤¥à¥à¤¯ à¤œà¤¾à¤‚à¤š à¤¬à¤¨à¤¾à¤ à¤°à¤–à¥‡à¤‚à¥¤",
        "ta": "à®šà¯à®¤à¯à®¤à®®à¯, à®šà®®à®¨à®¿à®²à¯ˆà®¯à®¾à®© à®‰à®£à®µà¯, à®µà®´à®•à¯à®•à®®à®¾à®© à®¤à®Ÿà¯à®ªà¯à®ªà¯‚à®šà®¿à®•à®³à¯ à®®à®±à¯à®±à¯à®®à¯ à®®à¯à®±à¯ˆà®¯à®¾à®© à®‰à®Ÿà®²à¯à®¨à®² à®šà¯‹à®¤à®©à¯ˆà®•à®³à¯ˆ à®ªà¯‡à®£à¯à®™à¯à®•à®³à¯.",
        "kn": "à²¸à³à²µà²šà³à²šà²¤à³†, à²¸à²®à²¤à³‹à²²à²¿à²¤ à²ªà³‹à²·à²£à³†, à²¨à²¿à²¯à²®à²¿à²¤ à²²à²¸à²¿à²•à³† à²®à²¤à³à²¤à³ à²¨à²¿à²¯à²®à²¿à²¤ à²†à²°à³‹à²—à³à²¯ à²¤à²ªà²¾à²¸à²£à³†à²—à²³à²¨à³à²¨à³ à¦¬à¦œà²¾à²¯à²¿à²¸à²¿."
    },
    "Diseased": {
        "en": "Isolate the animal, avoid shared water/feed, disinfect tools, and limit movement until vet advice.",
        "hi": "à¤ªà¤¶à¥ à¤•à¥‹ à¤…à¤²à¤— à¤°à¤–à¥‡à¤‚, à¤¸à¤¾à¤à¤¾ à¤ªà¤¾à¤¨à¥€/à¤šà¤¾à¤°à¤¾ à¤¨ à¤¦à¥‡à¤‚, à¤‰à¤ªà¤•à¤°à¤£ à¤•à¥€à¤Ÿà¤¾à¤£à¥à¤°à¤¹à¤¿à¤¤ à¤•à¤°à¥‡à¤‚ à¤”à¤° à¤ªà¤¶à¥ à¤šà¤¿à¤•à¤¿à¤¤à¥à¤¸à¤• à¤•à¥€ à¤¸à¤²à¤¾à¤¹ à¤¤à¤• à¤†à¤µà¤¾à¤œà¤¾à¤¹à¥€ à¤¸à¥€à¤®à¤¿à¤¤ à¤°à¤–à¥‡à¤‚à¥¤",
        "ta": "à®•à®¾à®²à¯à®¨à®Ÿà¯ˆà®¯à¯ˆ à®¤à®©à®¿à®®à¯ˆà®ªà¯à®ªà®Ÿà¯à®¤à¯à®¤à®¿, à®¨à¯€à®°à¯/à®‰à®£à®µà¯ˆ à®ªà®•à®¿à®° à®µà¯‡à®£à¯à®Ÿà®¾à®®à¯, à®•à®°à¯à®µà®¿à®•à®³à¯ˆ à®•à®¿à®°à¯à®®à®¿à®¨à®¾à®šà®¿à®©à®¿ à®šà¯†à®¯à¯à®¯à®µà¯à®®à¯, à®®à®°à¯à®¤à¯à®¤à¯à®µà®°à¯ à®†à®²à¯‹à®šà®©à¯ˆ à®µà®°à¯ˆ à®‡à®Ÿà®®à®¾à®±à¯à®±à®®à¯ à®•à®Ÿà¯à®Ÿà¯à®ªà¯à®ªà®Ÿà¯à®¤à¯à®¤à®µà¯à®®à¯.",
        "kn": "à²œà²¾à²¨à³à²µà²¾à²°à²¨à³à²¨à³ à²ªà³à²°à²¤à³à²¯à³‡à²•à²¿à²¸à²¿, à²¨à³€à²°à³/à²®à³‡à²µà³ à²¹à²‚à²šà²¿à²•à³Šà²³à³à²³à²¬à³‡à²¡à²¿, à²¸à²¾à²§à²¨à²—à²³à²¨à³à²¨à³ à²¨à²‚à²œà³à²¨à²¿à²°à³‹à²§à²• à²®à²¾à²¡à²¿ à²®à²¤à³à²¤à³ à²ªà²¶à³à²µà³ˆà²¦à³à²¯à²° à²¸à²²à²¹à³†à²¯à²µà²°à³†à²—à³‚ à²šà²²à²¨à²µà²²à²¨à²µà²¨à³à²¨à³ à²¨à²¿à²¯à²‚à²¤à³à²°à²¿à²¸à²¿."
    },
    "Foot-and-Mouth Disease": {
        "en": "Quarantine affected cattle, restrict farm visitors, disinfect footwear/equipment, and notify veterinary services.",
        "hi": "à¤ªà¥à¤°à¤­à¤¾à¤µà¤¿à¤¤ à¤ªà¤¶à¥à¤“à¤‚ à¤•à¥‹ à¤•à¥à¤µà¤¾à¤°à¤‚à¤Ÿà¥€à¤¨ à¤•à¤°à¥‡à¤‚, à¤«à¤¾à¤°à¥à¤® à¤ªà¤° à¤†à¤—à¤‚à¤¤à¥à¤•à¥‹à¤‚ à¤•à¥‹ à¤¸à¥€à¤®à¤¿à¤¤ à¤•à¤°à¥‡à¤‚, à¤œà¥‚à¤¤à¥‹à¤‚/à¤‰à¤ªà¤•à¤°à¤£à¥‹à¤‚ à¤•à¥‹ à¤•à¥€à¤Ÿà¤¾à¤£à¥à¤°à¤¹à¤¿à¤¤ à¤•à¤°à¥‡à¤‚ à¤”à¤° à¤ªà¤¶à¥ à¤šà¤¿à¤•à¤¿à¤¤à¥à¤¸à¤¾ à¤¸à¥‡à¤µà¤¾à¤“à¤‚ à¤•à¥‹ à¤¸à¥‚à¤šà¤¿à¤¤ à¤•à¤°à¥‡à¤‚à¥¤",
        "ta": "à®ªà®¾à®¤à®¿à®•à¯à®•à®ªà¯à®ªà®Ÿà¯à®Ÿ à®•à®¾à®²à¯à®¨à®Ÿà¯ˆà®•à®³à¯ˆ à®¤à®©à®¿à®®à¯ˆà®ªà¯à®ªà®Ÿà¯à®¤à¯à®¤à®¿, à®ªà®£à¯à®£à¯ˆ à®µà®°à¯à®•à¯ˆà®•à®³à¯ˆ à®•à®Ÿà¯à®Ÿà¯à®ªà¯à®ªà®Ÿà¯à®¤à¯à®¤à®¿, à®•à®¾à®²à®£à®¿à®•à®³à¯/à®•à®°à¯à®µà®¿à®•à®³à¯ˆ à®•à®¿à®°à¯à®®à®¿à®¨à®¾à®šà®¿à®©à®¿ à®šà¯†à®¯à¯à®¤à¯ à®•à®¾à®²à¯à®¨à®Ÿà¯ˆ à®šà¯‡à®µà¯ˆà®•à®³à¯ˆ à®…à®±à®¿à®µà®¿à®•à¯à®•à®µà¯à®®à¯.",
        "kn": "à²¬à²¾à²§à²¿à²¤ à²œà²¾à²¨à³à²µà²¾à²°à²¨à³à²¨à³ à²•à³à²µà²¾à²°à²‚à²Ÿà³ˆà²¨à³ à²®à²¾à²¡à²¿, à²«à²¾à²°à³à²®à³ à²­à³‡à²Ÿà²¿ à²¨à²¿à²°à³à²¬à²‚à²§à²¿à²¸à²¿, à²ªà²¾à²¦à²°à²•à³à²·à³†/à²‰à²ªà²•à²°à²£à²—à²³à²¨à³à²¨à³ à²¨à²‚à²œà³à²¨à²¿à²°à³‹à²§à²• à²®à²¾à²¡à²¿ à²®à²¤à³à²¤à³ à²ªà²¶à³ à²µà³ˆà²¦à³à²¯à²•à³€à²¯ à²¸à³‡à²µà³†à²—à²³à²¿à²—à³† à²¤à²¿à²³à²¿à²¸à²¿."
    },
    
}

def normalize_prediction(label):
    """Normalize model label for display and reporting."""
    if not isinstance(label, str):
        return label
    normalized = label.strip().lower()
    if normalized == 'healthy':
        return 'Healthy'
    if normalized in {'diseased', 'disease', 'sick', 'unhealthy'}:
        return 'Diseased'
    return label

# Load the trained model lazily to keep startup fast on free instances
MODEL_BACKEND = None
MODEL_LOADED = False
MODEL_LOAD_ERROR = None
model = None
device = None
transform = None
_model_lock = threading.Lock()


def load_model():
    global MODEL_BACKEND, MODEL_LOADED, MODEL_LOAD_ERROR, model, device, transform
    global torch, transforms, models

    if MODEL_LOADED:
        return True
    if MODEL_LOAD_ERROR:
        return False

    with _model_lock:
        if MODEL_LOADED:
            return True
        if MODEL_LOAD_ERROR:
            return False

        try:
            import torch  # noqa: F401
            import torchvision.transforms as transforms  # noqa: F401
            from torchvision import models  # noqa: F401
        except Exception as e:
            MODEL_LOAD_ERROR = f"torch/torchvision import failed: {e}"
            print(f"WARNING: {MODEL_LOAD_ERROR}")
            return False

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model_path = os.environ.get('MODEL_PATH')
            if not model_path:
                int8_path = os.path.join('models', 'cattle_disease_vit_model_int8.pth')
                model_path = int8_path if os.path.exists(int8_path) else os.path.join('models', 'cattle_disease_vit_model.pth')

            state_dict = torch.load(model_path, map_location=device)
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
                MODEL_BACKEND = 'transformers'
            else:
                model = models.vit_b_16(weights=None)
                model.heads = torch.nn.Linear(model.heads.head.in_features, num_classes)
                MODEL_BACKEND = 'torchvision'

            if model_path.endswith('_int8.pth'):
                # Quantized weights require a quantized model structure before loading.
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

            model.load_state_dict(state_dict, strict=True)
            model = model.to(device)
            model.eval()

            image_size = model_config.get('image_size', 224)
            if MODEL_BACKEND == 'transformers':
                normalize_mean = [0.5, 0.5, 0.5]
                normalize_std = [0.5, 0.5, 0.5]
            else:
                normalize_mean = [0.485, 0.456, 0.406]
                normalize_std = [0.229, 0.224, 0.225]

            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])

            MODEL_LOADED = True
            return True
        except Exception as e:
            MODEL_LOAD_ERROR = str(e)
            print(f"WARNING: Model not loaded - {MODEL_LOAD_ERROR}")
            MODEL_LOADED = False
            model = None
            transform = None
            try:
                device = torch.device('cpu')
            except Exception:
                device = None
            return False

# Database initialization
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Create reports table with language field
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT NOT NULL,
                  filepath TEXT NOT NULL,
                  prediction TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  notes TEXT,
                  cattle_id TEXT,
                  location TEXT,
                  language TEXT DEFAULT 'en')''')
    
    # Create users table for admin
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  role TEXT DEFAULT 'user')''')
    
    # Create default admin if not exists
    admin_hash = hashlib.sha256('admin123'.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                  ('admin', admin_hash, 'admin'))
    except sqlite3.IntegrityError:
        pass
    
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_translation(key, lang='en'):
    """Get translated text"""
    return translations.get(lang, {}).get(key, translations['en'].get(key, key))

def get_treatment_recommendation(disease, lang='en'):
    """Get treatment recommendation for disease"""
    return TREATMENT_RECOMMENDATIONS.get(disease, {}).get(lang, "Consult a veterinarian for proper treatment.")

def get_precaution_recommendation(disease, lang='en'):
    """Get precaution recommendation for disease"""
    return PRECAUTION_RECOMMENDATIONS.get(disease, {}).get(lang, "Follow biosecurity precautions and consult a veterinarian.")

def hash_identifier(value):
    """Create a stable, short hash for identifiers shown in UI."""
    if value is None:
        return ''
    value_str = str(value).strip()
    if not value_str:
        return ''
    salted = f"{value_str}|{app.config['SECRET_KEY']}"
    return hashlib.sha256(salted.encode('utf-8')).hexdigest()[:10]

def predict_image(image_path):
    """Predict disease from image"""
    if not load_model():
        message = 'Model not loaded. Please add trained model file.'
        if MODEL_LOAD_ERROR:
            message = f"Model not loaded. {MODEL_LOAD_ERROR}"
        return {'error': message}
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if MODEL_BACKEND == 'transformers':
                outputs = model(pixel_values=image_tensor)
                logits = outputs.logits
            else:
                logits = model(image_tensor)

            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = normalize_prediction(class_names[predicted.item()])
        confidence_score = confidence.item() * 100
        
        # Get all class probabilities
        all_probs = {normalize_prediction(class_names[i]): float(probabilities[0][i] * 100) 
                     for i in range(len(class_names))}
        
        return {
            'prediction': predicted_class,
            'confidence': round(confidence_score, 2),
            'all_probabilities': all_probs
        }
    except Exception as e:
        return {'error': str(e)}

def generate_pdf_report(report_data, lang='en'):
    """Generate PDF report"""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    p.setFont("Helvetica-Bold", 24)
    p.drawString(50, height - 50, get_translation('app_name', lang))
    
    # Report details
    p.setFont("Helvetica", 12)
    y = height - 100
    
    p.drawString(50, y, f"{get_translation('date', lang)}: {report_data['timestamp']}")
    y -= 30
    cattle_id_display = report_data.get('cattle_id_hash') or report_data.get('cattle_id') or 'N/A'
    p.drawString(50, y, f"{get_translation('cattle_id', lang)}: {cattle_id_display}")
    y -= 30
    p.drawString(50, y, f"{get_translation('location', lang)}: {report_data.get('location', 'N/A')}")
    y -= 30
    
    # Prediction
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, f"{get_translation('prediction', lang)}:")
    p.setFont("Helvetica", 14)
    p.drawString(200, y, report_data['prediction'])
    y -= 30
    
    # Confidence
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, f"{get_translation('confidence', lang)}:")
    p.setFont("Helvetica", 14)
    p.drawString(200, y, f"{report_data['confidence']}%")
    y -= 50
    
    # Treatment
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, f"{get_translation('treatment', lang)}:")
    y -= 20
    p.setFont("Helvetica", 10)
    treatment = get_treatment_recommendation(report_data['prediction'], lang)

    # Wrap text
    words = treatment.split()
    line = ""
    for word in words:
        if len(line + word) < 80:
            line += word + " "
        else:
            p.drawString(50, y, line)
            y -= 15
            line = word + " "
    if line:
        p.drawString(50, y, line)

    y -= 25
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, f"{get_translation('precaution', lang)}:")
    y -= 20
    p.setFont("Helvetica", 10)
    precaution = get_precaution_recommendation(report_data['prediction'], lang)

    words = precaution.split()
    line = ""
    for word in words:
        if len(line + word) < 80:
            line += word + " "
        else:
            p.drawString(50, y, line)
            y -= 15
            line = word + " "
    if line:
        p.drawString(50, y, line)
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL_LOADED,
        'model_error': MODEL_LOAD_ERROR
    }), 200


@app.route('/')
def home():
    lang = request.args.get('lang', 'en')
    return render_template('home.html', lang=lang, t=lambda k: get_translation(k, lang))

@app.route('/set_language/<lang>')
def set_language(lang):
    session['language'] = lang
    return redirect(request.referrer or url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    lang = session.get('language', 'en')
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': get_translation('error_upload', lang)}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': get_translation('error_upload', lang)}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            db_filepath = filepath.replace('\\', '/')
            
            # Predict
            result = predict_image(filepath)
            
            if 'error' in result:
                os.remove(filepath)
                return jsonify({'error': result['error']}), 500
            
            # Get treatment recommendation
            treatment = get_treatment_recommendation(result['prediction'], lang)
            precaution = get_precaution_recommendation(result['prediction'], lang)
            
            # Save to database
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("""INSERT INTO reports (filename, filepath, prediction, confidence, 
                         cattle_id, location, notes, language) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                      (filename, db_filepath, result['prediction'], result['confidence'],
                       request.form.get('cattle_id', ''),
                       request.form.get('location', ''),
                       request.form.get('notes', ''),
                       lang))
            report_id = c.lastrowid
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'report_id': report_id,
                'report_id_hash': hash_identifier(report_id),
                'filename': filename,
                'filepath': db_filepath,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'all_probabilities': result['all_probabilities'],
                'treatment': treatment,
                'precaution': precaution,
                'cattle_id_hash': hash_identifier(request.form.get('cattle_id', ''))
            })
        
        return jsonify({'error': get_translation('error_upload', lang)}), 400
    
    return render_template('upload.html', lang=lang, t=lambda k: get_translation(k, lang))

@app.route('/reports')
def reports():
    lang = session.get('language', 'en')
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""SELECT id, filename, filepath, prediction, confidence, 
                 timestamp, cattle_id, location, notes, language
                 FROM reports ORDER BY timestamp DESC LIMIT 100""")
    reports_data = c.fetchall()
    conn.close()
    
    reports_list = []
    for row in reports_data:
        web_path = row[2].replace('\\', '/')
        if web_path.startswith('static/'):
            web_path = '/' + web_path
        reports_list.append({
            'id': row[0],
            'id_hash': hash_identifier(row[0]),
            'filename': row[1],
            'filepath': row[2],
            'web_path': web_path,
            'prediction': row[3],
            'confidence': row[4],
            'timestamp': row[5],
            'cattle_id': row[6],
            'cattle_id_hash': hash_identifier(row[6]),
            'location': row[7],
            'notes': row[8],
            'language': row[9] if len(row) > 9 else 'en'
        })
    
    return render_template('reports.html', reports=reports_list, lang=lang, t=lambda k: get_translation(k, lang))

@app.route('/report/<int:report_id>')
def report_detail(report_id):
    lang = session.get('language', 'en')
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""SELECT id, filename, filepath, prediction, confidence, 
                 timestamp, cattle_id, location, notes, language
                 FROM reports WHERE id = ?""", (report_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        web_path = row[2].replace('\\', '/')
        if web_path.startswith('static/'):
            web_path = '/' + web_path
        report = {
            'id': row[0],
            'id_hash': hash_identifier(row[0]),
            'filename': row[1],
            'filepath': row[2],
            'web_path': web_path,
            'prediction': row[3],
            'confidence': row[4],
            'timestamp': row[5],
            'cattle_id': row[6],
            'cattle_id_hash': hash_identifier(row[6]),
            'location': row[7],
            'notes': row[8],
            'language': row[9] if len(row) > 9 else 'en',
            'treatment': get_treatment_recommendation(row[3], lang),
            'precaution': get_precaution_recommendation(row[3], lang)
        }
        return jsonify(report)
    return jsonify({'error': 'Report not found'}), 404

@app.route('/download_pdf/<int:report_id>')
def download_pdf(report_id):
    lang = session.get('language', 'en')
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""SELECT id, filename, filepath, prediction, confidence, 
                 timestamp, cattle_id, location, notes 
                 FROM reports WHERE id = ?""", (report_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        report_data = {
            'id': row[0],
            'filename': row[1],
            'filepath': row[2],
            'prediction': row[3],
            'confidence': row[4],
            'timestamp': row[5],
            'cattle_id': row[6],
            'cattle_id_hash': hash_identifier(row[6]),
            'location': row[7],
            'notes': row[8]
        }
        
        pdf_buffer = generate_pdf_report(report_data, lang)
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'cattle_report_{report_id}.pdf'
        )
    
    return jsonify({'error': 'Report not found'}), 404

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    lang = session.get('language', 'en')
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT id, role FROM users WHERE username = ? AND password_hash = ?",
                  (username, password_hash))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user[0]
            session['role'] = user[1]
            flash(get_translation('success_upload', lang), 'success')
            return redirect(url_for('admin'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('admin.html', login_page=True, lang=lang, t=lambda k: get_translation(k, lang))

@app.route('/admin')
def admin():
    lang = session.get('language', 'en')
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Get statistics
    c.execute("SELECT COUNT(*) FROM reports")
    total_reports = c.fetchone()[0]
    
    c.execute("SELECT prediction, COUNT(*) FROM reports GROUP BY prediction")
    disease_stats = c.fetchall()
    
    c.execute("""SELECT AVG(confidence) FROM reports 
                 WHERE LOWER(prediction) != 'healthy'""")
    avg_confidence = c.fetchone()[0] or 0
    
    # Get daily reports for chart
    c.execute("""SELECT DATE(timestamp) as date, COUNT(*) 
                 FROM reports 
                 GROUP BY DATE(timestamp) 
                 ORDER BY date DESC LIMIT 30""")
    daily_reports = c.fetchall()
    
    # Get monthly trends
    c.execute("""SELECT strftime('%Y-%m', timestamp) as month, COUNT(*) 
                 FROM reports 
                 GROUP BY month 
                 ORDER BY month DESC LIMIT 12""")
    monthly_reports = c.fetchall()
    
    conn.close()
    
    return render_template('admin.html', 
                          total_reports=total_reports,
                          disease_stats=disease_stats,
                          avg_confidence=round(avg_confidence, 2),
                          daily_reports=daily_reports,
                          monthly_reports=monthly_reports,
                          lang=lang,
                          t=lambda k: get_translation(k, lang))

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('home'))

@app.route('/api/stats')
def api_stats():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM reports")
    total = c.fetchone()[0]
    
    c.execute("SELECT prediction, COUNT(*) FROM reports GROUP BY prediction")
    by_disease = dict(c.fetchall())
    
    c.execute("""SELECT DATE(timestamp) as date, COUNT(*) 
                 FROM reports 
                 GROUP BY DATE(timestamp) 
                 ORDER BY date DESC LIMIT 30""")
    daily_reports = c.fetchall()
    
    c.execute("""SELECT strftime('%Y-%m', timestamp) as month, COUNT(*) 
                 FROM reports 
                 GROUP BY month 
                 ORDER BY month DESC LIMIT 12""")
    monthly_reports = c.fetchall()
    
    conn.close()
    
    return jsonify({
        'total_reports': total,
        'by_disease': by_disease,
        'daily_reports': daily_reports,
        'monthly_reports': monthly_reports
    })

@app.route('/api/reports')
def api_reports():
    try:
        limit = int(request.args.get('limit', 10))
    except ValueError:
        limit = 10

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""SELECT id, filename, filepath, prediction, confidence,
                 timestamp, cattle_id, location, notes, language
                 FROM reports ORDER BY timestamp DESC LIMIT ?""", (limit,))
    rows = c.fetchall()
    conn.close()

    reports_list = []
    for row in rows:
        web_path = row[2]
        if web_path.startswith('static/'):
            web_path = '/' + web_path
        reports_list.append({
            'id': row[0],
            'id_hash': hash_identifier(row[0]),
            'filename': row[1],
            'filepath': row[2],
            'web_path': web_path,
            'prediction': row[3],
            'confidence': row[4],
            'timestamp': row[5],
            'cattle_id': row[6],
            'cattle_id_hash': hash_identifier(row[6]),
            'location': row[7],
            'notes': row[8],
            'language': row[9] if len(row) > 9 else 'en'
        })

    return jsonify({'reports': reports_list})

@app.route('/delete_report/<int:report_id>', methods=['POST'])
def delete_report(report_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT filepath FROM reports WHERE id = ?", (report_id,))
    row = c.fetchone()
    
    if row:
        filepath = row[0]
        if os.path.exists(filepath):
            os.remove(filepath)
        
        c.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    
    conn.close()
    return jsonify({'error': 'Report not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
