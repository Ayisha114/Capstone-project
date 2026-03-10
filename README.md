---
title: Cattle Disease Detector
emoji: "🐄"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 5000
---
# 🐄 Cattle Disease Detection System - Premium Edition

![Cattle Health AI](https://img.shields.io/badge/AI-Powered-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)
![Multi-Language](https://img.shields.io/badge/Languages-4-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

An **advanced AI-powered web application** for detecting cattle diseases using Vision Transformer (ViT) deep learning models. Upload an image of your cattle and get instant diagnosis with confidence scores, treatment recommendations, and multi-language support.

---

## 🌟 **PREMIUM FEATURES**

### ✨ **NEW in v2.0**

#### 🌍 **Multi-Language Support**
- **4 Languages**: English, Hindi (हिंदी), Tamil (தமிழ்), Kannada (ಕನ್ನಡ)
- Real-time language switching
- Localized treatment recommendations
- Native script support

#### 💊 **Treatment Recommendations**
- Instant treatment suggestions for detected diseases
- Language-specific medical advice
- Emergency protocols for critical conditions
- Veterinary contact integration

#### 📄 **PDF Report Generation**
- Professional PDF reports with diagnosis details
- Multi-language report support
- Downloadable and shareable
- Includes treatment recommendations

#### 📊 **Advanced Admin Dashboard**
- **Interactive Charts**: Bar, Pie, Doughnut, Line charts
- **Real-time Analytics**: Disease distribution, trends, patterns
- **Monthly/Daily Reports**: Track diagnostics over time
- **Search & Filter**: Find specific reports instantly
- **Multi-chart Views**: Switch between visualization types

#### 🎨 **Stunning UI/UX**
- Professional Unsplash hero images
- Animated statistics counters
- Disease gallery with real cattle images
- Smooth transitions and hover effects
- Mobile-responsive design

---

## 🚀 **Quick Start**

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shashankxou/cattle-disease-detector.git
cd cattle-disease-detector
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your trained model**
   - Place your trained `cattle_disease_vit_model.pth` file in the `models/` directory
   - Update `models/class_names.json` with your disease classes
   - Update `models/model_config.json` with your model configuration

   **OR create a mock model for testing:**
   ```bash
   python create_mock_model.py
   ```

5. **Run the application**
```bash
python app.py
```

6. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`
   - Admin login: username `admin`, password `admin123`

---

## 🚀 **Deploy on Render (Docker)**

1. **Push to GitHub**
    - Commit your changes and push this repository to GitHub.

2. **Create a Render Web Service**
    - Go to Render → **New** → **Web Service**.
    - Connect your GitHub repo.

3. **Configure Docker Deployment**
    - Render will detect the Docker setup automatically (see `render.yaml`).
    - Ensure the service uses the Dockerfile at `./Dockerfile`.

4. **Set Environment Variables**
    - `PORT` is set automatically by Render, but `render.yaml` includes a fallback.
    - (Optional) Set `SECRET_KEY` in Render for production.

5. **Deploy**
    - Click **Create Web Service**.
    - Render will build the Docker image and deploy the app.

**Note:** Uploads are stored in the container file system. For persistent storage on Render, use a persistent disk or external storage (S3, etc.).

---

## 📁 **Project Structure**

```
cattle-disease-detector/
├── app.py                          # Main Flask application (UPGRADED)
├── translations.json               # Multi-language translations (NEW)
├── requirements.txt                # Python dependencies
├── create_mock_model.py           # Mock model generator (NEW)
├── database.db                     # SQLite database (auto-created)
├── models/
│   ├── cattle_disease_vit_model.pth   # Trained model weights
│   ├── class_names.json               # Disease class names
│   ├── model_config.json              # Model configuration
│   └── README.md                      # Model instructions
├── static/
│   ├── css/
│   │   └── style.css               # Application styles
│   ├── js/
│   │   └── script.js               # JavaScript utilities
│   └── uploads/                    # Uploaded images (runtime)
└── templates/
    ├── base.html                   # Base template
    ├── home.html                   # Landing page (UPGRADED)
    ├── upload.html                 # Upload & diagnosis page
    ├── reports.html                # Reports listing
    └── admin.html                  # Admin dashboard (UPGRADED)
```

---

## 🎯 **Usage**

### For Farmers/Users

1. **Select Language**: Choose your preferred language (English/Hindi/Tamil/Kannada)
2. **Home Page**: Learn about the system and its features
3. **Diagnose**: Upload cattle images for instant disease detection
   - Drag & drop or browse to select image
   - Add optional metadata (Cattle ID, Location, Notes)
   - Get instant results with confidence scores
   - View treatment recommendations in your language
4. **Download PDF**: Get professional PDF reports
5. **Reports**: View all past diagnoses with filtering and search

### For Administrators

1. **Login**: Access admin dashboard with credentials
2. **Dashboard Features**:
   - **Statistics Cards**: Total reports, disease cases, healthy cattle, avg confidence
   - **Disease Distribution**: Bar/Pie/Doughnut charts (switchable)
   - **Timeline Charts**: Daily and monthly trends
   - **Recent Reports Table**: Search, filter, and manage reports
   - **Multi-language Interface**: Switch dashboard language
3. **Analytics**: Monitor system usage and disease patterns

---

## 🌍 **Multi-Language Support**

### Supported Languages

| Language | Code | Script | Status |
|----------|------|--------|--------|
| English | `en` | Latin | ✅ Complete |
| Hindi | `hi` | Devanagari | ✅ Complete |
| Tamil | `ta` | Tamil | ✅ Complete |
| Kannada | `kn` | Kannada | ✅ Complete |

### How to Use

**Frontend:**
```html
<!-- Language selector automatically appears on all pages -->
<select id="languageSelect">
    <option value="en">English</option>
    <option value="hi">हिंदी</option>
    <option value="ta">தமிழ்</option>
    <option value="kn">ಕನ್ನಡ</option>
</select>
```

**Backend:**
```python
# Get translation
from app import get_translation
text = get_translation('app_name', 'hi')  # Returns: "पशु रोग डिटेक्टर"
```

---

## 💊 **Treatment Recommendations**

Each detected disease comes with immediate treatment advice:

### Example Recommendations

**Foot-and-Mouth Disease:**
- English: "URGENT: Isolate immediately. Contact veterinarian..."
- Hindi: "तत्काल: तुरंत अलग करें। पशु चिकित्सक से संपर्क करें..."
- Tamil: "அவசரம்: உடனடியாக தனிமைப்படுத்தவும்..."
- Kannada: "ತುರ್ತು: ತಕ್ಷಣ ಪ್ರತ್ಯೇಕಿಸಿ..."

**Customize Recommendations:**
Edit `TREATMENT_RECOMMENDATIONS` in `app.py`:
```python
TREATMENT_RECOMMENDATIONS = {
    "Disease Name": {
        "en": "English treatment advice",
        "hi": "Hindi treatment advice",
        "ta": "Tamil treatment advice",
        "kn": "Kannada treatment advice"
    }
}
```

---

## 📊 **Admin Dashboard Features**

### Statistics Cards
- **Total Reports**: All-time diagnosis count
- **Disease Cases**: Non-healthy detections
- **Healthy Cattle**: Healthy detections
- **Average Confidence**: Model confidence score

### Interactive Charts

#### Disease Distribution Chart
- **Types**: Bar, Pie, Doughnut (switchable)
- **Data**: Cases per disease type
- **Colors**: Gradient color scheme

#### Timeline Charts
- **Daily Reports**: Last 30 days
- **Monthly Trends**: Last 12 months
- **Line Charts**: Smooth trend visualization

### Reports Table
- **Search**: Real-time filtering
- **Columns**: ID, Image, Diagnosis, Confidence, Cattle ID, Location, Date
- **Actions**: View, Download PDF, Delete

---

## 📄 **PDF Report Generation**

Generate professional PDF reports with:
- Multi-language support
- Diagnosis details
- Confidence scores
- Treatment recommendations
- Cattle metadata

**Usage:**
```python
# Automatic download link in report view
<a href="/download_pdf/{{ report_id }}">Download PDF</a>
```

---

## 🔧 **Configuration**

### Model Configuration

Edit `models/model_config.json`:
```json
{
    "image_size": 224,
    "model_type": "vit_b_16",
    "num_classes": 4,
    "pretrained": false
}
```

### Class Names

Edit `models/class_names.json`:
```json
["Healthy", "Foot-and-Mouth Disease", "Lumpy Skin Disease", "Mastitis"]
```

### Environment Variables

Set these for production:
```bash
export SECRET_KEY="your-secret-key-here"
export FLASK_ENV="production"
```

---

## 🚀 **Deployment**

### Deploy to Railway

```bash
railway login
railway init
railway up
```

### Deploy to Heroku

```bash
heroku create cattle-disease-detector
git push heroku main
```

### Deploy to Docker

```bash
docker build -t cattle-disease-detector .
docker run -p 5000:5000 cattle-disease-detector
```

---

## 📊 **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/upload` | GET/POST | Upload & diagnosis |
| `/reports` | GET | View all reports |
| `/report/<id>` | GET | Get specific report |
| `/download_pdf/<id>` | GET | Download PDF report |
| `/admin` | GET | Admin dashboard |
| `/admin/login` | POST | Admin authentication |
| `/api/stats` | GET | Get statistics (JSON) |
| `/set_language/<lang>` | GET | Change language |

---

## 🛠️ **Technology Stack**

- **Backend**: Flask 3.0.0
- **ML Framework**: PyTorch 2.1.0, TorchVision 0.16.0
- **Model**: Vision Transformer (ViT-B/16)
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Charts**: Chart.js 4.4.0
- **PDF Generation**: ReportLab 4.0.7
- **Icons**: Font Awesome 6.4.0
- **Fonts**: Google Fonts (Inter)
- **Images**: Unsplash API

---

## 🎨 **Design Features**

### Hero Section
- Professional Unsplash cattle images
- Animated statistics counters
- Multi-language taglines
- Call-to-action buttons

### Disease Gallery
- Real cattle images from Unsplash
- Disease descriptions
- Hover effects

### Admin Dashboard
- Gradient stat cards
- Interactive Chart.js visualizations
- Real-time search
- Responsive grid layout

---

## 🔒 **Security**

- Password hashing with SHA-256
- Session-based authentication
- CSRF protection
- File upload validation
- SQL injection prevention
- XSS protection

---

## 📝 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 **Support**

For support, email support@cattlehealth.ai or open an issue on GitHub.

---

## 🙏 **Acknowledgments**

- Vision Transformer (ViT) paper by Dosovitskiy et al.
- PyTorch team for the excellent framework
- Flask community for the web framework
- Unsplash for beautiful cattle images
- Chart.js for interactive visualizations
- All contributors and testers

---

## 📈 **Roadmap**

- [x] Multi-language support (4 languages)
- [x] Treatment recommendations
- [x] PDF report generation
- [x] Advanced admin charts
- [x] Unsplash image integration
- [ ] WhatsApp/SMS notifications
- [ ] Voice input for farmers
- [ ] Mobile app (iOS/Android)
- [ ] Real-time video analysis
- [ ] Integration with veterinary systems
- [ ] Batch processing
- [ ] Email notifications
- [ ] Offline mode support
- [ ] Veterinary locator map

---

## 🌟 **What Makes This Special**

1. **True Multi-Language**: Not just UI translation - treatment advice, reports, everything
2. **Professional Design**: Unsplash images, Chart.js, modern gradients
3. **Farmer-Friendly**: Simple interface, visual feedback, instant results
4. **Admin Power**: Multiple chart types, search, trends, analytics
5. **Production-Ready**: PDF reports, security, error handling
6. **Extensible**: Easy to add languages, diseases, features

---

## 📸 **Screenshots**

### Home Page
- Beautiful hero section with Unsplash cattle images
- Animated statistics
- Multi-language selector
- Disease gallery

### Admin Dashboard
- Interactive charts (Bar/Pie/Doughnut)
- Monthly trends
- Real-time search
- Gradient stat cards

### Diagnosis Page
- Drag & drop upload
- Instant results
- Treatment recommendations
- PDF download

---

**Made with ❤️ for farmers and cattle health professionals worldwide**

**Version**: 2.0.0 Premium Edition  
**Last Updated**: January 2026  
**Maintainer**: Shashank (@Shashankxou)

---

🐄 **Protect Your Cattle with AI Technology** 🐄

