# API Documentation

This document describes the API endpoints available in the Cattle Disease Detection System.

## Base URL

```
http://localhost:5000
```

## Authentication

Admin endpoints require session-based authentication. Login via `/admin/login` to access protected routes.

---

## Endpoints

### 1. Home Page

**GET** `/`

Returns the home page with system information and features.

**Response:** HTML page

---

### 2. Upload Page

**GET** `/upload`

Returns the upload interface for disease diagnosis.

**Response:** HTML page

---

### 3. Upload Image for Diagnosis

**POST** `/upload`

Upload a cattle image for disease detection.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file` (required): Image file (JPG, PNG, GIF, max 16MB)
  - `cattle_id` (optional): Cattle identification number
  - `location` (optional): Location information
  - `notes` (optional): Additional notes

**Response:**
```json
{
    "success": true,
    "report_id": 1,
    "filename": "20240109_143022_cattle.jpg",
    "filepath": "static/uploads/20240109_143022_cattle.jpg",
    "prediction": "Healthy",
    "confidence": 95.67,
    "all_probabilities": {
        "Healthy": 95.67,
        "Foot-and-Mouth Disease": 2.15,
        "Lumpy Skin Disease": 1.89,
        "Mastitis": 0.29
    }
}
```

**Error Response:**
```json
{
    "error": "No file uploaded"
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request (no file, invalid file type)
- `500`: Server error (prediction failed)

**Example (cURL):**
```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@cattle_image.jpg" \
  -F "cattle_id=COW-001" \
  -F "location=Farm A, Barn 2" \
  -F "notes=Showing symptoms"
```

**Example (Python):**
```python
import requests

url = "http://localhost:5000/upload"
files = {"file": open("cattle_image.jpg", "rb")}
data = {
    "cattle_id": "COW-001",
    "location": "Farm A, Barn 2",
    "notes": "Showing symptoms"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Example (JavaScript):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('cattle_id', 'COW-001');
formData.append('location', 'Farm A, Barn 2');
formData.append('notes', 'Showing symptoms');

fetch('http://localhost:5000/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### 4. View Reports

**GET** `/reports`

Returns a page displaying all diagnosis reports.

**Response:** HTML page with reports list

---

### 5. Get Report Details

**GET** `/report/<report_id>`

Get detailed information about a specific report.

**Parameters:**
- `report_id` (path): Report ID number

**Response:**
```json
{
    "id": 1,
    "filename": "20240109_143022_cattle.jpg",
    "filepath": "static/uploads/20240109_143022_cattle.jpg",
    "prediction": "Healthy",
    "confidence": 95.67,
    "timestamp": "2024-01-09 14:30:22",
    "cattle_id": "COW-001",
    "location": "Farm A, Barn 2",
    "notes": "Showing symptoms"
}
```

**Error Response:**
```json
{
    "error": "Report not found"
}
```

**Status Codes:**
- `200`: Success
- `404`: Report not found

**Example:**
```bash
curl http://localhost:5000/report/1
```

---

### 6. Get Statistics

**GET** `/api/stats`

Get system statistics including total reports, disease distribution, and daily reports.

**Response:**
```json
{
    "total_reports": 150,
    "by_disease": {
        "Healthy": 120,
        "Foot-and-Mouth Disease": 15,
        "Lumpy Skin Disease": 10,
        "Mastitis": 5
    },
    "daily_reports": [
        ["2024-01-09", 12],
        ["2024-01-08", 15],
        ["2024-01-07", 8]
    ]
}
```

**Example:**
```bash
curl http://localhost:5000/api/stats
```

---

### 7. Admin Login Page

**GET** `/admin/login`

Returns the admin login page.

**Response:** HTML login form

---

### 8. Admin Login

**POST** `/admin/login`

Authenticate as admin user.

**Request:**
- Content-Type: `application/x-www-form-urlencoded`
- Body:
  - `username`: Admin username
  - `password`: Admin password

**Response:** Redirect to admin dashboard on success

**Default Credentials:**
- Username: `admin`
- Password: `admin123`

**Example:**
```bash
curl -X POST http://localhost:5000/admin/login \
  -d "username=admin" \
  -d "password=admin123"
```

---

### 9. Admin Dashboard

**GET** `/admin`

Access the admin dashboard (requires authentication).

**Response:** HTML dashboard with statistics and charts

**Authentication:** Required (session-based)

---

### 10. Admin Logout

**GET** `/admin/logout`

Logout from admin session.

**Response:** Redirect to home page

---

### 11. Delete Report

**POST** `/delete_report/<report_id>`

Delete a specific report (admin only).

**Parameters:**
- `report_id` (path): Report ID to delete

**Response:**
```json
{
    "success": true
}
```

**Error Response:**
```json
{
    "error": "Unauthorized"
}
```

**Status Codes:**
- `200`: Success
- `403`: Unauthorized
- `404`: Report not found

**Authentication:** Required (admin)

---

## Error Handling

All endpoints return appropriate HTTP status codes and error messages:

**Common Error Responses:**

```json
{
    "error": "Error message description"
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad Request
- `403`: Forbidden
- `404`: Not Found
- `500`: Internal Server Error

---

## Rate Limiting

Currently, there are no rate limits. For production deployment, consider implementing rate limiting using Flask-Limiter.

---

## CORS

CORS is not enabled by default. To enable for cross-origin requests, add Flask-CORS:

```python
from flask_cors import CORS
CORS(app)
```

---

## Webhooks

Webhooks are not currently supported but can be added for:
- New diagnosis notifications
- Report updates
- System alerts

---

## SDK Examples

### Python SDK Example

```python
import requests
from typing import Dict, Optional

class CattleHealthAPI:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def diagnose(self, image_path: str, cattle_id: Optional[str] = None,
                 location: Optional[str] = None, notes: Optional[str] = None) -> Dict:
        """Upload image for diagnosis"""
        url = f"{self.base_url}/upload"
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {}
            if cattle_id:
                data['cattle_id'] = cattle_id
            if location:
                data['location'] = location
            if notes:
                data['notes'] = notes
            
            response = self.session.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()
    
    def get_report(self, report_id: int) -> Dict:
        """Get report details"""
        url = f"{self.base_url}/report/{report_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        url = f"{self.base_url}/api/stats"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

# Usage
api = CattleHealthAPI()
result = api.diagnose('cattle.jpg', cattle_id='COW-001')
print(f"Diagnosis: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")
```

### JavaScript SDK Example

```javascript
class CattleHealthAPI {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
    }
    
    async diagnose(file, options = {}) {
        const formData = new FormData();
        formData.append('file', file);
        
        if (options.cattleId) formData.append('cattle_id', options.cattleId);
        if (options.location) formData.append('location', options.location);
        if (options.notes) formData.append('notes', options.notes);
        
        const response = await fetch(`${this.baseUrl}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Upload failed');
        return await response.json();
    }
    
    async getReport(reportId) {
        const response = await fetch(`${this.baseUrl}/report/${reportId}`);
        if (!response.ok) throw new Error('Report not found');
        return await response.json();
    }
    
    async getStats() {
        const response = await fetch(`${this.baseUrl}/api/stats`);
        if (!response.ok) throw new Error('Failed to fetch stats');
        return await response.json();
    }
}

// Usage
const api = new CattleHealthAPI();
const result = await api.diagnose(fileInput.files[0], {
    cattleId: 'COW-001',
    location: 'Farm A'
});
console.log(`Diagnosis: ${result.prediction}`);
console.log(`Confidence: ${result.confidence}%`);
```

---

## Postman Collection

Import this collection into Postman for easy API testing:

```json
{
    "info": {
        "name": "Cattle Disease Detection API",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    },
    "item": [
        {
            "name": "Upload Image",
            "request": {
                "method": "POST",
                "header": [],
                "body": {
                    "mode": "formdata",
                    "formdata": [
                        {
                            "key": "file",
                            "type": "file",
                            "src": ""
                        },
                        {
                            "key": "cattle_id",
                            "value": "COW-001",
                            "type": "text"
                        }
                    ]
                },
                "url": {
                    "raw": "http://localhost:5000/upload",
                    "protocol": "http",
                    "host": ["localhost"],
                    "port": "5000",
                    "path": ["upload"]
                }
            }
        },
        {
            "name": "Get Statistics",
            "request": {
                "method": "GET",
                "url": {
                    "raw": "http://localhost:5000/api/stats",
                    "protocol": "http",
                    "host": ["localhost"],
                    "port": "5000",
                    "path": ["api", "stats"]
                }
            }
        }
    ]
}
```

---

## Support

For API support:
- GitHub Issues: https://github.com/Shashankxou/cattle-disease-detector/issues
- Email: support@cattlehealth.ai

---

Made with ❤️ for developers
