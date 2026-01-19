# Deployment Guide

This guide covers multiple deployment options for the Cattle Disease Detection System.

## Table of Contents
- [Local Development](#local-development)
- [Railway Deployment](#railway-deployment)
- [Heroku Deployment](#heroku-deployment)
- [Docker Deployment](#docker-deployment)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [Production Checklist](#production-checklist)

---

## Local Development

### Prerequisites
- Python 3.8+
- pip
- Virtual environment

### Steps

1. **Clone and setup**
```bash
git clone https://github.com/Shashankxou/cattle-disease-detector.git
cd cattle-disease-detector
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Add your model**
- Place `cattle_disease_vit_model.pth` in `models/` directory
- Update `models/class_names.json` and `models/model_config.json`

3. **Run**
```bash
python app.py
```

4. **Access**: http://localhost:5000

---

## Railway Deployment

Railway offers free tier with automatic deployments from GitHub.

### Steps

1. **Install Railway CLI**
```bash
npm install -g @railway/cli
```

2. **Login**
```bash
railway login
```

3. **Initialize project**
```bash
railway init
```

4. **Add environment variables**
```bash
railway variables set SECRET_KEY="your-secret-key-here"
```

5. **Deploy**
```bash
railway up
```

6. **Get URL**
```bash
railway domain
```

### Automatic Deployments
- Connect your GitHub repository in Railway dashboard
- Enable automatic deployments
- Every push to main branch will trigger deployment

---

## Heroku Deployment

### Prerequisites
- Heroku account
- Heroku CLI installed

### Steps

1. **Login**
```bash
heroku login
```

2. **Create app**
```bash
heroku create your-app-name
```

3. **Set environment variables**
```bash
heroku config:set SECRET_KEY="your-secret-key-here"
heroku config:set FLASK_ENV="production"
```

4. **Deploy**
```bash
git push heroku main
```

5. **Open app**
```bash
heroku open
```

### Scaling
```bash
heroku ps:scale web=1
```

### Logs
```bash
heroku logs --tail
```

---

## Docker Deployment

### Local Docker

1. **Build image**
```bash
docker build -t cattle-disease-detector .
```

2. **Run container**
```bash
docker run -p 5000:5000 \
  -v $(pwd)/static/uploads:/app/static/uploads \
  -v $(pwd)/database.db:/app/database.db \
  -e SECRET_KEY="your-secret-key" \
  cattle-disease-detector
```

### Docker Compose

1. **Start services**
```bash
docker-compose up -d
```

2. **Stop services**
```bash
docker-compose down
```

3. **View logs**
```bash
docker-compose logs -f
```

### Docker Hub

1. **Tag image**
```bash
docker tag cattle-disease-detector yourusername/cattle-disease-detector:latest
```

2. **Push to Docker Hub**
```bash
docker push yourusername/cattle-disease-detector:latest
```

---

## AWS EC2 Deployment

### Prerequisites
- AWS account
- EC2 instance (t2.medium or larger recommended)
- Security group allowing HTTP (80) and HTTPS (443)

### Steps

1. **Connect to EC2**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

2. **Install dependencies**
```bash
sudo apt update
sudo apt install python3-pip python3-venv nginx -y
```

3. **Clone repository**
```bash
git clone https://github.com/Shashankxou/cattle-disease-detector.git
cd cattle-disease-detector
```

4. **Setup virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. **Add model files**
```bash
# Upload your model file
scp -i your-key.pem cattle_disease_vit_model.pth ubuntu@your-ec2-ip:~/cattle-disease-detector/models/
```

6. **Create systemd service**
```bash
sudo nano /etc/systemd/system/cattle-app.service
```

Add:
```ini
[Unit]
Description=Cattle Disease Detection App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/cattle-disease-detector
Environment="PATH=/home/ubuntu/cattle-disease-detector/venv/bin"
ExecStart=/home/ubuntu/cattle-disease-detector/venv/bin/gunicorn --workers 4 --bind 0.0.0.0:5000 app:app

[Install]
WantedBy=multi-user.target
```

7. **Start service**
```bash
sudo systemctl start cattle-app
sudo systemctl enable cattle-app
sudo systemctl status cattle-app
```

8. **Configure Nginx**
```bash
sudo nano /etc/nginx/sites-available/cattle-app
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static {
        alias /home/ubuntu/cattle-disease-detector/static;
    }
}
```

9. **Enable site**
```bash
sudo ln -s /etc/nginx/sites-available/cattle-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

10. **Setup SSL (optional but recommended)**
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

---

## Production Checklist

### Security
- [ ] Change default admin password
- [ ] Set strong SECRET_KEY
- [ ] Enable HTTPS/SSL
- [ ] Configure firewall
- [ ] Regular security updates
- [ ] Implement rate limiting
- [ ] Add CORS if needed

### Performance
- [ ] Use production WSGI server (Gunicorn)
- [ ] Configure worker processes
- [ ] Enable gzip compression
- [ ] Setup CDN for static files
- [ ] Implement caching
- [ ] Database optimization

### Monitoring
- [ ] Setup error logging
- [ ] Configure application monitoring
- [ ] Setup uptime monitoring
- [ ] Enable performance tracking
- [ ] Configure alerts

### Backup
- [ ] Database backup strategy
- [ ] Model file backup
- [ ] Uploaded images backup
- [ ] Configuration backup

### Environment Variables

Set these in production:

```bash
SECRET_KEY="your-strong-secret-key-here"
FLASK_ENV="production"
DATABASE_URL="your-database-url"  # If using external DB
MAX_CONTENT_LENGTH="16777216"  # 16MB
```

### Scaling Considerations

**Horizontal Scaling:**
- Use load balancer
- Shared database
- Shared file storage (S3, etc.)
- Session management (Redis)

**Vertical Scaling:**
- Increase worker processes
- Upgrade server resources
- Optimize model inference

---

## Troubleshooting

### Common Issues

**1. Model not loading**
```bash
# Check model file exists
ls -lh models/cattle_disease_vit_model.pth

# Check file permissions
chmod 644 models/cattle_disease_vit_model.pth
```

**2. Database errors**
```bash
# Reset database
rm database.db
python app.py  # Will recreate
```

**3. Upload directory issues**
```bash
# Create directory
mkdir -p static/uploads
chmod 755 static/uploads
```

**4. Port already in use**
```bash
# Find process
lsof -i :5000

# Kill process
kill -9 <PID>
```

### Logs

**View application logs:**
```bash
# Systemd service
sudo journalctl -u cattle-app -f

# Docker
docker logs -f container-name

# Heroku
heroku logs --tail
```

---

## Support

For deployment issues:
- Check logs first
- Review this guide
- Open GitHub issue
- Email: support@cattlehealth.ai

---

## Updates

To update deployed application:

**Railway/Heroku:**
```bash
git push origin main  # Automatic deployment
```

**Docker:**
```bash
docker-compose down
docker-compose pull
docker-compose up -d
```

**EC2:**
```bash
cd cattle-disease-detector
git pull
sudo systemctl restart cattle-app
```

---

Made with ❤️ for easy deployment
