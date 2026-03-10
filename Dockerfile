FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including git-lfs for large model files)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    git-lfs \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libjpeg62-turbo \
    libpng16-16 \
    libfreetype6 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads models

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Run the application (Render sets PORT). Keep a single worker to avoid
# loading the large model multiple times in memory.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 app:app"]
