#!/bin/bash

# Cattle Disease Detection System - Quick Start Script
# This script sets up and runs the application

set -e  # Exit on error

echo "🐄 Cattle Disease Detection System - Quick Start"
echo "================================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version found"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Check for model file
echo "🔍 Checking for model file..."
if [ -f "models/cattle_disease_vit_model.pth" ]; then
    echo "✓ Model file found"
else
    echo "⚠️  WARNING: Model file not found!"
    echo "   Please add your trained model to models/cattle_disease_vit_model.pth"
    echo "   See models/README.md for instructions"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p static/uploads
mkdir -p models
echo "✓ Directories created"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created (please update with your values)"
else
    echo "✓ .env file already exists"
fi
echo ""

# Run the application
echo "🚀 Starting application..."
echo "================================================"
echo ""
echo "Application will be available at:"
echo "  🌐 http://localhost:5000"
echo ""
echo "Admin credentials:"
echo "  👤 Username: admin"
echo "  🔑 Password: admin123"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"
echo ""

python app.py
