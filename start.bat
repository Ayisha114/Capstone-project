@echo off
REM Cattle Disease Detection System - Quick Start Script (Windows)
REM This script sets up and runs the application

echo.
echo 🐄 Cattle Disease Detection System - Quick Start
echo ================================================
echo.

REM Check Python version
echo 📋 Checking Python version...
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8 or higher
    pause
    exit /b 1
)
echo ✓ Python found
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)
echo.

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
echo ✓ Dependencies installed
echo.

REM Check for model file
echo 🔍 Checking for model file...
if exist "models\cattle_disease_vit_model.pth" (
    echo ✓ Model file found
) else (
    echo ⚠️  WARNING: Model file not found!
    echo    Please add your trained model to models\cattle_disease_vit_model.pth
    echo    See models\README.md for instructions
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)
echo.

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "static\uploads" mkdir static\uploads
if not exist "models" mkdir models
echo ✓ Directories created
echo.

REM Check for .env file
if not exist ".env" (
    echo 📝 Creating .env file from template...
    copy .env.example .env >nul
    echo ✓ .env file created (please update with your values)
) else (
    echo ✓ .env file already exists
)
echo.

REM Run the application
echo 🚀 Starting application...
echo ================================================
echo.
echo Application will be available at:
echo   🌐 http://localhost:5000
echo.
echo Admin credentials:
echo   👤 Username: admin
echo   🔑 Password: admin123
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

python app.py

pause
