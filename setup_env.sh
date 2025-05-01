#!/bin/bash
# Setup script for claim-assistant

set -e

echo "Setting up claim-assistant environment..."

# Check if Python 3.10+ is installed
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
python_major=$(echo $python_version | cut -d'.' -f1)
python_minor=$(echo $python_version | cut -d'.' -f2)

if [ "$python_major" -lt 3 ] || [ "$python_major" -eq 3 -a "$python_minor" -lt 10 ]; then
    echo "Error: Python 3.10 or higher is required. Found Python $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed"
    exit 1
fi

echo "✅ pip3 detected"

# Check if tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo "⚠️ Tesseract OCR not found. Installing..."
    
    # Detect OS and install
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install tesseract
        else
            echo "Error: Homebrew not found. Please install Homebrew first."
            echo "Visit https://brew.sh/ for installation instructions."
            exit 1
        fi
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y tesseract-ocr
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y tesseract
        else
            echo "Error: Unsupported Linux distribution. Please install Tesseract manually."
            echo "Visit https://github.com/tesseract-ocr/tesseract"
            exit 1
        fi
    else
        echo "Error: Unsupported operating system. Please install Tesseract manually."
        echo "Visit https://github.com/tesseract-ocr/tesseract"
        exit 1
    fi
fi

echo "✅ Tesseract OCR detected"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -e .

# Create config file if not exists
if [ ! -f ~/.claimctl.ini ]; then
    echo "Creating default config file at ~/.claimctl.ini"
    if [ -f claimctl.ini.sample ]; then
        cp claimctl.ini.sample ~/.claimctl.ini
        echo "⚠️ Please edit ~/.claimctl.ini to set your OpenAI API key"
    else
        echo "⚠️ Sample config file not found, using CLI to create default config"
        # The CLI will create a default config automatically
    fi
fi

# Initialize directories
echo "Initializing directories..."
python -m claimctl.cli config init

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To get started, run:"
echo "  claimctl --help"
echo ""
echo "To start the interactive shell, run:"
echo "  ./run.sh"