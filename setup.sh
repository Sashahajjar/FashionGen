#!/bin/bash
# Setup script for Flickr8k project
# Uses Python 3.12 (PyTorch doesn't support Python 3.13 yet)

echo "=========================================="
echo "Setting up Flickr8k Project Environment"
echo "=========================================="
echo ""

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Error: Python 3.12 not found!"
    echo "Please install Python 3.12 first."
    echo "You can install it via Homebrew: brew install python@3.12"
    exit 1
fi

echo "✓ Found Python 3.12"

# Remove old venv if it exists
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create new venv with Python 3.12
echo "Creating virtual environment with Python 3.12..."
python3.12 -m venv venv

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test with mock data, run:"
echo "  python training/train.py"
echo ""

