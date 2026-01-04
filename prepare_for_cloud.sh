#!/bin/bash
# Script to prepare project for cloud usage (free up local storage)

echo "=========================================="
echo "Preparing Project for Cloud Usage"
echo "=========================================="
echo ""
echo "This will remove large files to free up space:"
echo "  - venv/ (~700MB)"
echo "  - saved_models/ (~735MB)"
echo "  - __pycache__/ (varies)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Cleaning up..."

# Remove virtual environment
if [ -d "venv" ]; then
    echo "Removing venv/..."
    rm -rf venv/
    echo "✓ venv/ removed"
fi

# Remove saved models (optional - ask first)
if [ -d "saved_models" ]; then
    echo ""
    read -p "Remove saved_models/ (~735MB)? You can retrain later. (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf saved_models/
        echo "✓ saved_models/ removed"
    else
        echo "Keeping saved_models/"
    fi
fi

# Remove Python cache
echo "Removing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
echo "✓ Python cache removed"

# Calculate new size
echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Project is now ready for cloud usage."
echo ""
echo "Next steps:"
echo "  1. Push to GitHub: git init && git add . && git commit -m 'Initial commit'"
echo "  2. Use Google Colab: Upload notebooks/FashionGen_Colab_Training.ipynb"
echo "  3. See CLOUD_SETUP.md for detailed instructions"
echo ""

