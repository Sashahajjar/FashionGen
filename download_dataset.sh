#!/bin/bash
# Automated script to download and set up Flickr8k dataset

set -e  # Exit on error

echo "=========================================="
echo "Flickr8k Dataset Download & Setup"
echo "=========================================="
echo ""

PROJECT_DIR="/Users/sashahajjar/Desktop/ML Project/fashiongen-project"
cd "$PROJECT_DIR"

# Create necessary directories
echo "Creating directories..."
mkdir -p data/images
mkdir -p data/captions
mkdir -p data/downloads

# Check if Kaggle is installed
if command -v kaggle &> /dev/null; then
    echo "✓ Kaggle CLI found"
    
    # Check if credentials exist
    if [ -f ~/.kaggle/kaggle.json ]; then
        echo "✓ Kaggle credentials found"
        
        echo ""
        echo "Downloading Flickr8k dataset from Kaggle..."
        echo "This may take a while (dataset is ~1-2 GB)..."
        
        kaggle datasets download -d adityajn105/flickr8k -p data/downloads || \
        kaggle datasets download adityajn105/flickr8k -p data/downloads
        
        if [ $? -eq 0 ]; then
            echo "✓ Download complete"
        else
            echo "✗ Kaggle download failed. Try manual download from:"
            echo "  https://www.kaggle.com/datasets/adityajn105/flickr8k"
            exit 1
        fi
    else
        echo "✗ Kaggle credentials not found at ~/.kaggle/kaggle.json"
        echo ""
        echo "To set up Kaggle:"
        echo "1. Go to https://www.kaggle.com/account"
        echo "2. Click 'Create New Token' to download kaggle.json"
        echo "3. Run: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/"
        echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
        echo ""
        echo "Or download manually from: https://www.kaggle.com/datasets/adityajn105/flickr8k"
        exit 1
    fi
else
    echo "✗ Kaggle CLI not found"
    echo ""
    echo "Install it with: pip install kaggle"
    echo "Or download manually from: https://www.kaggle.com/datasets/adityajn105/flickr8k"
    exit 1
fi

# Extract the dataset
echo ""
echo "Extracting dataset..."
cd data/downloads

# Find the zip file
ZIP_FILE=$(ls -1 *.zip 2>/dev/null | head -1)

if [ -z "$ZIP_FILE" ]; then
    echo "✗ No zip file found in data/downloads/"
    exit 1
fi

echo "Found: $ZIP_FILE"
unzip -q "$ZIP_FILE" || unzip "$ZIP_FILE"

# Organize files
echo ""
echo "Organizing files..."

# Find and copy images
if [ -d "Flickr8k_Dataset" ]; then
    echo "Copying images from Flickr8k_Dataset/..."
    cp -r Flickr8k_Dataset/* ../images/
elif [ -d "Flicker8k_Dataset" ]; then
    echo "Copying images from Flicker8k_Dataset/..."
    cp -r Flicker8k_Dataset/* ../images/
else
    # Try to find images folder
    IMG_DIR=$(find . -type d -name "*8k*Dataset*" -o -name "*Images*" | head -1)
    if [ -n "$IMG_DIR" ]; then
        echo "Copying images from $IMG_DIR/..."
        cp -r "$IMG_DIR"/* ../images/
    else
        echo "⚠️  Could not find images directory. Please copy images manually to data/images/"
    fi
fi

# Find and copy captions file
if [ -f "Flickr8k.token.txt" ]; then
    cp Flickr8k.token.txt ../captions/
elif [ -d "Flickr8k_text" ] && [ -f "Flickr8k_text/Flickr8k.token.txt" ]; then
    cp Flickr8k_text/Flickr8k.token.txt ../captions/
else
    TOKEN_FILE=$(find . -name "*.token.txt" -o -name "*token*" | head -1)
    if [ -n "$TOKEN_FILE" ]; then
        cp "$TOKEN_FILE" ../captions/Flickr8k.token.txt
    else
        echo "⚠️  Could not find token file. Please copy it manually to data/captions/Flickr8k.token.txt"
    fi
fi

cd "$PROJECT_DIR"

# Verify setup
echo ""
echo "Verifying setup..."
IMAGE_COUNT=$(ls -1 data/images/*.jpg data/images/*.jpeg 2>/dev/null | wc -l | tr -d ' ')
TOKEN_EXISTS=$(test -f data/captions/Flickr8k.token.txt && echo "yes" || echo "no")

echo "Images found: $IMAGE_COUNT"
echo "Token file exists: $TOKEN_EXISTS"

if [ "$IMAGE_COUNT" -gt 0 ] && [ "$TOKEN_EXISTS" = "yes" ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "You can now train with real data:"
    echo "  python training/train.py"
    echo ""
else
    echo ""
    echo "⚠️  Setup incomplete. Please check:"
    echo "  - Images in data/images/ (expected ~8000)"
    echo "  - Captions file at data/captions/Flickr8k.token.txt"
fi

