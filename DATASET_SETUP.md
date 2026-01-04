# Flickr8k Dataset Setup Guide

This guide will help you download and set up the Flickr8k dataset for training.

## Step 1: Download the Dataset

You have several options to download Flickr8k:

### Option A: From Kaggle (Recommended)

1. **Create a Kaggle account** (if you don't have one):
   - Go to https://www.kaggle.com/
   - Sign up for a free account

2. **Download Kaggle API credentials**:
   - Go to your Kaggle account settings: https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New Token" - this downloads `kaggle.json`

3. **Install Kaggle CLI** (if not already installed):
   ```bash
   pip install kaggle
   ```

4. **Set up Kaggle credentials**:
   ```bash
   # Create .kaggle directory
   mkdir -p ~/.kaggle
   
   # Move your downloaded kaggle.json file there
   # (Replace with actual path to your downloaded file)
   mv ~/Downloads/kaggle.json ~/.kaggle/
   
   # Set permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

5. **Download the dataset**:
   ```bash
   cd "/Users/sashahajjar/Desktop/ML Project/fashiongen-project"
   
   # Create download directory
   mkdir -p data/downloads
   
   # Download Flickr8k dataset
   kaggle datasets download -d adityajn105/flickr8k -p data/downloads
   
   # Or if that doesn't work, try:
   kaggle datasets download adityajn105/flickr8k -p data/downloads
   ```

### Option B: Direct Download (Alternative)

1. **Visit the dataset page**:
   - https://www.kaggle.com/datasets/adityajn105/flickr8k
   - Or search for "Flickr8k" on Kaggle

2. **Download manually**:
   - Click "Download" button
   - You'll get a zip file (usually named `flickr8k.zip`)

3. **Extract the files**:
   ```bash
   cd "/Users/sashahajjar/Desktop/ML Project/fashiongen-project"
   
   # Create downloads directory
   mkdir -p data/downloads
   
   # Move your downloaded zip file here, then extract
   unzip ~/Downloads/flickr8k.zip -d data/downloads/
   ```

### Option C: Using wget/curl (If direct links available)

Some repositories provide direct download links. Check:
- https://github.com/jbrownlee/Datasets (may have direct links)

## Step 2: Extract and Organize Files

After downloading, you need to organize the files correctly:

```bash
cd "/Users/sashahajjar/Desktop/ML Project/fashiongen-project"

# The dataset typically contains:
# - Flickr8k_Dataset/ (folder with images)
# - Flickr8k.token.txt (caption file)

# Extract if needed (if you downloaded a zip)
# unzip data/downloads/flickr8k.zip -d data/downloads/

# Move images to the correct location
# Option 1: If images are in a subfolder
if [ -d "data/downloads/Flickr8k_Dataset" ]; then
    cp -r data/downloads/Flickr8k_Dataset/* data/images/
elif [ -d "data/downloads/Flicker8k_Dataset" ]; then
    cp -r data/downloads/Flicker8k_Dataset/* data/images/
else
    # Find the images folder
    find data/downloads -type d -name "*8k*" -o -name "*Dataset*" | head -1
    # Then copy images from that folder
fi

# Move captions file
if [ -f "data/downloads/Flickr8k.token.txt" ]; then
    cp data/downloads/Flickr8k.token.txt data/captions/
elif [ -f "data/downloads/Flickr8k_text/Flickr8k.token.txt" ]; then
    cp data/downloads/Flickr8k_text/Flickr8k.token.txt data/captions/
else
    # Find the token file
    find data/downloads -name "*.token.txt" -o -name "*token*" | head -1
    # Then copy it
fi
```

## Step 3: Verify the Setup

Check that files are in the right place:

```bash
# Check images directory
ls data/images/ | head -5
# Should show image files like: 1000268201_693b08cb0e.jpg

# Check captions file
head -3 data/captions/Flickr8k.token.txt
# Should show lines like:
# 1000268201_693b08cb0e#0	A child in a pink dress is climbing up a set of stairs in an entry way .
```

Expected structure:
```
data/
├── images/
│   ├── 1000268201_693b08cb0e.jpg
│   ├── 1001773457_577c3a5d70.jpg
│   └── ... (about 8000 images)
└── captions/
    └── Flickr8k.token.txt
```

## Step 4: Test the Dataset

Run a quick test to make sure everything works:

```bash
# Activate your virtual environment
source venv/bin/activate

# Test loading the dataset
python -c "
from data.dataset import Flickr8kDataset
import os

images_dir = 'data/images'
captions_file = 'data/captions/Flickr8k.token.txt'

if os.path.exists(captions_file) and os.path.exists(images_dir):
    print('✓ Files found!')
    dataset = Flickr8kDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        split='train',
        max_samples=10  # Just test with 10 samples
    )
    print(f'✓ Dataset loaded: {len(dataset)} samples')
    sample = dataset[0]
    print(f'✓ Sample loaded: image_id={sample[\"image_id\"]}, caption={sample[\"caption_text\"][:50]}...')
else:
    print('✗ Files not found. Check paths.')
"
```

## Step 5: Train with Real Data

Once verified, train with the real dataset:

```bash
# Full training
python training/train.py

# Or with limited samples for quick test
python training/train.py --max_samples 1000

# Or specify custom paths
python training/train.py \
    --images_dir data/images \
    --captions_file data/captions/Flickr8k.token.txt
```

## Troubleshooting

### Problem: "Captions file not found"
- **Solution**: Make sure `Flickr8k.token.txt` is in `data/captions/`
- Check the file name matches exactly (case-sensitive)

### Problem: "Image not found for ID: ..."
- **Solution**: Images might have different extensions (.jpg vs .jpeg)
- Check the actual file names in `data/images/`
- The dataset loader tries common extensions automatically

### Problem: "No images in directory"
- **Solution**: Make sure images are directly in `data/images/`, not in a subfolder
- Run: `ls data/images/ | wc -l` to count images (should be ~8000)

### Problem: Dataset is too large for my system
- **Solution**: Use `--max_samples` to limit the dataset size:
  ```bash
  python training/train.py --max_samples 1000
  ```

## Quick Setup Script

I can create an automated setup script if you prefer. Would you like me to create one?

## Next Steps After Setup

1. ✅ Dataset downloaded and organized
2. ✅ Test dataset loading
3. ✅ Train with real data
4. ✅ Evaluate the model
5. ✅ Run inference/demo

---

**Note**: The dataset is about 1-2 GB in size. Make sure you have enough disk space.

