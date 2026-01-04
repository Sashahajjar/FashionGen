# Cloud Setup Guide - No Local Storage Needed

## Option 1: Google Colab (Recommended - FREE GPU!)

You already have Colab notebooks ready! This is the best option for limited storage.

### Steps:

1. **Upload project to Google Drive or GitHub**
   - Upload the project folder (excluding `venv/` and `saved_models/`)
   - Or push to GitHub and clone in Colab

2. **Open Colab Notebook**
   - Go to: https://colab.research.google.com/
   - Upload `notebooks/FashionGen_Colab_Training.ipynb`
   - Or create new notebook and copy code

3. **Mount Google Drive (if using Drive)**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Clone from GitHub (if using GitHub)**
   ```python
   !git clone https://github.com/yourusername/fashiongen-project.git
   ```

5. **Run the notebook** - It will:
   - Install all dependencies automatically
   - Download FashionGen data from Kaggle
   - Train the model on GPU (free!)
   - Save models to Drive or download

### Benefits:
- ✅ Free GPU (Tesla T4, 15GB)
- ✅ No local storage needed
- ✅ Automatic dependency installation
- ✅ Can download trained models when done

---

## Option 2: Remove Local Files to Free Space

If you want to keep working locally but need space:

### Files you can safely delete:

1. **Virtual Environment** (~700MB)
   ```bash
   rm -rf venv/
   ```
   - Can recreate later: `python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

2. **Saved Models** (~735MB)
   ```bash
   rm -rf saved_models/
   ```
   - These are just checkpoints - you can retrain
   - Or upload to cloud storage (Google Drive, Dropbox)

3. **Python Cache** (small but can add up)
   ```bash
   find . -type d -name __pycache__ -exec rm -r {} +
   find . -name "*.pyc" -delete
   ```

### After cleanup, you'll have:
- ✅ Only source code (~50MB)
- ✅ Can use Colab for training
- ✅ Can recreate venv when needed

---

## Option 3: Use Cloud Storage Services

### Upload to:
- **Google Drive** - 15GB free
- **Dropbox** - 2GB free
- **GitHub** - Unlimited (public repos) or 500MB (private)
- **AWS S3** - Pay as you go

### What to upload:
- All `.py` files
- `requirements.txt`
- `README.md`
- `notebooks/` folder
- `.gitignore`

### What NOT to upload:
- `venv/` (too large, recreate on cloud)
- `saved_models/` (can retrain or upload separately)
- `__pycache__/` (auto-generated)
- `data/` (download from Kaggle in Colab)

---

## Option 4: GitHub + Colab Workflow

1. **Push code to GitHub** (exclude venv, saved_models)
   ```bash
   git init
   git add *.py *.md *.txt notebooks/ data/ models/ training/ inference/ utils/
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/fashiongen-project.git
   git push -u origin main
   ```

2. **In Colab:**
   ```python
   !git clone https://github.com/yourusername/fashiongen-project.git
   %cd fashiongen-project
   !pip install -r requirements.txt
   ```

3. **Train in Colab** - Use the provided notebook

---

## Recommended Workflow

**Best for no local storage:**

1. ✅ Keep only source code locally (~50MB)
2. ✅ Push to GitHub
3. ✅ Use Google Colab for all training
4. ✅ Download trained models when needed
5. ✅ Delete models from Colab after downloading

**Quick Start in Colab:**

```python
# Cell 1: Clone repo
!git clone https://github.com/yourusername/fashiongen-project.git
%cd fashiongen-project

# Cell 2: Install dependencies
!pip install torch torchvision numpy matplotlib Pillow scikit-learn h5py kaggle

# Cell 3: Setup Kaggle (upload kaggle.json)
from google.colab import files
files.upload()  # Upload kaggle.json

# Cell 4: Download dataset
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d bothin/fashiongen-validation

# Cell 5: Train
!python training/train.py --h5_file fashiongen_256_256_train.h5
```

---

## Summary

**If you have NO local storage:**
- Use Google Colab (free GPU!)
- Upload code to GitHub
- Train in cloud
- Download results when done

**If you have SOME storage:**
- Keep source code only (~50MB)
- Delete venv and saved_models
- Use Colab for training
- Recreate venv locally only if needed for inference

