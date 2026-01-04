# FashionGen Multi-Modal Classification Project

A PyTorch-based deep learning project for fashion item classification using both images and text captions. This project implements a fusion model that combines CNN (for images) and RNN (for text) to create a unified representation for fashion items.

## 🎯 Project Overview

This project implements a **multi-modal deep learning model** that:
- Extracts visual features from fashion images using a ResNet50 CNN
- Extracts textual features from fashion captions using a bidirectional LSTM/GRU
- Fuses image and text features using concatenation, addition, or multiplication
- Classifies fashion items into categories

## 📁 Project Structure

```
fashiongen-project/
├── data/                    # Data handling modules
│   ├── dataset.py          # Main dataset class (supports mock & real data)
│   └── h5_dataset.py       # HDF5 loader for Fashion-Gen Kaggle data
├── models/                  # Model architectures
│   ├── cnn_model.py        # CNN for image feature extraction
│   ├── rnn_model.py        # RNN for text feature extraction
│   └── fusion_model.py     # Fusion model combining CNN + RNN
├── training/                # Training scripts
│   ├── config.py           # Configuration and hyperparameters
│   ├── train.py            # Main training script
│   ├── train_utils.py      # Training utilities (checkpointing, etc.)
│   └── evaluate.py         # Model evaluation script
├── inference/               # Inference scripts
│   ├── predict.py          # Prediction script
│   └── demo.py             # Demo script with visualization
├── utils/                   # Utility functions
│   ├── helpers.py          # General helper functions
│   ├── image_preprocessing.py
│   ├── text_preprocessing.py
│   └── visualization.py
├── notebooks/               # Jupyter notebooks for Colab
│   ├── FashionGen_Colab.ipynb
│   └── FashionGen_Colab_Training.ipynb
├── saved_models/            # Saved model checkpoints
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Setup Instructions

### ⚠️ Important: Python Version Compatibility

**PyTorch does NOT support Python 3.13 yet!** 

If you're using Python 3.13, you have two options:
1. **Use Python 3.12** (recommended) - Run `./fix_python313.sh` to automatically fix this
2. See `PYTHON313_FIX.md` for detailed instructions

### Prerequisites

- **Python 3.8-3.12** (Python 3.11 or 3.12 recommended)
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation

#### Quick Fix for Python 3.13 Users

If you have Python 3.13 and Python 3.12 installed:

```bash
./fix_python313.sh
```

This script will automatically recreate your virtual environment with Python 3.12 and install all dependencies.

#### Option 1: Using Existing Virtual Environment

If you have a virtual environment already set up:

```bash
cd fashiongen-project
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option 2: Create New Virtual Environment

**For Python 3.11 or 3.12 (Recommended):**

```bash
cd fashiongen-project
python3.12 -m venv venv  # or python3.11
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**For Python 3.13+ (if PyTorch supports it):**

You may need to install PyTorch from nightly builds:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# For GPU support, replace 'cpu' with 'cu121' or check PyTorch website
```

Then install other dependencies:

```bash
pip install numpy h5py Pillow scikit-learn matplotlib
```

#### Option 3: Install PyTorch Manually

Visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) and select your configuration, then:

```bash
# Example for CPU-only (replace with your configuration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install other dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully!')"
python -c "import torchvision; print(f'Torchvision {torchvision.__version__} installed successfully!')"
```

## 📊 Data Setup

### Using Mock Data (Default)

The project works out-of-the-box with mock data for testing. No data download required.

### Using Real Fashion-Gen Data

1. Download the Fashion-Gen dataset from [Kaggle](https://www.kaggle.com/datasets/bothin/fashiongen-validation/data)
2. Extract HDF5 files to the project directory
3. Update paths in `training/config.py` if needed
4. Run training with the `--h5_file` flag:

```bash
python training/train.py --h5_file path/to/fashiongen_256_256_train.h5
```

## 🏃 Usage

### Training

**With mock data (for testing):**
```bash
python training/train.py
```

**With real HDF5 data:**
```bash
python training/train.py --h5_file data/fashiongen_256_256_train.h5
```

**With early stopping:**
```bash
python training/train.py --early_stop
```

**With limited samples (for quick testing):**
```bash
python training/train.py --max_samples 1000
```

### Evaluation

```bash
python training/evaluate.py
```

### Inference/Demo

```bash
python inference/demo.py
```

### Prediction

```bash
python inference/predict.py
```

## ⚙️ Configuration

All hyperparameters and settings are in `training/config.py`:

- **Model Configuration**: Feature dimensions, fusion method, vocabulary size
- **Training Configuration**: Batch size, learning rate, epochs, device settings
- **Paths**: Data directories and model save locations

Key settings you might want to adjust:

```python
MODEL_CONFIG = {
    'num_classes': 10,           # Number of clothing categories
    'fusion_method': 'concat',    # 'concat', 'add', or 'multiply'
    'vocab_size': 10000,         # Vocabulary size for text
}

TRAIN_CONFIG = {
    'batch_size': 16,             # Adjust based on GPU memory
    'num_epochs': 5,              # Number of training epochs
    'learning_rate': 1e-4,       # Learning rate
    'freeze_cnn': True,           # Freeze pretrained CNN weights
}
```

## 📈 Model Architecture

1. **CNN Branch**: ResNet50 (pretrained) → Feature extraction (512-dim)
2. **RNN Branch**: Bidirectional LSTM → Text feature extraction (512-dim)
3. **Fusion Layer**: Combines CNN and RNN features
4. **Classifier**: Final classification head (10 classes)

## 📝 Features

- ✅ Multi-modal fusion (image + text)
- ✅ Support for mock and real data
- ✅ HDF5 dataset loading
- ✅ Model checkpointing (best by loss/accuracy)
- ✅ Early stopping support
- ✅ Learning rate scheduling
- ✅ Evaluation metrics (accuracy, confusion matrix)
- ✅ Inference and demo scripts
- ✅ Jupyter notebooks for Colab

## 🔧 Troubleshooting

### PyTorch Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: 
- Ensure you're using Python 3.8-3.12
- Install PyTorch from the [official website](https://pytorch.org/get-started/locally/)
- Verify installation: `python -c "import torch; print(torch.__version__)"`

### Python 3.13 Compatibility

**Problem**: PyTorch not available for Python 3.13

**Solution**:
- Use Python 3.11 or 3.12 instead
- Or try PyTorch nightly builds (may be unstable)

### CUDA/GPU Issues

**Problem**: CUDA not available

**Solution**:
- The project works on CPU (slower but functional)
- For GPU: Install CUDA-compatible PyTorch from [PyTorch website](https://pytorch.org/get-started/locally/)

### Data Loading Issues

**Problem**: HDF5 file not found

**Solution**:
- Check file path in `--h5_file` argument
- Ensure HDF5 file is in the correct format
- Project will fall back to mock data if HDF5 not found

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Fashion-Gen Dataset on Kaggle](https://www.kaggle.com/datasets/bothin/fashiongen-validation/data)
- Project notebooks in `notebooks/` for Google Colab usage

## 📄 License

This project is for educational/research purposes.

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements.

---

**Note**: This project is designed to validate the CNN + RNN + Fusion architecture. For production use, consider:
- Using larger datasets
- Hyperparameter tuning
- Model architecture improvements
- Additional data augmentation

