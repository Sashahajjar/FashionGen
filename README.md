# Flickr8k Multi-Modal Classification Project

A PyTorch-based deep learning project for image classification using both images and text captions. This project implements a fusion model that combines CNN (for images) and RNN (for text) to create a unified representation for multimodal classification.

## 🎯 Project Overview

This project implements a **multi-modal deep learning model** that:
- Extracts visual features from images using a ResNet50 CNN
- Extracts textual features from captions using a bidirectional LSTM/GRU
- Fuses image and text features using concatenation
- Classifies images into 10 categories based on caption content

## 📊 Results

- **Validation Accuracy**: 82.36%
- **Test Accuracy**: 89.28%
- **Best Validation Loss**: 0.5229
- **Dataset**: Flickr8k (8,091 images, ~40,000 captions)
- **Classes**: Dog, Cat, Person, Vehicle, Water, Building, Tree, Food, Sport, Sky

## 📁 Project Structure

```
fashiongen-project/
├── data/                    # Data handling modules
│   ├── dataset.py          # Flickr8kDataset class (supports CSV and token formats)
│   ├── images/             # Flickr8k images directory
│   └── captions/           # Captions file directory
├── models/                  # Model architectures
│   ├── cnn_model.py        # CNN for image feature extraction (ResNet50)
│   ├── rnn_model.py        # RNN for text feature extraction (Bidirectional LSTM)
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
│   └── Flickr8k_Training.ipynb  # Complete Colab training notebook
├── saved_models/            # Saved model checkpoints (gitignored)
├── requirements.txt         # Python dependencies
├── setup.sh                 # Environment setup script
├── download_dataset.sh      # Dataset download script
├── DATASET_SETUP.md         # Dataset setup guide
└── README.md               # This file
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. **Open the notebook**: Upload `notebooks/Flickr8k_Training.ipynb` to [Google Colab](https://colab.research.google.com/)
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Run all cells**: The notebook will automatically:
   - Clone the repository
   - Install dependencies
   - Download Flickr8k dataset
   - Train the model
   - Evaluate and run inference

### Option 2: Local Setup

#### Prerequisites

- **Python 3.8-3.12** (Python 3.12 recommended)
- pip package manager
- (Optional) CUDA-capable GPU for faster training

#### Installation

```bash
# Clone the repository
git clone https://github.com/Sashahajjar/FashionGen.git
cd FashionGen

# Create virtual environment with Python 3.12
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Or use the setup script:

```bash
./setup.sh
```

#### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully!')"
python -c "import torchvision; print(f'Torchvision {torchvision.__version__} installed successfully!')"
```

## 📊 Dataset Setup

### Using Mock Data (Default)

The project works out-of-the-box with mock data for testing. No data download required.

```bash
python training/train.py  # Automatically uses mock data
```

### Using Real Flickr8k Data

#### Option 1: Automatic Download (Kaggle API)

1. Get Kaggle API credentials from [Kaggle Account Settings](https://www.kaggle.com/account)
2. Download `kaggle.json` and place it in `~/.kaggle/`
3. Run the download script:

```bash
./download_dataset.sh
```

#### Option 2: Manual Download

1. Download from [Kaggle: Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. Extract images to `data/images/`
3. Extract `captions.txt` to `data/captions/Flickr8k.token.txt`

#### Option 3: Google Colab

The Colab notebook automatically downloads the dataset. See `notebooks/Flickr8k_Training.ipynb`.

### Dataset Structure

```
data/
├── images/
│   ├── 1000268201_693b08cb0e.jpg
│   ├── 1001773457_577c3a5d70.jpg
│   └── ... (8,091 images)
└── captions/
    └── Flickr8k.token.txt  # CSV format: image,caption
```

## 🏃 Usage

### Training

**With real data:**
```bash
python training/train.py
```

**With limited samples (for quick testing):**
```bash
python training/train.py --max_samples 1000
```

**With custom paths:**
```bash
python training/train.py \
    --images_dir data/images \
    --captions_file data/captions/Flickr8k.token.txt
```

### Evaluation

```bash
python training/evaluate.py
```

### Inference Demo

```bash
python inference/demo.py
```

### Prediction

```bash
python inference/predict.py
```

## ⚙️ Configuration

All hyperparameters and settings are in `training/config.py`:

```python
MODEL_CONFIG = {
    'num_classes': 10,           # Number of image categories
    'fusion_method': 'concat',   # 'concat', 'add', or 'multiply'
    'vocab_size': 10000,         # Vocabulary size for text
    'cnn_feature_dim': 512,      # CNN output dimension
    'rnn_feature_dim': 512,      # RNN output dimension
}

TRAIN_CONFIG = {
    'batch_size': 16,            # Adjust based on GPU memory
    'num_epochs': 5,             # Number of training epochs
    'learning_rate': 1e-4,       # Learning rate
    'freeze_cnn': True,          # Freeze pretrained CNN weights
    'train_split': 0.7,          # Training split (70%)
    'val_split': 0.15,           # Validation split (15%)
}
```

## 📈 Model Architecture

1. **CNN Branch**: ResNet50 (pretrained) → Feature extraction (512-dim)
2. **RNN Branch**: Bidirectional LSTM → Text feature extraction (512-dim)
3. **Fusion Layer**: Concatenates CNN and RNN features
4. **Classifier**: Final classification head (10 classes)

### Classification Categories

The model classifies images into 10 categories based on caption keywords:
- Class 0: Dog
- Class 1: Cat
- Class 2: Person
- Class 3: Vehicle
- Class 4: Water
- Class 5: Building
- Class 6: Tree
- Class 7: Food
- Class 8: Sport
- Class 9: Sky

## 📝 Features

- ✅ Multi-modal fusion (image + text)
- ✅ Automatic mock data fallback for testing
- ✅ Support for CSV and token caption formats
- ✅ Automatic train/val/test splitting
- ✅ Model checkpointing (best by loss/accuracy)
- ✅ Early stopping support
- ✅ Learning rate scheduling
- ✅ Evaluation metrics (accuracy, confusion matrix, per-class accuracy)
- ✅ Inference and demo scripts
- ✅ Google Colab notebook for cloud training
- ✅ GPU and CPU support

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
- Run `./setup.sh` to automatically set up Python 3.12 environment

### Dataset Loading Issues

**Problem**: "Captions file not found"

**Solution**:
- Check that `data/captions/Flickr8k.token.txt` exists
- Verify the file format (CSV: `image,caption` or token: `image_id#num caption`)
- The dataset loader supports both formats automatically

**Problem**: "Image not found for ID: ..."

**Solution**:
- Ensure images are in `data/images/` directory
- Check image file extensions (.jpg, .jpeg, .png)
- The loader tries common extensions automatically

### CUDA/GPU Issues

**Problem**: CUDA not available

**Solution**:
- The project works on CPU (slower but functional)
- For GPU: Install CUDA-compatible PyTorch from [PyTorch website](https://pytorch.org/get-started/locally/)
- In Colab: Enable GPU in Runtime settings

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [Google Colab](https://colab.research.google.com/) - For cloud training with free GPU

## 📄 License

This project is for educational/research purposes.

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements.

---

## 🎓 Project Summary

This project successfully demonstrates:
- Multi-modal deep learning (CNN + RNN fusion)
- End-to-end training pipeline
- Real-world dataset integration (Flickr8k)
- Model evaluation and inference
- Cloud training with Google Colab

**Final Results**: 89.28% test accuracy on 10-class image classification using both image and text features.
