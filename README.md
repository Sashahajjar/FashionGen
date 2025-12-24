# Fashion-Gen Project

A PyTorch-based project for fashion image and text fusion using CNN and RNN models.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd fashiongen-project
```

### 2. Create Virtual Environment
```bash
python3.12 -m venv venv
```

If you don't have Python 3.12, you can use Python 3.11 or 3.10:
```bash
python3.11 -m venv venv
# or
python3.10 -m venv venv
```

### 3. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run Training (with mock data)
```bash
python training/train.py
```

### 6. Evaluate Model
```bash
python training/evaluate.py
```

### 7. Run Inference Demo
```bash
python inference/demo.py
```

## Project Structure

```
fashiongen-project/
├── data/              # Data loading and preprocessing
├── models/            # CNN, RNN, and Fusion models
├── training/          # Training scripts and configuration
├── inference/         # Inference scripts
├── utils/             # Utility functions
├── notebooks/          # Jupyter notebooks
└── saved_models/      # Trained model checkpoints
```

## Training Outputs

### During Training
The training script prints clear metrics for each epoch:
```
Epoch 3/10 | Train Loss: 0.8234 Train Acc: 45.23% | Val Loss: 0.9123 Val Acc: 42.15%
```

### After Training
- Final validation accuracy
- Best model saved to `saved_models/multimodal.pth`
- Model checkpoint includes:
  - Model state dict
  - Optimizer state
  - Training metadata (vocab_size, num_classes, config)

### Evaluation Outputs
- Test accuracy
- Per-class accuracy
- Confusion matrix
- Classification report

### Inference Outputs
- Predicted class and confidence
- Top-3 predictions with probabilities
- All class probabilities
- True vs Predicted comparison (if true label provided)

## Notes

- The project currently works with **mock data** for testing
- Real Fashion-Gen data loading will be integrated later (see TODO comments in code)
- The virtual environment (`venv/`) is **not** included in git - each person must create their own
- Best model checkpoint is saved to `saved_models/multimodal.pth`
- All outputs are formatted for easy copying into presentation slides

## Requirements

- Python 3.10, 3.11, or 3.12
- PyTorch 2.0+
- See `requirements.txt` for full list of dependencies

