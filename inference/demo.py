"""
Inference/Demo script for Fashion-Gen model

This script performs inference using the trained model.
Takes an image and text description and outputs predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import create_fusion_model
from training.config import MODEL_CONFIG, TRAIN_CONFIG, PATHS
from training.train_utils import load_checkpoint


# Mock class names (replace with real Fashion-Gen categories)
CLASS_NAMES = [
    'T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
]


def predict(model, image, caption_tokens, caption_length, device, class_names=None):
    """
    Make a prediction from an image and text caption.
    
    Args:
        model: The fusion model
        image: Image tensor of shape (1, 3, H, W)
        caption_tokens: Caption token IDs tensor of shape (1, seq_len)
        caption_length: Caption length tensor of shape (1,)
        device: Device to run on
        class_names: Optional list of class names
    
    Returns:
        Dictionary with predictions
    """
    model.eval()
    
    if class_names is None:
        class_names = CLASS_NAMES
    
    with torch.no_grad():
        # Move to device
        image = image.to(device)
        caption_tokens = caption_tokens.to(device)
        caption_length = caption_length.to(device)
        
        # Forward pass
        logits = model(image, caption_tokens, caption_length)
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        prob_values, predicted_classes = torch.max(probs, 1)
        
        # Get top-3 predictions
        top3_probs, top3_classes = torch.topk(probs, 3, dim=1)
        
        return {
            'predicted_class': predicted_classes[0].item(),
            'predicted_class_name': class_names[predicted_classes[0].item()],
            'confidence': prob_values[0].item(),
            'all_probabilities': probs[0].cpu().numpy(),
            'top3_classes': top3_classes[0].cpu().numpy(),
            'top3_probs': top3_probs[0].cpu().numpy(),
            'top3_names': [class_names[i] for i in top3_classes[0].cpu().numpy()]
        }


def print_prediction(prediction, true_label=None, class_names=None):
    """
    Print prediction results in a clean format.
    
    Args:
        prediction: Dictionary with prediction results
        true_label: Optional true label (int)
        class_names: Optional list of class names
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    print("\n" + "=" * 80)
    print("Prediction Results")
    print("=" * 80)
    
    print(f"\nPredicted Class: {prediction['predicted_class_name']} "
          f"(Class {prediction['predicted_class']})")
    print(f"Confidence: {prediction['confidence'] * 100:.2f}%")
    
    if true_label is not None:
        true_class_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
        print(f"\nTrue Label: {true_class_name} (Class {true_label})")
        is_correct = prediction['predicted_class'] == true_label
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    
    print("\n" + "-" * 80)
    print("Top-3 Predictions")
    print("-" * 80)
    for i, (class_idx, prob, name) in enumerate(zip(
        prediction['top3_classes'],
        prediction['top3_probs'],
        prediction['top3_names']
    ), 1):
        print(f"{i}. {name:20s} (Class {class_idx:2d}): {prob * 100:6.2f}%")
    
    print("\n" + "-" * 80)
    print("All Class Probabilities")
    print("-" * 80)
    for i, (name, prob) in enumerate(zip(class_names, prediction['all_probabilities'])):
        marker = " <--" if i == prediction['predicted_class'] else ""
        print(f"{name:20s}: {prob * 100:6.2f}%{marker}")
    
    print("=" * 80 + "\n")


def demo():
    """
    Demo function that runs inference on mock data.
    
    TODO: Replace with real image and caption loading.
    """
    print("=" * 80)
    print("Fashion-Gen Inference Demo")
    print("=" * 80)
    
    # Set device
    device = torch.device(TRAIN_CONFIG['device'])
    print(f"Device: {device}")
    
    # Create model
    print("\nLoading model...")
    model = create_fusion_model(
        cnn_feature_dim=MODEL_CONFIG['cnn_feature_dim'],
        rnn_feature_dim=MODEL_CONFIG['rnn_feature_dim'],
        fusion_dim=MODEL_CONFIG['fusion_dim'],
        output_dim=MODEL_CONFIG['output_dim'],
        fusion_method=MODEL_CONFIG['fusion_method'],
        num_classes=MODEL_CONFIG['num_classes']
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(PATHS['saved_models_dir'], 'multimodal.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        num_classes = checkpoint.get('num_classes', MODEL_CONFIG['num_classes'])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model.")
        num_classes = MODEL_CONFIG['num_classes']
    
    # Run inference on a few examples
    print("\n" + "=" * 80)
    print("Running Inference on Mock Examples")
    print("=" * 80)
    
    num_examples = 3
    for example_idx in range(num_examples):
        print(f"\n{'=' * 80}")
        print(f"Example {example_idx + 1}/{num_examples}")
        print(f"{'=' * 80}")
        
        # Create mock input
        # TODO: Replace with real image and caption loading
        image = torch.rand(1, 3, 224, 224)  # Single image
        seq_len = np.random.randint(10, 30)
        caption_tokens = torch.randint(1, MODEL_CONFIG['vocab_size'], (1, seq_len))
        caption_length = torch.tensor([seq_len])
        
        # Generate random true label for demo
        true_label = np.random.randint(0, num_classes)
        
        print(f"Image: Random tensor (224x224x3)")
        print(f"Caption: Random tokens (length: {seq_len})")
        print(f"True Label: {CLASS_NAMES[true_label] if true_label < len(CLASS_NAMES) else f'Class {true_label}'} (Class {true_label})")
        
        # Make prediction
        prediction = predict(
            model, image, caption_tokens, caption_length, device, CLASS_NAMES
        )
        
        # Print results
        print_prediction(prediction, true_label=true_label, class_names=CLASS_NAMES)
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nNote: This was a test run with mock data.")
    print("TODO: Replace with real Fashion-Gen image and caption loading.")


if __name__ == '__main__':
    demo()

