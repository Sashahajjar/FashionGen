"""
Inference/Demo script for Flickr8k model

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
from data.dataset import Flickr8kDataset
from training.config import MODEL_CONFIG, TRAIN_CONFIG, PATHS
from training.train_utils import load_checkpoint


# Class names based on Flickr8k caption keywords
CLASS_NAMES = [
    'Dog', 'Cat', 'Person', 'Vehicle', 'Water',
    'Building', 'Tree', 'Food', 'Sport', 'Sky'
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
    Demo function that runs inference on Flickr8k test data.
    """
    print("=" * 80)
    print("Flickr8k Inference Demo")
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
    
    # Load vocabulary from training data
    print("\nLoading vocabulary...")
    images_dir = PATHS['images_dir']
    captions_file = PATHS['captions_file']
    
    train_dataset_for_vocab = Flickr8kDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        image_size=TRAIN_CONFIG['image_size'],
        max_seq_len=TRAIN_CONFIG['max_seq_len'],
        vocab_size=MODEL_CONFIG['vocab_size'],
        split='train',
        build_vocab=True,
        num_classes=MODEL_CONFIG['num_classes'],
        train_split=TRAIN_CONFIG['train_split'],
        val_split=TRAIN_CONFIG['val_split']
    )
    vocab = train_dataset_for_vocab.vocab
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = Flickr8kDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        image_size=TRAIN_CONFIG['image_size'],
        max_seq_len=TRAIN_CONFIG['max_seq_len'],
        vocab_size=MODEL_CONFIG['vocab_size'],
        split='test',
        build_vocab=False,
        vocab=vocab,
        num_classes=MODEL_CONFIG['num_classes'],
        train_split=TRAIN_CONFIG['train_split'],
        val_split=TRAIN_CONFIG['val_split']
    )
    
    # Run inference on a few examples
    print("\n" + "=" * 80)
    print("Running Inference on Test Examples")
    print("=" * 80)
    
    num_examples = min(3, len(test_dataset))
    for example_idx in range(num_examples):
        print(f"\n{'=' * 80}")
        print(f"Example {example_idx + 1}/{num_examples}")
        print(f"{'=' * 80}")
        
        # Get sample from test dataset
        sample = test_dataset[example_idx]
        image = sample['image'].unsqueeze(0)  # Add batch dimension
        caption_tokens = sample['caption_tokens'].unsqueeze(0)
        caption_length = torch.tensor([sample['caption_length']])
        true_label = sample['label']
        
        print(f"Image ID: {sample['image_id']}")
        print(f"Caption: {sample['caption_text']}")
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


if __name__ == '__main__':
    demo()

