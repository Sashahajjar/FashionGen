"""
Evaluation script for Fashion-Gen model

This script evaluates the trained model on test data.
Currently contains placeholder functions. Real Fashion-Gen evaluation
will be integrated later.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import create_fusion_model
from data.dataset import FashionGenDataset, create_dataloader
from training.config import MODEL_CONFIG, TRAIN_CONFIG, PATHS
from training.train_utils import load_checkpoint


def evaluate(model, dataloader, criterion, device, num_classes):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The fusion model
        dataloader: DataLoader with test data
        criterion: Loss function
        device: Device to run on
        num_classes: Number of classes
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            caption_tokens = batch['caption_tokens'].to(device)
            caption_lengths = batch['caption_lengths'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(images, caption_tokens, caption_lengths)
            
            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_acc,
        'predictions': all_preds,
        'labels': all_labels
    }


def print_evaluation_results(metrics, num_classes, class_names=None):
    """
    Print evaluation results in a clean format.
    
    Args:
        metrics: Dictionary with evaluation metrics
        num_classes: Number of classes
        class_names: Optional list of class names
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    
    print("\n" + "-" * 80)
    print("Per-Class Accuracy")
    print("-" * 80)
    for i, acc in enumerate(metrics['per_class_accuracy']):
        print(f"{class_names[i]:20s}: {acc:6.2f}%")
    
    print("\n" + "-" * 80)
    print("Confusion Matrix")
    print("-" * 80)
    cm = metrics['confusion_matrix']
    
    # Print header
    print(f"{'':20s}", end="")
    for i in range(num_classes):
        print(f"{i:6d}", end="")
    print()
    
    # Print rows
    for i in range(num_classes):
        print(f"{class_names[i]:20s}", end="")
        for j in range(num_classes):
            print(f"{cm[i, j]:6d}", end="")
        print()
    
    print("\n" + "=" * 80)


def main():
    """
    Main evaluation function.
    
    TODO: Replace with real Fashion-Gen evaluation pipeline.
    """
    print("=" * 80)
    print("Fashion-Gen Model Evaluation")
    print("=" * 80)
    
    # Set device
    device = torch.device(TRAIN_CONFIG['device'])
    print(f"Device: {device}")
    
    # Create test dataset and dataloader
    print("\nCreating test dataset with mock data...")
    test_dataset = FashionGenDataset(
        image_size=TRAIN_CONFIG['image_size'],
        max_seq_len=TRAIN_CONFIG['max_seq_len'],
        vocab_size=MODEL_CONFIG['vocab_size'],
        num_samples=100,  # Test samples
        num_classes=MODEL_CONFIG['num_classes']
    )
    
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers']
    )
    
    # Create model
    print("\nCreating fusion model...")
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
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get metadata from checkpoint
        num_classes = checkpoint.get('num_classes', MODEL_CONFIG['num_classes'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 0.0):.2f}%")
    else:
        print(f"\nWarning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model.")
        num_classes = MODEL_CONFIG['num_classes']
    
    # Evaluate
    print("\n" + "=" * 80)
    print("Running Evaluation")
    print("=" * 80)
    
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_dataloader, criterion, device, num_classes)
    
    # Print results
    print_evaluation_results(metrics, num_classes)
    
    print("\nNote: This was a test run with mock data.")
    print("TODO: Replace with real Fashion-Gen evaluation metrics.")


if __name__ == '__main__':
    main()
