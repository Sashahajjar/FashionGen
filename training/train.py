"""
Training script for Fashion-Gen model

This script trains the fusion model on Fashion-Gen data.
Currently runs with mock data for testing. Real Fashion-Gen data loading
will be integrated later.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import create_fusion_model
from data.dataset import FashionGenDataset, create_dataloader
from training.config import MODEL_CONFIG, TRAIN_CONFIG, PATHS
from training.train_utils import save_checkpoint, load_checkpoint


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The fusion model
        dataloader: DataLoader with training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Average loss and accuracy for this epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in dataloader:
        # Move data to device
        images = batch['images'].to(device)
        caption_tokens = batch['caption_tokens'].to(device)
        caption_lengths = batch['caption_lengths'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images, caption_tokens, caption_lengths)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The fusion model
        dataloader: DataLoader with validation data
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Average loss and accuracy for validation set
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            images = batch['images'].to(device)
            caption_tokens = batch['caption_tokens'].to(device)
            caption_lengths = batch['caption_lengths'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(images, caption_tokens, caption_lengths)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train():
    """
    Main training function.
    
    Currently runs with mock data for testing.
    TODO: Replace with full training loop using real Fashion-Gen data.
    """
    print("=" * 80)
    print("Fashion-Gen Multi-Modal Classification Training")
    print("=" * 80)
    
    # Set device
    device = torch.device(TRAIN_CONFIG['device'])
    print(f"Device: {device}")
    print(f"Number of classes: {MODEL_CONFIG['num_classes']}")
    
    # Create train and validation datasets
    print("\nCreating datasets with mock data...")
    train_dataset = FashionGenDataset(
        image_size=TRAIN_CONFIG['image_size'],
        max_seq_len=TRAIN_CONFIG['max_seq_len'],
        vocab_size=MODEL_CONFIG['vocab_size'],
        num_samples=TRAIN_CONFIG['num_train_samples'],
        num_classes=MODEL_CONFIG['num_classes']
    )
    
    val_dataset = FashionGenDataset(
        image_size=TRAIN_CONFIG['image_size'],
        max_seq_len=TRAIN_CONFIG['max_seq_len'],
        vocab_size=MODEL_CONFIG['vocab_size'],
        num_samples=TRAIN_CONFIG['num_val_samples'],
        num_classes=MODEL_CONFIG['num_classes']
    )
    
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAIN_CONFIG['num_workers']
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers']
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
    
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
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nOptimizer: Adam (lr={TRAIN_CONFIG['learning_rate']})")
    print(f"Loss function: CrossEntropyLoss")
    print(f"Number of epochs: {TRAIN_CONFIG['num_epochs']}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    if TRAIN_CONFIG['resume_from']:
        print(f"\nResuming from checkpoint: {TRAIN_CONFIG['resume_from']}")
        checkpoint = load_checkpoint(TRAIN_CONFIG['resume_from'], device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # Training loop
    print("\n" + "=" * 80)
    print("Training Progress")
    print("=" * 80)
    
    for epoch in range(start_epoch, TRAIN_CONFIG['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_dataloader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_dataloader, criterion, device
        )
        
        # Print epoch results
        print(f"Epoch {epoch + 1}/{TRAIN_CONFIG['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if TRAIN_CONFIG['save_best'] and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = PATHS['saved_models_dir']
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, 'multimodal.pth')
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'num_classes': MODEL_CONFIG['num_classes'],
                'vocab_size': MODEL_CONFIG['vocab_size'],
                'model_config': MODEL_CONFIG,
                'train_config': TRAIN_CONFIG
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"  -> Best model saved! (Val Acc: {best_val_acc:.2f}%)")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {os.path.join(PATHS['saved_models_dir'], 'multimodal.pth')}")
    
    print("\nNote: This was a test run with mock data.")
    print("TODO: Replace with real Fashion-Gen data loading for full training.")


if __name__ == '__main__':
    train()
