"""
Training script for Flickr8k model

This script trains the fusion model on Flickr8k data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import create_fusion_model
from data.dataset import Flickr8kDataset, create_dataloader
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


def train(early_stop=None, max_samples=None, images_dir=None, captions_file=None):
    """
    Main training function.
    
    Args:
        early_stop: If True, enable early stopping. If None, use config default.
        max_samples: Maximum number of samples to use from dataset (None = use all)
        images_dir: Path to directory containing Flickr8k images (default: from config)
        captions_file: Path to Flickr8k.token.txt file (default: from config)
    """
    # Override early stopping if provided via command line
    if early_stop is not None:
        TRAIN_CONFIG['use_early_stopping'] = early_stop
    
    print("=" * 80)
    print("Flickr8k Multi-Modal Classification Training")
    print("=" * 80)
    
    # Set device
    device = torch.device(TRAIN_CONFIG['device'])
    print(f"Device: {device}")
    print(f"Number of classes: {MODEL_CONFIG['num_classes']}")
    
    # Get paths from config or arguments
    images_dir = images_dir or PATHS['images_dir']
    captions_file = captions_file or PATHS['captions_file']
    
    print(f"\nImages directory: {images_dir}")
    print(f"Captions file: {captions_file}")
    
    # Build vocabulary from training data first
    print("\nBuilding vocabulary from training data...")
    train_dataset_for_vocab = Flickr8kDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        image_size=TRAIN_CONFIG['image_size'],
        max_seq_len=TRAIN_CONFIG['max_seq_len'],
        vocab_size=MODEL_CONFIG['vocab_size'],
        split='train',
        build_vocab=True,
        max_samples=max_samples,
        num_classes=MODEL_CONFIG['num_classes'],
        train_split=TRAIN_CONFIG['train_split'],
        val_split=TRAIN_CONFIG['val_split']
    )
    
    # Use the same vocabulary for all splits
    vocab = train_dataset_for_vocab.vocab
    
    # Create train and validation datasets with shared vocabulary
    print("\nCreating train dataset...")
    train_dataset = Flickr8kDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        image_size=TRAIN_CONFIG['image_size'],
        max_seq_len=TRAIN_CONFIG['max_seq_len'],
        vocab_size=MODEL_CONFIG['vocab_size'],
        split='train',
        build_vocab=False,
        vocab=vocab,
        max_samples=max_samples,
        num_classes=MODEL_CONFIG['num_classes'],
        train_split=TRAIN_CONFIG['train_split'],
        val_split=TRAIN_CONFIG['val_split']
    )
    
    print("Creating validation dataset...")
    val_dataset = Flickr8kDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        image_size=TRAIN_CONFIG['image_size'],
        max_seq_len=TRAIN_CONFIG['max_seq_len'],
        vocab_size=MODEL_CONFIG['vocab_size'],
        split='val',
        build_vocab=False,
        vocab=vocab,
        max_samples=max_samples,
        num_classes=MODEL_CONFIG['num_classes'],
        train_split=TRAIN_CONFIG['train_split'],
        val_split=TRAIN_CONFIG['val_split']
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
    
    # Freeze CNN layers if configured
    if TRAIN_CONFIG.get('freeze_cnn', False):
        freeze_layers = TRAIN_CONFIG.get('freeze_cnn_layers', 'all')
        model.freeze_cnn(freeze=True, freeze_layers=freeze_layers)
        print(f"CNN layers frozen: {freeze_layers}")
    else:
        print("CNN layers: trainable")
    
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
    
    # Learning rate scheduler (reduce LR on plateau)
    scheduler = None
    if TRAIN_CONFIG.get('use_lr_scheduler', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Monitor validation accuracy (maximize)
            factor=TRAIN_CONFIG.get('lr_scheduler_factor', 0.5),
            patience=TRAIN_CONFIG.get('lr_scheduler_patience', 5),
            min_lr=TRAIN_CONFIG.get('lr_scheduler_min', 1e-6),
            verbose=True
        )
    
    print(f"\nOptimizer: Adam (lr={TRAIN_CONFIG['learning_rate']})")
    print(f"Loss function: CrossEntropyLoss")
    print(f"Number of epochs: {TRAIN_CONFIG['num_epochs']}")
    if scheduler:
        print(f"LR Scheduler: ReduceLROnPlateau (patience={TRAIN_CONFIG.get('lr_scheduler_patience', 5)})")
    if TRAIN_CONFIG.get('use_early_stopping', False):
        print(f"Early Stopping: Enabled (patience={TRAIN_CONFIG.get('early_stopping_patience', 10)})")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_val_acc_epoch = 0
    best_val_loss_epoch = 0
    epochs_no_improve = 0  # For early stopping
    early_stopping_metric = TRAIN_CONFIG.get('early_stopping_metric', 'loss')  # 'loss' or 'accuracy'
    
    if TRAIN_CONFIG['resume_from']:
        print(f"\nResuming from checkpoint: {TRAIN_CONFIG['resume_from']}")
        checkpoint = load_checkpoint(TRAIN_CONFIG['resume_from'], device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_acc_epoch = checkpoint.get('best_val_acc_epoch', 0)
        best_val_loss_epoch = checkpoint.get('best_val_loss_epoch', 0)
    
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
        
        # Update learning rate scheduler (if enabled)
        if scheduler:
            if early_stopping_metric == 'loss':
                scheduler.step(val_loss)
            else:
                scheduler.step(val_acc)
        
        # Print epoch results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{TRAIN_CONFIG['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.2e}")
        
        # Always track best validation accuracy and loss (regardless of early stopping metric)
        improved_loss = False
        improved_acc = False
        min_delta = TRAIN_CONFIG.get('early_stopping_min_delta', 0.001)
        
        # Track best validation loss
        if val_loss < best_val_loss - min_delta:
            improved_loss = True
            best_val_loss = val_loss
            best_val_loss_epoch = epoch + 1
        
        # Track best validation accuracy
        if val_acc > best_val_acc + min_delta:
            improved_acc = True
            best_val_acc = val_acc
            best_val_acc_epoch = epoch + 1
        
        # Check for improvement based on early stopping metric (only if early stopping enabled)
        improved = False
        if TRAIN_CONFIG.get('use_early_stopping', False):
            if early_stopping_metric == 'loss':
                if improved_loss:
                    improved = True
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:  # accuracy
                if improved_acc:
                    improved = True
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        # Save best model by loss
        if TRAIN_CONFIG['save_best'] and improved_loss:
            save_dir = PATHS['saved_models_dir']
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, 'multimodal_best_loss.pth')
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_loss_epoch': best_val_loss_epoch,
                'best_val_acc': best_val_acc,
                'best_val_acc_epoch': best_val_acc_epoch,
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
            print(f"  -> Best model by loss saved! (Val Loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch})")
        
        # Save best model by accuracy
        if TRAIN_CONFIG['save_best'] and improved_acc:
            save_dir = PATHS['saved_models_dir']
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, 'multimodal_best_acc.pth')
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_loss_epoch': best_val_loss_epoch,
                'best_val_acc': best_val_acc,
                'best_val_acc_epoch': best_val_acc_epoch,
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
            print(f"  -> Best model by accuracy saved! (Val Acc: {best_val_acc:.2f}% at epoch {best_val_acc_epoch})")
        
        # Also save a general "best" model (for backward compatibility)
        if TRAIN_CONFIG['save_best'] and (improved_loss or improved_acc):
            save_dir = PATHS['saved_models_dir']
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, 'multimodal.pth')
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_loss_epoch': best_val_loss_epoch,
                'best_val_acc': best_val_acc,
                'best_val_acc_epoch': best_val_acc_epoch,
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
        
        # Early stopping (only if enabled)
        if TRAIN_CONFIG.get('use_early_stopping', False):
            patience = TRAIN_CONFIG.get('early_stopping_patience', 3)
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered! No improvement for {epochs_no_improve} epochs.")
                if early_stopping_metric == 'loss':
                    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_val_loss_epoch})")
                else:
                    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_val_acc_epoch})")
                break
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best Validation Loss: {best_val_loss:.4f} (at epoch {best_val_loss_epoch})")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (at epoch {best_val_acc_epoch})")
    print(f"\nSaved models:")
    print(f"  - Best by loss: {os.path.join(PATHS['saved_models_dir'], 'multimodal_best_loss.pth')}")
    print(f"  - Best by accuracy: {os.path.join(PATHS['saved_models_dir'], 'multimodal_best_acc.pth')}")
    print(f"  - Latest: {os.path.join(PATHS['saved_models_dir'], 'multimodal.pth')}")
    
    print("\nTraining complete! Model architecture validated on Flickr8k dataset.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Flickr8k Multi-Modal Model')
    parser.add_argument('--early_stop', action='store_true',
                        help='Enable early stopping (disabled by default for full learning curves)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use from dataset (for Colab/subset training)')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Path to directory containing Flickr8k images (default: from config)')
    parser.add_argument('--captions_file', type=str, default=None,
                        help='Path to Flickr8k.token.txt file (default: from config)')
    args = parser.parse_args()
    
    train(early_stop=args.early_stop if args.early_stop else None,
          max_samples=args.max_samples,
          images_dir=args.images_dir,
          captions_file=args.captions_file)
