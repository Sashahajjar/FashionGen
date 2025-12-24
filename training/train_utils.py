"""
Training utility functions

This module contains helper functions for training, such as checkpoint saving/loading,
logging, and evaluation metrics.
"""

import torch
import os


def save_checkpoint(model, optimizer, iteration, filepath):
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        iteration: Current iteration number
        filepath: Path to save the checkpoint
    """
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, device='cpu'):
    """
    Load a training checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        device: Device to load the checkpoint on
    
    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(filepath, map_location=device)
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def count_parameters(model):
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with total and trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_summary(model, input_shape):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (for testing forward pass)
    """
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    
    # Count parameters
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            # Create dummy input
            if isinstance(input_shape, dict):
                dummy_input = {k: torch.randn(v) for k, v in input_shape.items()}
                output = model(**dummy_input)
            else:
                dummy_input = torch.randn(input_shape)
                output = model(dummy_input)
        
        print(f"\nInput shape: {input_shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"\nCould not test forward pass: {e}")
    
    print("=" * 60 + "\n")

