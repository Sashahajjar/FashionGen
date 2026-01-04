"""
Helper utility functions

This module contains various helper functions used throughout the project.
"""

import torch
import numpy as np
import random


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the available device (CUDA if available, else CPU).
    
    Returns:
        torch.device object
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

