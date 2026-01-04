"""
Visualization utilities

This module contains functions for visualizing model outputs, training progress,
and Flickr8k data samples.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np


def visualize_batch(images, captions=None, num_samples=4):
    """
    Visualize a batch of images with optional captions.
    
    Args:
        images: Tensor of shape (B, 3, H, W) or list of image tensors
        captions: Optional list of caption strings
        num_samples: Number of samples to visualize
    
    TODO: Add Flickr8k specific visualization.
    """
    # TODO: Implement visualization for Flickr8k images and captions
    # For now, this is a placeholder
    pass


def plot_training_curves(train_losses, val_losses=None, save_path=None):
    """
    Plot training curves.
    
    Args:
        train_losses: List of training losses
        val_losses: Optional list of validation losses
        save_path: Optional path to save the plot
    
    TODO: Add more metrics (accuracy, recall, etc.) when real data is available.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

