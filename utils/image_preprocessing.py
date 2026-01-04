"""
Image preprocessing utilities

This module contains functions for preprocessing images.
Note: The main image transforms are in data/dataset.py (Flickr8kDataset).
"""

import torch
from torchvision import transforms


def get_image_transform(image_size=(224, 224), is_training=True):
    """
    Get image transformation pipeline.
    
    Args:
        image_size: Tuple of (height, width) for resizing
        is_training: Whether this is for training (includes augmentation)
    
    Returns:
        torchvision.transforms.Compose object
    
    Image preprocessing utilities for Flickr8k dataset.
    """
    if is_training:
        # Training transforms with augmentation
        # Image augmentation for training
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def load_and_preprocess_image(image_path, image_size=(224, 224)):
    """
    Load and preprocess an image from file path.
    
    Args:
        image_path: Path to the image file
        image_size: Tuple of (height, width) for resizing
    
    Returns:
        Preprocessed image tensor
    
    Load image from file path.
    """
    from PIL import Image
    
    # Load image from path
    # image = Image.open(image_path).convert('RGB')
    # transform = get_image_transform(image_size, is_training=False)
    # return transform(image)
    
    # Placeholder: return random tensor for mock data
    return torch.rand(3, image_size[0], image_size[1])

