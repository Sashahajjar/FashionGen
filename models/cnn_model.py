"""
CNN Model for Fashion Image Feature Extraction

This module implements a CNN that extracts visual features from fashion images.
Currently works with mock data (random tensors). Real Fashion-Gen image loading
will be integrated later.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class FashionCNN(nn.Module):
    """
    CNN model for extracting features from fashion images.
    
    Input: Image tensors of shape (B, 3, 224, 224)
    Output: Feature vectors of shape (B, feature_dim)
    
    TODO: Replace mock data with real Fashion-Gen image loading
    """
    
    def __init__(self, feature_dim=512, pretrained=True):
        """
        Initialize the CNN model.
        
        Args:
            feature_dim: Dimension of the output feature vector
            pretrained: Whether to use pretrained weights
        """
        super(FashionCNN, self).__init__()
        self.feature_dim = feature_dim
        
        # Use ResNet50 as backbone (can be changed to other architectures)
        # TODO: Replace with Fashion-Gen specific architecture if needed
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add a projection layer to get desired feature dimension
        self.projection = nn.Linear(2048, feature_dim)
        self.relu = nn.ReLU()
        
    def forward(self, images):
        """
        Forward pass through the CNN.
        
        Args:
            images: Tensor of shape (B, 3, 224, 224)
                   Currently accepts random tensors for mock data
                   TODO: Replace with real Fashion-Gen images
        
        Returns:
            features: Tensor of shape (B, feature_dim)
        """
        # Extract features using ResNet backbone
        # Input: (B, 3, 224, 224) -> Output: (B, 2048, 1, 1)
        x = self.backbone(images)
        
        # Flatten: (B, 2048, 1, 1) -> (B, 2048)
        x = x.view(x.size(0), -1)
        
        # Project to desired dimension: (B, 2048) -> (B, feature_dim)
        features = self.projection(x)
        features = self.relu(features)
        
        return features


def create_cnn_model(feature_dim=512, pretrained=True):
    """
    Factory function to create a CNN model instance.
    
    Args:
        feature_dim: Dimension of the output feature vector
        pretrained: Whether to use pretrained weights
    
    Returns:
        FashionCNN model instance
    """
    return FashionCNN(feature_dim=feature_dim, pretrained=pretrained)

