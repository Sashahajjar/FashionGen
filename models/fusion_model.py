"""
Fusion Model for Combining Image and Text Features

This module implements a fusion model that combines features from CNN (images)
and RNN (text) to create a unified representation for multimodal classification.
"""

import torch
import torch.nn as nn
from .cnn_model import ImageCNN
from .rnn_model import TextRNN


class MultimodalFusionModel(nn.Module):
    """
    Fusion model that combines image and text features.
    
    Input:
        - Images: Tensor of shape (B, 3, 224, 224)
        - Token IDs: Tensor of shape (B, seq_len)
        - Lengths: Tensor of shape (B,)
    
    Output: Combined feature vector of shape (B, output_dim)
    
    """
    
    def __init__(
        self,
        cnn_feature_dim=512,
        rnn_feature_dim=512,
        fusion_dim=512,
        output_dim=256,
        fusion_method='concat',
        num_classes=10
    ):
        """
        Initialize the fusion model.
        
        Args:
            cnn_feature_dim: Output dimension of CNN features
            rnn_feature_dim: Output dimension of RNN features
            fusion_dim: Dimension after fusion
            output_dim: Final output dimension
            fusion_method: Method to fuse features ('concat', 'add', 'multiply')
            num_classes: Number of classification classes
        """
        super(MultimodalFusionModel, self).__init__()
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        
        # Initialize CNN and RNN models
        self.cnn = ImageCNN(feature_dim=cnn_feature_dim, pretrained=True)
        self.rnn = TextRNN(feature_dim=rnn_feature_dim)
        
        # Store freeze_cnn flag for later use
        self._freeze_cnn = False
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_input_dim = cnn_feature_dim + rnn_feature_dim
        elif fusion_method == 'add' or fusion_method == 'multiply':
            # For add/multiply, dimensions must match
            assert cnn_feature_dim == rnn_feature_dim, \
                "For add/multiply fusion, cnn_feature_dim must equal rnn_feature_dim"
            fusion_input_dim = cnn_feature_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, output_dim),
            nn.ReLU()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(output_dim, num_classes)
        )
        
    def forward(self, images, token_ids, lengths):
        """
        Forward pass through the fusion model.
        
        Args:
            images: Tensor of shape (B, 3, 224, 224)
            token_ids: Tensor of shape (B, seq_len)
            lengths: Tensor of shape (B,)
        
        Returns:
            fused_features: Tensor of shape (B, output_dim)
        """
        # Extract image features: (B, 3, 224, 224) -> (B, cnn_feature_dim)
        image_features = self.cnn(images)
        
        # Extract text features: (B, seq_len) -> (B, rnn_feature_dim)
        text_features = self.rnn(token_ids, lengths)
        
        # Fuse features
        if self.fusion_method == 'concat':
            fused = torch.cat([image_features, text_features], dim=1)
        elif self.fusion_method == 'add':
            fused = image_features + text_features
        elif self.fusion_method == 'multiply':
            fused = image_features * text_features
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Pass through fusion layer: (B, fusion_input_dim) -> (B, output_dim)
        features = self.fusion_layer(fused)
        
        # Classification: (B, output_dim) -> (B, num_classes)
        logits = self.classifier(features)
        
        return logits
    
    def freeze_cnn(self, freeze=True, freeze_layers='all'):
        """
        Freeze or unfreeze CNN layers.
        
        Args:
            freeze: If True, freeze CNN layers; if False, unfreeze
            freeze_layers: 'all' to freeze all CNN layers, 'partial' to freeze only backbone
        """
        self._freeze_cnn = freeze
        
        if freeze_layers == 'all':
            # Freeze all CNN parameters (backbone + projection)
            for param in self.cnn.parameters():
                param.requires_grad = not freeze
        elif freeze_layers == 'partial':
            # Freeze only backbone, allow projection layer to train
            for param in self.cnn.backbone.parameters():
                param.requires_grad = not freeze
            # Projection layer always trainable
            for param in self.cnn.projection.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze_layers option: {freeze_layers}")


def create_fusion_model(
    cnn_feature_dim=512,
    rnn_feature_dim=512,
    fusion_dim=512,
    output_dim=256,
    fusion_method='concat',
    num_classes=10
):
    """
    Factory function to create a fusion model instance.
    
    Args:
        cnn_feature_dim: Output dimension of CNN features
        rnn_feature_dim: Output dimension of RNN features
        fusion_dim: Dimension after fusion
        output_dim: Final output dimension
        fusion_method: Method to fuse features ('concat', 'add', 'multiply')
        num_classes: Number of classification classes
    
    Returns:
        MultimodalFusionModel instance
    """
    return MultimodalFusionModel(
        cnn_feature_dim=cnn_feature_dim,
        rnn_feature_dim=rnn_feature_dim,
        fusion_dim=fusion_dim,
        output_dim=output_dim,
        fusion_method=fusion_method,
        num_classes=num_classes
    )
