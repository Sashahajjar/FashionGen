"""
Inference script for Flickr8k model

This script performs inference using the trained model.
"""

import torch
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import create_fusion_model
from data.dataset import Flickr8kDataset, SimpleVocabulary
from training.config import MODEL_CONFIG, TRAIN_CONFIG, PATHS
from training.train_utils import load_checkpoint
from PIL import Image
import torchvision.transforms as transforms


def predict_from_image_and_text(model, image, caption_tokens, caption_length, device):
    """
    Make a prediction from an image and text caption.
    
    Args:
        model: The fusion model
        image: Image tensor of shape (1, 3, H, W)
        caption_tokens: Caption token IDs tensor of shape (1, seq_len)
        caption_length: Caption length tensor of shape (1,)
        device: Device to run on
    
    Returns:
        Model output logits for classification
    """
    model.eval()
    
    with torch.no_grad():
        # Move to device
        image = image.to(device)
        caption_tokens = caption_tokens.to(device)
        caption_length = caption_length.to(device)
        
        # Forward pass
        output = model(image, caption_tokens, caption_length)
        
        return output


def main():
    """
    Main inference function.
    """
    print("=" * 60)
    print("Flickr8k Model Inference")
    print("=" * 60)
    
    # Set device
    device = torch.device(TRAIN_CONFIG['device'])
    print(f"Using device: {device}")
    
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
    
    # Load checkpoint if available
    checkpoint_path = os.path.join(PATHS['saved_models_dir'], 'multimodal.pth')
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        num_classes = checkpoint.get('num_classes', MODEL_CONFIG['num_classes'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print("\nNo checkpoint found. Using randomly initialized model.")
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
    
    # Create transform for inference
    transform = transforms.Compose([
        transforms.Resize(TRAIN_CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Example: Load a sample from test set for inference
    print("\nLoading sample from test dataset...")
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
    
    if len(test_dataset) > 0:
        sample = test_dataset[0]
        image = sample['image'].unsqueeze(0)  # Add batch dimension
        caption_tokens = sample['caption_tokens'].unsqueeze(0)
        caption_length = torch.tensor([sample['caption_length']])
        
        print(f"Image ID: {sample['image_id']}")
        print(f"Caption: {sample['caption_text']}")
        
        # Make prediction
        print("\nRunning inference...")
        output = predict_from_image_and_text(
            model, image, caption_tokens, caption_length, device
        )
        
        # Get predicted class
        _, predicted_class = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        print(f"\nOutput shape: {output.shape}")
        print(f"Predicted class: {predicted_class.item()}")
        print(f"Confidence: {probabilities[0][predicted_class.item()].item():.4f}")
        print(f"\nTop 3 predictions:")
        top_probs, top_indices = torch.topk(probabilities[0], 3)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            print(f"  {i+1}. Class {idx.item()}: {prob.item():.4f}")
    else:
        print("No test samples available.")


if __name__ == '__main__':
    main()

