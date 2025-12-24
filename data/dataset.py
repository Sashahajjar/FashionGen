"""
Dataset for Fashion-Gen Data

This module implements a PyTorch Dataset that loads fashion images and captions.
Currently returns random mock data for testing. Real Fashion-Gen data loading
will be integrated later.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class FashionGenDataset(Dataset):
    """
    Dataset class for Fashion-Gen data.
    
    Currently returns random mock data for testing purposes.
    TODO: Replace with real Fashion-Gen image and caption loading.
    """
    
    def __init__(
        self,
        image_size=(224, 224),
        max_seq_len=50,
        vocab_size=10000,
        num_samples=1000,
        num_classes=10
    ):
        """
        Initialize the dataset.
        
        Args:
            image_size: Tuple of (height, width) for images
            max_seq_len: Maximum sequence length for captions
            vocab_size: Size of vocabulary for token IDs
            num_samples: Number of samples in the dataset (for mock data)
            num_classes: Number of clothing categories
        """
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        # TODO: Load real Fashion-Gen data here
        # - Load image paths from data/images/
        # - Load captions from data/captions/
        # - Create vocabulary and tokenizer
        # - Preprocess images and text
        
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Currently returns random mock data.
        TODO: Replace with real Fashion-Gen data loading.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary containing:
                - 'image': Tensor of shape (3, H, W)
                - 'caption_tokens': Tensor of shape (seq_len,)
                - 'caption_length': Integer length of the caption
        """
        # TODO: Replace with real image loading
        # image_path = self.image_paths[idx]
        # image = load_and_preprocess_image(image_path)
        
        # Generate random image tensor: (3, H, W)
        # Values normalized to [0, 1] range (will be normalized later)
        image = torch.rand(3, self.image_size[0], self.image_size[1])
        
        # TODO: Replace with real caption loading
        # caption = self.captions[idx]
        # caption_tokens = self.tokenizer.encode(caption)
        
        # Generate random caption tokens and length
        # Random sequence length between 5 and max_seq_len
        seq_len = np.random.randint(5, self.max_seq_len + 1)
        caption_tokens = torch.randint(1, self.vocab_size, (seq_len,))
        caption_length = seq_len
        
        # Generate random class label (for mock data)
        # TODO: Replace with real Fashion-Gen category labels
        label = np.random.randint(0, self.num_classes)
        
        return {
            'image': image,
            'caption_tokens': caption_tokens,
            'caption_length': caption_length,
            'label': label,
            'idx': idx  # For debugging
        }


def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
):
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: FashionGenDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
    
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length sequences.
        
        Args:
            batch: List of samples from the dataset
        
        Returns:
            Dictionary with batched tensors
        """
        images = torch.stack([item['image'] for item in batch])
        
        # Pad sequences to the same length
        caption_tokens = [item['caption_tokens'] for item in batch]
        caption_tokens_padded = pad_sequence(
            caption_tokens, batch_first=True, padding_value=0
        )
        
        caption_lengths = torch.tensor(
            [item['caption_length'] for item in batch], dtype=torch.long
        )
        
        labels = torch.tensor(
            [item['label'] for item in batch], dtype=torch.long
        )
        
        return {
            'images': images,
            'caption_tokens': caption_tokens_padded,
            'caption_lengths': caption_lengths,
            'labels': labels
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

