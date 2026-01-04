"""
Dataset for Fashion-Gen Data

This module implements a PyTorch Dataset that loads fashion images and captions.
Supports both mock data (for testing) and real HDF5 data from Kaggle.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os


class FashionGenDataset(Dataset):
    """
    Dataset class for Fashion-Gen data.
    
    Supports both mock data (for testing) and real HDF5 data from Kaggle.
    Automatically detects if HDF5 file exists, otherwise uses mock data.
    """
    
    def __init__(
        self,
        image_size=(224, 224),
        max_seq_len=50,
        vocab_size=10000,
        num_samples=1000,
        num_classes=10,
        h5_file_path=None,
        split='train',
        use_mock_data=False,
        max_samples=None
    ):
        """
        Initialize the dataset.
        
        Args:
            image_size: Tuple of (height, width) for images
            max_seq_len: Maximum sequence length for captions
            vocab_size: Size of vocabulary for token IDs
            num_samples: Number of samples in the dataset (for mock data)
            num_classes: Number of clothing categories
            h5_file_path: Path to HDF5 file (if None, tries to find automatically)
            split: 'train', 'val', or 'test'
            use_mock_data: Force use of mock data even if HDF5 file exists
        """
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.split = split
        self.use_mock_data = use_mock_data
        
        # Try to load real HDF5 data
        self.h5_dataset = None
        if not use_mock_data:
            # Try to find HDF5 file
            if h5_file_path is None:
                # Look for common HDF5 file names
                possible_files = [
                    f'data/fashiongen_256_256_{split}.h5',
                    f'data/fashiongen_{split}.h5',
                    f'fashiongen_256_256_{split}.h5',
                    f'fashiongen_{split}.h5',
                ]
                
                for file_path in possible_files:
                    if os.path.exists(file_path):
                        h5_file_path = file_path
                        break
            
            # Load HDF5 dataset if file exists
            if h5_file_path and os.path.exists(h5_file_path):
                try:
                    from .h5_dataset import FashionGenH5Dataset
                    print(f"Loading real Fashion-Gen data from: {h5_file_path}")
                    self.h5_dataset = FashionGenH5Dataset(
                        h5_file_path=h5_file_path,
                        image_size=image_size,
                        max_seq_len=max_seq_len,
                        vocab_size=vocab_size,
                        split=split,
                        build_vocab=(split == 'train'),  # Only build vocab on train
                        max_samples=max_samples,
                        num_classes=num_classes
                    )
                    self.num_samples = len(self.h5_dataset)
                    print(f"Successfully loaded {self.num_samples} samples from HDF5 file")
                except Exception as e:
                    print(f"Warning: Could not load HDF5 file: {e}")
                    print("Falling back to mock data")
                    self.h5_dataset = None
            else:
                print("No HDF5 file found. Using mock data.")
        
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
        
        Uses real HDF5 data if available, otherwise returns mock data.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary containing:
                - 'image': Tensor of shape (3, H, W)
                - 'caption_tokens': Tensor of shape (seq_len,)
                - 'caption_length': Integer length of the caption
                - 'label': Category label
        """
        # Use real HDF5 data if available
        if self.h5_dataset is not None:
            return self.h5_dataset[idx]
        
        # Otherwise, generate mock data
        # Generate random image tensor: (3, H, W)
        # Values normalized to [0, 1] range (will be normalized later)
        image = torch.rand(3, self.image_size[0], self.image_size[1])
        
        # Generate random caption tokens and length
        # Random sequence length between 5 and max_seq_len
        seq_len = np.random.randint(5, self.max_seq_len + 1)
        caption_tokens = torch.randint(1, self.vocab_size, (seq_len,))
        caption_length = seq_len
        
        # Generate random class label (for mock data)
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

