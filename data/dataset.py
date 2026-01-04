"""
Dataset for Flickr8k Data

This module implements a PyTorch Dataset that loads Flickr8k images and captions.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
import re
from typing import Optional, Dict, List, Tuple


class SimpleVocabulary:
    """Simple vocabulary class for text tokenization."""
    
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        
        # Initialize with special tokens
        self.word2idx[self.PAD_TOKEN] = 0
        self.word2idx[self.UNK_TOKEN] = 1
        self.word2idx[self.SOS_TOKEN] = 2
        self.word2idx[self.EOS_TOKEN] = 3
        
        self.idx2word[0] = self.PAD_TOKEN
        self.idx2word[1] = self.UNK_TOKEN
        self.idx2word[2] = self.SOS_TOKEN
        self.idx2word[3] = self.EOS_TOKEN
    
    def build_vocab(self, captions):
        """Build vocabulary from list of captions."""
        # Tokenize all captions
        for caption in captions:
            if caption:
                words = self._tokenize(caption)
                self.word_counts.update(words)
        
        # Add most common words to vocabulary
        for word, count in self.word_counts.most_common(self.max_size - 4):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def _tokenize(self, text):
        """Simple tokenization (split on whitespace and punctuation)."""
        # Convert to lowercase and split
        text = text.lower()
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def encode(self, text, max_length=50):
        """Encode text to token IDs."""
        words = self._tokenize(text)
        
        # Convert to token IDs
        token_ids = [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]
        
        # Truncate or pad
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            actual_length = max_length
        else:
            actual_length = len(token_ids)
            # Pad with PAD token
            token_ids.extend([self.word2idx[self.PAD_TOKEN]] * (max_length - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long), actual_length
    
    def decode(self, token_ids):
        """Decode token IDs back to text."""
        words = []
        for idx in token_ids:
            if idx.item() in self.idx2word:
                word = self.idx2word[idx.item()]
                if word not in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                    words.append(word)
        return ' '.join(words)
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.word2idx)


class Flickr8kDataset(Dataset):
    """
    Dataset class for Flickr8k data.
    
    Loads images from a folder and captions from a token file.
    Creates classification labels based on caption content.
    """
    
    def __init__(
        self,
        images_dir: str,
        captions_file: str,
        image_size: Tuple[int, int] = (224, 224),
        max_seq_len: int = 50,
        vocab_size: int = 10000,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        build_vocab: bool = True,
        vocab: Optional[SimpleVocabulary] = None,
        max_samples: Optional[int] = None,
        num_classes: int = 10,
        train_split: float = 0.7,
        val_split: float = 0.15,
        use_mock_data: bool = False,
        num_samples: Optional[int] = None
    ):
        """
        Initialize the Flickr8k dataset.
        
        Args:
            images_dir: Directory containing images
            captions_file: Path to Flickr8k.token.txt file
            image_size: Tuple of (height, width) for images
            max_seq_len: Maximum sequence length for captions
            vocab_size: Maximum vocabulary size
            split: 'train', 'val', or 'test'
            transform: Optional image transform (if None, uses default)
            build_vocab: Whether to build vocabulary from captions
            vocab: Pre-built vocabulary (if None, builds from data)
            max_samples: Maximum number of samples to use
            num_classes: Number of classification classes
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            use_mock_data: If True, use mock data even if real data exists
            num_samples: Number of mock samples to generate (for mock data only)
        """
        self.images_dir = images_dir
        self.captions_file = captions_file
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.split = split
        self.num_classes = num_classes
        self.use_mock_data = use_mock_data
        self.num_samples = num_samples or (100 if split == 'train' else (20 if split == 'val' else 20))
        
        # Set up image transforms
        if transform is None:
            self.transform = self._get_default_transform(image_size, split == 'train')
        else:
            self.transform = transform
        
        # Check if we should use mock data
        self.is_mock = use_mock_data or not os.path.exists(captions_file) or not os.path.exists(images_dir)
        
        if self.is_mock:
            print(f"Using mock data for {split} split ({self.num_samples} samples)")
            self.image_caption_pairs = None
            self.indices = list(range(self.num_samples))
            
            # Build vocabulary for mock data
            if vocab is not None:
                self.vocab = vocab
            elif build_vocab:
                print("Building vocabulary from mock captions...")
                self.vocab = self._build_mock_vocabulary()
                print(f"Vocabulary size: {len(self.vocab.word2idx)}")
            else:
                self.vocab = SimpleVocabulary(vocab_size)
        else:
            # Load real image-caption pairs
            self.image_caption_pairs = self._load_captions()
            
            # Split dataset
            self.indices = self._split_dataset(train_split, val_split)
            
            # Limit samples if specified
            if max_samples and max_samples < len(self.indices):
                self.indices = self.indices[:max_samples]
            
            print(f"Loaded {len(self.indices)} samples for {split} split")
            
            # Build vocabulary from captions
            if vocab is not None:
                self.vocab = vocab
            elif build_vocab:
                print("Building vocabulary from captions...")
                self.vocab = self._build_vocabulary()
                print(f"Vocabulary size: {len(self.vocab.word2idx)}")
            else:
                # Use a simple vocabulary placeholder
                self.vocab = SimpleVocabulary(vocab_size)
    
    def _get_default_transform(self, image_size, is_training):
        """Get default image transformation pipeline."""
        if is_training:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def _build_mock_vocabulary(self) -> SimpleVocabulary:
        """Build a simple vocabulary for mock data."""
        vocab = SimpleVocabulary(self.vocab_size)
        
        # Generate some mock captions for vocabulary building
        mock_captions = [
            "a dog is running in the park",
            "a cat is sitting on a chair",
            "a person is walking down the street",
            "a car is driving on the road",
            "water is flowing in the river",
            "a building stands tall in the city",
            "a tree grows in the forest",
            "food is being prepared in the kitchen",
            "a sport is being played on the field",
            "the sky is blue with clouds"
        ] * 10  # Repeat to build a reasonable vocabulary
        
        vocab.build_vocab(mock_captions)
        return vocab
    
    def _load_captions(self) -> List[Dict[str, str]]:
        """
        Load captions from Flickr8k captions file.
        
        Supports multiple formats:
        - CSV: image,caption
        - Token format: image_id#caption_number caption_text
        Returns: List of dicts with 'image_id' and 'caption'
        """
        image_caption_pairs = []
        
        if not os.path.exists(self.captions_file):
            raise FileNotFoundError(f"Captions file not found: {self.captions_file}")
        
        with open(self.captions_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            f.seek(0)  # Reset to beginning
            
            # Check if CSV format (has header like "image,caption")
            is_csv = first_line.lower().startswith('image') and ',' in first_line
            
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                # Skip CSV header
                if is_csv and line_num == 0:
                    continue
                
                if is_csv:
                    # Parse CSV format: image,caption
                    # Example: 1000268201_693b08cb0e.jpg,A child in a pink dress...
                    parts = line.split(',', 1)
                    if len(parts) != 2:
                        continue
                    
                    image_filename = parts[0].strip()
                    caption_text = parts[1].strip()
                    
                    # Remove .jpg extension to get image_id
                    image_id = image_filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                else:
                    # Parse token format: image_id#caption_number caption_text
                    # Example: 1000268201_693b08cb0e#0	A child in a pink dress...
                    parts = line.split('\t', 1)
                    if len(parts) != 2:
                        # Try space split as fallback
                        parts = line.split(' ', 1)
                        if len(parts) != 2:
                            continue
                    
                    image_caption_id = parts[0].strip()
                    caption_text = parts[1].strip()
                    
                    # Extract image_id (remove #caption_number)
                    if '#' in image_caption_id:
                        image_id = image_caption_id.split('#')[0]
                    else:
                        image_id = image_caption_id
                
                image_caption_pairs.append({
                    'image_id': image_id,
                    'caption': caption_text
                })
        
        return image_caption_pairs
    
    def _split_dataset(self, train_split: float, val_split: float) -> List[int]:
        """
        Split dataset into train/val/test based on unique image IDs.
        
        Args:
            train_split: Fraction for training
            val_split: Fraction for validation (test gets the rest)
        
        Returns:
            List of indices for this split
        """
        # Get unique image IDs
        unique_image_ids = list(set([pair['image_id'] for pair in self.image_caption_pairs]))
        unique_image_ids.sort()  # Sort for reproducibility
        
        # Split unique image IDs
        n_total = len(unique_image_ids)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_ids = set(unique_image_ids[:n_train])
        val_ids = set(unique_image_ids[n_train:n_train + n_val])
        test_ids = set(unique_image_ids[n_train + n_val:])
        
        # Get indices for this split
        indices = []
        for idx, pair in enumerate(self.image_caption_pairs):
            image_id = pair['image_id']
            if self.split == 'train' and image_id in train_ids:
                indices.append(idx)
            elif self.split == 'val' and image_id in val_ids:
                indices.append(idx)
            elif self.split == 'test' and image_id in test_ids:
                indices.append(idx)
        
        return indices
    
    def _build_vocabulary(self) -> SimpleVocabulary:
        """Build vocabulary from all captions in the dataset."""
        vocab = SimpleVocabulary(self.vocab_size)
        
        # Collect captions for this split
        all_captions = []
        for idx in self.indices:
            caption = self.image_caption_pairs[idx]['caption']
            if caption:
                all_captions.append(caption)
        
        # Build vocabulary
        vocab.build_vocab(all_captions)
        return vocab
    
    def _get_label_from_caption(self, caption: str) -> int:
        """
        Generate classification label from caption text.
        
        Uses keyword-based classification to create meaningful labels.
        """
        caption_lower = caption.lower()
        
        # Define keyword categories (adjust based on your needs)
        category_keywords = {
            0: ['dog', 'puppy', 'canine', 'puppies'],
            1: ['cat', 'kitten', 'feline', 'kittens'],
            2: ['person', 'people', 'man', 'woman', 'boy', 'girl', 'child', 'children'],
            3: ['car', 'vehicle', 'truck', 'bus', 'automobile'],
            4: ['water', 'ocean', 'sea', 'beach', 'lake', 'river'],
            5: ['building', 'house', 'structure', 'wall', 'door'],
            6: ['tree', 'forest', 'wood', 'branch', 'leaves'],
            7: ['food', 'eating', 'meal', 'restaurant', 'kitchen'],
            8: ['sport', 'ball', 'game', 'player', 'field', 'playing'],
            9: ['sky', 'cloud', 'sun', 'sunset', 'sunrise', 'blue']
        }
        
        # Count matches for each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in caption_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or hash-based default
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            # Fallback: hash-based label for captions without keywords
            return abs(hash(caption)) % self.num_classes
    
    def _get_image_path(self, image_id: str) -> str:
        """Get full path to image file."""
        # Try common image extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            image_path = os.path.join(self.images_dir, image_id + ext)
            if os.path.exists(image_path):
                return image_path
        
        # If not found, try with image_id as-is
        image_path = os.path.join(self.images_dir, image_id)
        if os.path.exists(image_path):
            return image_path
        
        raise FileNotFoundError(f"Image not found for ID: {image_id}")
    
    def __len__(self):
        """Return the number of samples."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if self.is_mock:
            # Generate mock data
            # Create a random PIL Image (transforms expect PIL Image)
            image_array = np.random.randint(0, 256, (self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            image = self.transform(image)
            
            # Generate random caption tokens and length
            seq_len = np.random.randint(5, self.max_seq_len + 1)
            caption_tokens = torch.randint(1, self.vocab_size, (self.max_seq_len,))
            caption_length = seq_len
            
            # Generate random class label
            label = np.random.randint(0, self.num_classes)
            
            # Generate mock caption text
            mock_captions = [
                "a dog is running in the park",
                "a cat is sitting on a chair",
                "a person is walking down the street",
                "a car is driving on the road",
                "water is flowing in the river",
                "a building stands tall in the city",
                "a tree grows in the forest",
                "food is being prepared in the kitchen",
                "a sport is being played on the field",
                "the sky is blue with clouds"
            ]
            caption_text = mock_captions[label % len(mock_captions)]
            
            return {
                'image': image,
                'caption_tokens': caption_tokens,
                'caption_length': caption_length,
                'label': label,
                'idx': idx,
                'image_id': f'mock_image_{idx}',
                'caption_text': caption_text
            }
        else:
            # Use real data
            actual_idx = self.indices[idx]
            pair = self.image_caption_pairs[actual_idx]
            
            # Get image
            image_id = pair['image_id']
            image_path = self._get_image_path(image_id)
            
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                # If image loading fails, create a black image as fallback
                print(f"Warning: Could not load image {image_path}: {e}")
                image = Image.new('RGB', self.image_size, color='black')
            
            image = self.transform(image)
            
            # Get caption
            caption_text = pair['caption']
            caption_tokens, caption_length = self.vocab.encode(caption_text, self.max_seq_len)
            
            # Get label from caption
            label = self._get_label_from_caption(caption_text)
            
            return {
                'image': image,
                'caption_tokens': caption_tokens,
                'caption_length': caption_length,
                'label': label,
                'idx': idx,
                'image_id': image_id,
                'caption_text': caption_text
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
        dataset: Flickr8kDataset instance
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
