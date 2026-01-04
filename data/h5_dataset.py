"""
HDF5 Dataset Loader for Fashion-Gen Data

This module implements loading Fashion-Gen data from HDF5 files.
The dataset is available at: https://www.kaggle.com/datasets/bothin/fashiongen-validation/data
"""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
import re


class FashionGenH5Dataset(Dataset):
    """
    Dataset class for Fashion-Gen HDF5 data.
    
    Loads images and captions from HDF5 files downloaded from Kaggle.
    """
    
    def __init__(
        self,
        h5_file_path,
        image_size=(224, 224),
        max_seq_len=50,
        vocab_size=10000,
        split='train',
        transform=None,
        build_vocab=True,
        vocab=None,
        max_samples=None,
        num_classes=10
    ):
        """
        Initialize the HDF5 dataset.
        
        Args:
            h5_file_path: Path to the HDF5 file (e.g., 'fashiongen_256_256_train.h5')
            image_size: Tuple of (height, width) for images
            max_seq_len: Maximum sequence length for captions
            vocab_size: Maximum vocabulary size
            split: 'train', 'val', or 'test'
            transform: Optional image transform (if None, uses default)
            build_vocab: Whether to build vocabulary from captions
            vocab: Pre-built vocabulary (if None, builds from data)
        """
        self.h5_file_path = h5_file_path
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.split = split
        self.num_classes = num_classes
        
        # Set up image transforms
        if transform is None:
            self.transform = self._get_default_transform(image_size, split == 'train')
        else:
            self.transform = transform
        
        # Open HDF5 file
        self.h5_file = h5py.File(h5_file_path, 'r')
        
        # Load data indices
        # Fashion-Gen HDF5 structure typically has:
        # - 'images': image data
        # - 'captions': caption text
        # - 'categories': category labels (if available)
        
        # Get dataset size
        if 'images' in self.h5_file:
            full_size = len(self.h5_file['images'])
        elif 'input_image' in self.h5_file:
            full_size = len(self.h5_file['input_image'])
        else:
            # Try to infer from first available dataset
            first_key = list(self.h5_file.keys())[0]
            full_size = len(self.h5_file[first_key])
        
        # Use entire file - no internal splitting
        # FashionGen provides separate train/val files, so we use the whole file
        self.full_size = full_size
        self.max_samples = max_samples if max_samples else full_size
        self.split = split
        
        # Use entire file (no splitting)
        if max_samples and max_samples < full_size:
            self.num_samples = max_samples
            print(f"Using {self.num_samples} samples for {split} split (from {max_samples} max, {full_size} total)")
        else:
            self.num_samples = full_size
            print(f"Using {self.num_samples} samples for {split} split (from {full_size} total)")
        
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
    
    def _build_vocabulary(self):
        """Build vocabulary from all captions in the dataset."""
        vocab = SimpleVocabulary(self.vocab_size)
        
        # Collect all captions
        all_captions = []
        captions_key = None
        
        # Find captions key
        for key in ['captions', 'input_description', 'description', 'text']:
            if key in self.h5_file:
                captions_key = key
                break
        
        if captions_key is None:
            print("Warning: Could not find captions in HDF5 file. Using placeholder vocabulary.")
            return vocab
        
        # Read captions
        captions_data = self.h5_file[captions_key]
        
        # Handle different HDF5 structures
        if isinstance(captions_data, h5py.Dataset):
            # Direct dataset
            for i in range(min(10000, self.num_samples)):  # Sample for vocab building
                caption = self._get_caption_text(i, captions_key)
                if caption:
                    all_captions.append(caption)
        else:
            # Group or other structure
            for i in range(min(10000, self.num_samples)):
                caption = self._get_caption_text(i, captions_key)
                if caption:
                    all_captions.append(caption)
        
        # Build vocabulary
        vocab.build_vocab(all_captions)
        return vocab
    
    def _get_caption_text(self, idx, captions_key):
        """Get caption text for a given index."""
        try:
            captions_data = self.h5_file[captions_key]
            caption = captions_data[idx]
            
            # Handle numpy array with bytes (HDF5 stores as |S400)
            if isinstance(caption, np.ndarray):
                if caption.dtype.kind == 'S':  # String/bytes array
                    caption = caption[0] if len(caption) > 0 else b''
            
            # Decode bytes to string
            if isinstance(caption, (bytes, np.bytes_)):
                caption = caption.decode('utf-8')
            elif not isinstance(caption, str):
                caption = str(caption)
            
            return caption.strip() if caption else ""
        except:
            return ""
    
    def _get_image(self, idx):
        """Get image at given index."""
        # Try different possible keys for images
        image_keys = ['images', 'input_image', 'image', 'data']
        image_data = None
        image_key = None
        
        for key in image_keys:
            if key in self.h5_file:
                image_key = key
                break
        
        if image_key is None:
            raise ValueError("Could not find image data in HDF5 file")
        
        # Get image data
        img_array = self.h5_file[image_key][idx]
        
        # Handle different image formats
        if isinstance(img_array, np.ndarray):
            # Convert to PIL Image
            if img_array.dtype == np.uint8:
                # Normalize to [0, 255] range
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
            else:
                # Normalize float arrays
                if img_array.max() > 1.0:
                    img_array = img_array.astype(np.uint8)
                else:
                    img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
        else:
            # Try to convert
            img_array = np.array(img_array)
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    
    def _get_category(self, idx):
        """Get category label for given index."""
        # Try different possible keys for categories
        category_keys = ['input_category', 'categories', 'category', 'label', 'labels', 'class']
        
        for key in category_keys:
            if key in self.h5_file:
                try:
                    category = self.h5_file[key][idx]
                    
                    # Handle numpy array with bytes (common in HDF5)
                    if isinstance(category, np.ndarray):
                        if category.dtype.kind == 'S':  # String/bytes array
                            category = category[0] if len(category) > 0 else b''
                        else:
                            return int(category[0]) if len(category) > 0 else 0
                    
                    # Handle bytes
                    if isinstance(category, (bytes, np.bytes_)):
                        cat_str = category.decode('utf-8') if isinstance(category, bytes) else str(category, 'utf-8')
                        # Hash string to integer (consistent mapping)
                        return abs(hash(cat_str)) % (self.num_classes if hasattr(self, 'num_classes') else 10)
                    
                    # Handle string
                    if isinstance(category, str):
                        return abs(hash(category)) % (self.num_classes if hasattr(self, 'num_classes') else 10)
                    
                    # Try to convert to int
                    return int(category)
                except:
                    continue
        
        # Default: return 0 if no category found
        return 0
    
    def __len__(self):
        """Return the number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        # Use entire file - no splitting
        # FashionGen provides separate train/val files
        actual_idx = idx
        # Ensure we don't go out of bounds
        if actual_idx >= self.full_size:
            actual_idx = self.full_size - 1
        
        # Get image
        image = self._get_image(actual_idx)
        image = self.transform(image)
        
        # Get caption
        captions_key = None
        for key in ['captions', 'input_description', 'description', 'text']:
            if key in self.h5_file:
                captions_key = key
                break
        
        if captions_key:
            caption_text = self._get_caption_text(actual_idx, captions_key)
            if caption_text:
                caption_tokens, caption_length = self.vocab.encode(caption_text, self.max_seq_len)
            else:
                # Fallback: empty caption
                caption_tokens = torch.zeros(self.max_seq_len, dtype=torch.long)
                caption_length = 0
        else:
            caption_tokens = torch.zeros(self.max_seq_len, dtype=torch.long)
            caption_length = 0
        
        # Get category/label
        label = self._get_category(actual_idx)
        
        return {
            'image': image,
            'caption_tokens': caption_tokens,
            'caption_length': caption_length,
            'label': label,
            'idx': idx
        }
    
    def __del__(self):
        """Close HDF5 file when dataset is deleted."""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()


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


