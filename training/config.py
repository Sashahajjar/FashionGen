"""
Configuration file for training

This module contains all hyperparameters and configuration settings for training.
"""

# Model configuration
MODEL_CONFIG = {
    'cnn_feature_dim': 512,
    'rnn_feature_dim': 512,
    'fusion_dim': 512,
    'output_dim': 256,
    'fusion_method': 'concat',  # 'concat', 'add', or 'multiply'
    
    # Classification
    'num_classes': 10,  # Number of clothing categories (mock: 10 classes)
    
    # RNN specific
    'vocab_size': 10000,
    'embedding_dim': 256,
    'hidden_dim': 512,
    'num_layers': 2,
    'rnn_type': 'LSTM',  # 'LSTM' or 'GRU'
    'dropout': 0.3,
    
    # CNN specific
    'pretrained': True,
}

# Training configuration
TRAIN_CONFIG = {
    # Batch size: 16 for CPU/Colab, 32 if GPU memory allows
    'batch_size': 16,
    
    # Number of epochs: 5 for architecture validation, max 10 if needed
    # This project focuses on validating CNN + RNN + fusion architecture, not full convergence
    'num_epochs': 5,
    'max_epochs': 10,  # Maximum epochs if early stopping doesn't trigger
    
    # Learning rate: 1e-4 is good for fine-tuning pretrained models
    'learning_rate': 1e-4,
    
    # Learning rate scheduler: Optional, can disable for quick validation
    'use_lr_scheduler': False,  # Disabled for quick validation runs
    'lr_scheduler_patience': 3,  # Reduce LR if no improvement for 3 epochs
    'lr_scheduler_factor': 0.5,  # Multiply LR by this factor
    'lr_scheduler_min': 1e-6,  # Minimum learning rate
    
    # Weight decay for regularization (L2)
    'weight_decay': 1e-5,
    
    # Data configuration
    'image_size': (224, 224),
    'max_seq_len': 50,
    
    # For mock data only (ignored when real HDF5 data is loaded)
    'num_train_samples': 32,  # Mock data only
    'num_val_samples': 8,  # Mock data only
    'val_split': 0.2,  # Validation split ratio
    
    # Training settings
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'num_workers': 0,  # Set to 0 for Colab/CPU, 2-4 for GPU
    'shuffle': True,
    
    # CNN freezing: Freeze pretrained weights initially (faster training, less memory)
    'freeze_cnn': True,  # Freeze pretrained CNN weights
    'freeze_cnn_layers': 'all',  # 'all' to freeze all, or 'partial' to fine-tune only top layers
    
    # Early stopping: Disabled by default for full learning curves (academic projects)
    # Enable with --early_stop flag or set use_early_stopping=True for long runs
    'use_early_stopping': False,  # Disabled by default - always run full epochs for learning curves
    'early_stopping_patience': 3,  # Stop if no improvement for N epochs (only if enabled)
    'early_stopping_min_delta': 0.001,  # Minimum change to qualify as improvement
    'early_stopping_metric': 'loss',  # 'loss' or 'accuracy' - use 'loss' for validation
    
    # Checkpointing
    'save_dir': 'saved_models',
    'save_every': 1,  # Save checkpoint every N epochs
    'resume_from': None,  # Path to checkpoint to resume from
    'save_best': True,  # Save best model based on validation loss
}

# Paths (TODO: Update with real Fashion-Gen paths)
PATHS = {
    'images_dir': 'data/images',
    'captions_dir': 'data/captions',
    'processed_dir': 'data/processed',
    'saved_models_dir': 'saved_models',
}

