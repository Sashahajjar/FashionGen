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
    'batch_size': 32,
    'num_epochs': 5,  # Number of epochs for training
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    
    # Data configuration
    'image_size': (224, 224),
    'max_seq_len': 50,
    'num_train_samples': 200,  # Number of mock training samples
    'num_val_samples': 50,  # Number of mock validation samples
    'val_split': 0.2,  # Validation split ratio
    
    # Training settings
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'num_workers': 0,  # Set to 0 for mock data, increase for real data
    'shuffle': True,
    
    # Checkpointing
    'save_dir': 'saved_models',
    'save_every': 1,  # Save checkpoint every N epochs
    'resume_from': None,  # Path to checkpoint to resume from
    'save_best': True,  # Save best model based on validation accuracy
}

# Paths (TODO: Update with real Fashion-Gen paths)
PATHS = {
    'images_dir': 'data/images',
    'captions_dir': 'data/captions',
    'processed_dir': 'data/processed',
    'saved_models_dir': 'saved_models',
}

