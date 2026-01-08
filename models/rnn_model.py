"""
RNN Model for Text Feature Extraction

This module implements an RNN (LSTM/GRU) that extracts features from
text captions using bidirectional LSTM/GRU.
"""

import torch
import torch.nn as nn


class TextRNN(nn.Module):
    """
    RNN model for extracting features from text captions.
    
    Input: Token IDs of shape (B, seq_len) and lengths of shape (B,)
    Output: Feature vectors of shape (B, feature_dim)
    """
    
    def __init__(
        self,
        vocab_size=10000,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        feature_dim=512,
        rnn_type='LSTM',
        dropout=0.3
    ):
        """
        Initialize the RNN model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of RNN
            num_layers: Number of RNN layers
            feature_dim: Dimension of the output feature vector
            rnn_type: Type of RNN ('LSTM' or 'GRU')
            dropout: Dropout probability
        """
        super(TextRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
        else:  # GRU
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
        
        # Projection layer to get desired feature dimension
        # Bidirectional RNN outputs 2 * hidden_dim
        self.projection = nn.Linear(2 * hidden_dim, feature_dim)
        self.relu = nn.ReLU()
        
    def forward(self, token_ids, lengths):
        """
        Forward pass through the RNN.
        
        Args:
            token_ids: Tensor of shape (B, seq_len) containing token IDs
            lengths: Tensor of shape (B,) containing actual sequence lengths
        
        Returns:
            features: Tensor of shape (B, feature_dim)
        """
        # Embed tokens: (B, seq_len) -> (B, seq_len, embedding_dim)
        embedded = self.embedding(token_ids)
        
        # Pack sequences for efficient processing
        # Note: lengths must be on CPU for pack_padded_sequence
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        
        # Pass through RNN
        packed_output, (hidden, cell) = self.rnn(packed)
        
        # Get the last hidden state from both directions
        if self.rnn_type == 'LSTM':
            # hidden shape: (num_layers * 2, B, hidden_dim) for bidirectional
            # Take the last layer's forward and backward hidden states
            forward_hidden = hidden[-2]  # Last forward layer
            backward_hidden = hidden[-1]  # Last backward layer
            # Concatenate: (B, hidden_dim) + (B, hidden_dim) -> (B, 2*hidden_dim)
            combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:  # GRU
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Project to desired dimension: (B, 2*hidden_dim) -> (B, feature_dim)
        features = self.projection(combined_hidden)
        features = self.relu(features)
        
        return features


def create_rnn_model(
    vocab_size=10000,
    embedding_dim=256,
    hidden_dim=512,
    num_layers=2,
    feature_dim=512,
    rnn_type='LSTM',
    dropout=0.3
):
    """
    Factory function to create an RNN model instance.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Hidden dimension of RNN
        num_layers: Number of RNN layers
        feature_dim: Dimension of the output feature vector
        rnn_type: Type of RNN ('LSTM' or 'GRU')
        dropout: Dropout probability
    
    Returns:
        TextRNN model instance
    """
    return TextRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        feature_dim=feature_dim,
        rnn_type=rnn_type,
        dropout=dropout
    )

