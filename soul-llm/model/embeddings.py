"""
Embeddings Module
=================
What are Embeddings? (Simple explanation):
1. Token Embeddings:
   Each word or character token is just an integer ID (like 42).
   A token embedding looks up this integer in a table of size (vocab_size, d_model).
   It converts the single integer into a rich vector of 256 floating-point numbers.
   During training, the model learns to place similar concepts close together in this vector space.

2. Positional Embeddings:
   Unlike Recurrent Neural Networks (RNNs) that read text step-by-step from left to right,
   Transformers process all tokens simultaneously in parallel!
   Because of this, the Transformer has no inherent sense of order.
   The sentence "Dog bites man" and "Man bites dog" would look identical without positions.
   Positional embeddings provide a unique learned vector for each position index (0, 1, 2, ...),
   telling the model exactly where each token is located in the sequence.

Combined Input:
   Final Vector = Token_Embedding + Positional_Embedding
"""

import math
import torch
import torch.nn as nn
from config import SoulConfig


class SoulEmbeddings(nn.Module):
    """
    Combines token embeddings and learnable positional embeddings with dropout.
    """

    def __init__(self, config: SoulConfig):
        super().__init__()
        self.config = config

        # Token embedding lookup table: Maps token_id -> vector of size d_model
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # Learnable positional embedding table: Maps position_index (0..block_size-1) -> vector of size d_model
        self.pos_embeddings = nn.Embedding(config.block_size, config.d_model)

        # Dropout layer to randomly zero out some values during training to prevent memorization
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: Tensor of token IDs with shape (Batch_size, Sequence_length) [B, T]
        
        Returns:
            Tensor of embedded representations with shape (B, T, d_model)
        """
        device = idx.device
        b, t = idx.size()

        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}."
        )

        # Generate position indices: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (T)

        # Retrieve token embeddings: (B, T) -> (B, T, d_model)
        tok_emb = self.tok_embeddings(idx)

        # Retrieve position embeddings: (T) -> (T, d_model)
        pos_emb = self.pos_embeddings(pos)

        # Add token representations and position representations element-wise: (B, T, d_model)
        x = self.drop(tok_emb + pos_emb)
        return x
