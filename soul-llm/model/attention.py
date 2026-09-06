"""
Multi-Head Causal Self-Attention for SOUL-LLM
==============================================
Implements scaled dot-product attention with a causal mask.
This is the core mechanism that allows the model to "attend" 
to previous tokens in the sequence.

How it works:
  1. Project input into Query, Key, Value vectors
  2. Compute attention scores = Q @ K^T / sqrt(d_k)
  3. Apply causal mask (prevent attending to future tokens)
  4. Softmax → weighted sum of Values
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    
    Args:
        embed_dim: Dimensionality of the input embeddings
        num_heads: Number of parallel attention heads
        dropout: Dropout probability for attention weights
        max_seq_len: Maximum sequence length (for the causal mask)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 128):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Pre-compute causal mask (lower-triangular)
        # This prevents the model from "seeing the future"
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Step 1: Compute Q, K, V all at once
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, T, C)

        # Step 2: Reshape for multi-head attention
        # (B, T, C) → (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Scaled dot-product attention
        # Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        scale = math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Step 4: Apply causal mask
        # Set future positions to -infinity so softmax gives them ~0 weight
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        # Step 5: Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: Weighted sum of values
        out = attn_weights @ v  # (B, H, T, head_dim)

        # Step 7: Reshape back and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out
