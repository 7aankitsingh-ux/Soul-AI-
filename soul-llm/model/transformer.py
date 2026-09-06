"""
SOUL-LLM Transformer Model
============================
A small decoder-only Transformer trained from scratch.
No pretrained weights. No external LLM. Pure PyTorch.

Architecture:
  Input IDs → Token Embedding + Positional Embedding
            → N × TransformerBlock (LayerNorm → Attention → LayerNorm → FFN)
            → Final LayerNorm → Linear Head → Logits

This uses Pre-LayerNorm (more stable training) and residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Two linear layers with GELU activation in between.
    FFN(x) = Linear(GELU(Linear(x)))
    
    This gives the model non-linear transformation capacity
    at each position independently.
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block.
    
    Uses Pre-LayerNorm architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    
    Pre-LN is more stable for training small models from scratch
    compared to Post-LN (used in the original "Attention Is All You Need").
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, max_seq_len: int = 128):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection around attention
        x = x + self.attn(self.ln1(x))
        # Residual connection around feed-forward
        x = x + self.ffn(self.ln2(x))
        return x


class SoulTransformer(nn.Module):
    """
    SOUL-LLM: The complete decoder-only Transformer.
    
    This is the main model class. It combines:
      - Token embeddings (learned lookup table)
      - Positional embeddings (learned, not sinusoidal)
      - A stack of TransformerBlocks
      - A final linear head that predicts the next token
    
    Args:
        vocab_size: Number of tokens in the vocabulary
        embed_dim: Dimension of token/position embeddings
        num_heads: Number of attention heads per block
        num_layers: Number of TransformerBlock layers
        ff_dim: Hidden dimension of the feed-forward network
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 4, ff_dim: int = 512, max_seq_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()

        self.max_seq_len = max_seq_len

        # Token embedding: maps each token ID → a dense vector
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embedding: encodes position information
        # (learned, not sinusoidal — simpler for small models)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.drop = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, max_seq_len)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(embed_dim)

        # Language model head: projects back to vocabulary size
        # Output logits for each token in the vocabulary
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and LM head
        # This is a common technique that improves performance and reduces parameters
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[SOUL-LLM] Model initialized: {n_params:,} parameters")

    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot uniform."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            targets: Target token IDs for computing loss (optional)
        
        Returns:
            logits: Shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss (only if targets provided)
        """
        B, T = input_ids.size()
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        # Create position indices: [0, 1, 2, ..., T-1]
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)

        # Embed tokens and positions, then add them
        tok_emb = self.token_embedding(input_ids)   # (B, T, embed_dim)
        pos_emb = self.position_embedding(positions) # (T, embed_dim)
        x = self.drop(tok_emb + pos_emb)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_final(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy: (B*T, vocab_size) vs (B*T,)
            # NOTE: use reshape (not view) — targets may be a non-contiguous
            # slice like data[:, 1:], and view() would raise on strided tensors.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0  # Ignore padding tokens
            )

        return logits, loss
