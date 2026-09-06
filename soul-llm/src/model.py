"""
SOUL-LLM Student Model Architecture
===================================
A lightweight decoder-only Transformer built completely from scratch in PyTorch.
This student model learns through knowledge distillation with zero external model dependencies.
It uses NO pretrained weights, NO external LLM libraries, and NO Qwen code.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism.
    Each token can only attend to previous tokens and itself (autoregressive constraint).
    """

    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combined Q, K, V linear projection for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular matrix of ones
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch, Sequence Length, Channels (Embedding Dim)

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (prevent looking into future tokens)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax probabilities
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Context output
        out = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block using Pre-LayerNorm:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, max_seq_len: int = 128):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SoulTransformer(nn.Module):
    """
    SOUL-LLM: Decoder-Only Transformer Language Model.
    Architecture:
      Tokens + Positional Embeddings
      -> N TransformerBlocks (Pre-LN)
      -> Final LayerNorm
      -> Linear Projection Head -> Logits
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 4, ff_dim: int = 512, max_seq_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, max_seq_len)
            for _ in range(num_layers)
        ])

        # Final normalization and LM projection head
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying (tie token embeddings and lm_head for parameter efficiency)
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()
        param_count = sum(p.numel() for p in self.parameters())
        print(f"[SOUL-LLM] Student model initialized: {param_count:,} parameters (Vocab: {vocab_size})")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"

        positions = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Sum token and positional representations
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)
        x = self.drop(tok_emb + pos_emb)

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=0)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 40,
                 deterministic: bool = False) -> torch.Tensor:
        """Autoregressively generate next tokens given a conditioning context."""
        for _ in range(max_new_tokens):
            # Crop sequence to maximum context length if needed
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Focus only on the last token's predictions

            if deterministic or temperature <= 0.01:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled_logits = logits / max(temperature, 0.01)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)

        return idx


# Alias for cross-module compatibility
SoulLLM = SoulTransformer
