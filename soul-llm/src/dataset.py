"""
SOUL-LLM Dataset & DataLoader Module
====================================
Prepares tokenized sequences for training and distillation.
Handles slicing sequences into input (x) and next-token target (y) pairs.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


class TextTokenDataset(Dataset):
    """
    PyTorch Dataset for causal autoregressive language modeling.
    Extracts continuous chunks of length (block_size + 1):
      x = chunk[:-1]
      y = chunk[1:]
    """

    def __init__(self, token_ids: List[int], block_size: int = 128, stride: Optional[int] = None):
        self.block_size = block_size
        self.stride = stride or max(1, block_size // 2)
        self.data = torch.tensor(token_ids, dtype=torch.long)
        if len(self.data) <= block_size:
            self.indices = [0]
        else:
            self.indices = list(range(0, len(self.data) - block_size, self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        chunk = self.data[start : start + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            # Pad if at the very end
            pad_len = (self.block_size + 1) - len(chunk)
            chunk = torch.cat([chunk, torch.zeros(pad_len, dtype=torch.long)])

        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_corpus(paths: List[str]) -> str:
    """Loads and concatenates text from multiple file paths."""
    corpus_parts = []
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    corpus_parts.append(content)
                    print(f"[Dataset] Loaded {len(content):,} characters from {p}")
    return "\n\n".join(corpus_parts)


def prepare_dataloaders(
    corpus_text: str,
    tokenizer,
    block_size: int = 128,
    batch_size: int = 32,
    train_ratio: float = 0.9
) -> Tuple[DataLoader, DataLoader]:
    """Encodes corpus, splits into train/val sets, and returns DataLoaders."""
    token_ids = tokenizer.encode(corpus_text)

    split_idx = int(len(token_ids) * train_ratio)
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:] if split_idx < len(token_ids) else token_ids[-block_size:]

    train_dataset = TextTokenDataset(train_tokens, block_size=block_size)
    val_dataset = TextTokenDataset(val_tokens, block_size=block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"[Dataset] Total tokens: {len(token_ids):,} (Train: {len(train_tokens):,}, Val: {len(val_tokens):,})")
    return train_loader, val_loader
