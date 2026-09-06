"""
Evaluation Module
=================
What is Model Evaluation? (Simple explanation):
During training, the model tries to minimize loss on the training data.
However, a model could cheat by simply memorizing the training sentences!
To verify that the model actually understands language patterns and generalizes well,
we test it on a separate validation dataset that it was never trained on.

Key Metrics:
1. Cross-Entropy Loss: Measures how far the model's predictions are from the true next token.
   Lower is better.
2. Perplexity (PPL): exp(loss).
   Intuitive explanation: If perplexity is 10, the model is as confused as someone
   choosing randomly between 10 equally likely words. A lower perplexity means high confidence!
"""

import math
from typing import Tuple
import torch
from model.transformer import SoulLLM


@torch.no_grad()
def evaluate_loss(
    model: SoulLLM,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str
) -> Tuple[float, float]:
    """
    Estimate cross-entropy loss and perplexity on a dataset split.
    
    Returns:
        (average_loss, perplexity)
    """
    # Put model into evaluation mode (disables dropout)
    model.eval()
    losses = torch.zeros(eval_iters)

    # Ensure dataset is long enough for at least one sequence
    n = len(data)
    effective_block_size = min(block_size, n - 2)

    for k in range(eval_iters):
        # Sample random starting indices for the batch
        if n - effective_block_size <= 0:
            break
        max_idx = n - effective_block_size - 1
        ix = torch.randint(0, max_idx + 1, (batch_size,))

        # Construct inputs (x) and targets (y = x shifted by 1)
        x = torch.stack([data[i:i + effective_block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + effective_block_size + 1] for i in ix]).to(device)

        _, loss = model(x, y)
        losses[k] = loss.item()

    # Re-enable training mode (turns dropout back on)
    model.train()

    mean_loss = losses.mean().item()
    # Perplexity = e^(loss). Cap to avoid math overflow if loss is large
    perplexity = math.exp(min(mean_loss, 20.0))

    return mean_loss, perplexity
