"""
Checkpoint Management Module
============================
What is a Checkpoint? (Simple explanation):
Training a neural network changes the internal weight numbers step-by-step.
A checkpoint is a saved snapshot of:
1. The model's weights (`state_dict`)
2. The optimizer's internal memory (learning rate, momentum)
3. The training epoch and step count
4. The best validation loss achieved so far
5. The model's configuration parameters

This allows:
- Resuming training seamlessly if interrupted
- Loading the best model for inference or API serving
"""

import os
from typing import Optional, Tuple
import torch
from config import SoulConfig
from model.transformer import SoulLLM


def save_checkpoint(
    model: SoulLLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    val_loss: float,
    config: SoulConfig,
    filepath: str,
    is_best: bool = False
):
    """
    Save model weights, optimizer state, and training metadata to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "config": config,
        "is_best": is_best
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    device: str = "cpu"
) -> Tuple[SoulLLM, SoulConfig, int, float]:
    """
    Load a saved checkpoint and reconstruct the SoulLLM model.
    
    Returns:
        (model, config, epoch, val_loss)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint.get("config", SoulConfig())
    state_dict = checkpoint["model_state_dict"]

    # Match vocab_size to trained checkpoint layer shape
    vocab_size = checkpoint.get("vocab_size", 70)
    if "lm_head.weight" in state_dict:
        vocab_size = state_dict["lm_head.weight"].shape[0]

    # Instantiate fresh model architecture with the saved configuration
    if isinstance(config, SoulConfig):
        config.vocab_size = vocab_size
        model = SoulLLM(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
    elif isinstance(config, dict):
        config["vocab_size"] = vocab_size
        model = SoulLLM(
            vocab_size=vocab_size,
            embed_dim=config.get("d_model", config.get("embed_dim", 256)),
            num_heads=config.get("n_head", config.get("num_heads", 4)),
            num_layers=config.get("n_layer", config.get("num_layers", 4)),
            ff_dim=config.get("ff_dim", 512),
            max_seq_len=config.get("block_size", config.get("max_seq_len", 128)),
            dropout=config.get("dropout", 0.1)
        )
    else:
        model = SoulLLM(vocab_size=vocab_size)
    model.load_state_dict(state_dict)
    model.to(device)

    epoch = checkpoint.get("epoch", 0)
    val_loss = checkpoint.get("val_loss", checkpoint.get("loss", float("inf")))

    return model, config, epoch, val_loss
