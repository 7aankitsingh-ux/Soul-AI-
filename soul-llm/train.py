"""
SOUL-LLM Training Script
========================
Trains the SOUL-LLM decoder-only Transformer from scratch using PyTorch.
Can train on raw educational text, teacher-distilled text, or both combined.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import SoulTransformer
from src.dataset import load_corpus, prepare_dataloaders
from tokenizer.tokenizer import SimpleTokenizer


def load_yaml_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def train(
    config_path: str = "config.yaml",
    extra_corpus_path: str = None,
    epochs: int = None
):
    cfg = load_yaml_config(config_path)

    student_cfg = cfg.get("student", {})
    dist_cfg = cfg.get("distillation", {})
    paths_cfg = cfg.get("paths", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== SOUL-LLM Training Pipeline ===")
    print(f"Device: {device.upper()}")

    # 1. Collect training text corpora
    corpus_files = [paths_cfg.get("raw_data", "data/raw/sample.txt")]
    teacher_file = paths_cfg.get("teacher_data_file", "data/teacher_generated/distilled_data.txt")
    if os.path.exists(teacher_file) and teacher_file not in corpus_files:
        corpus_files.append(teacher_file)
    if extra_corpus_path and os.path.exists(extra_corpus_path) and extra_corpus_path not in corpus_files:
        corpus_files.append(extra_corpus_path)

    corpus_text = load_corpus(corpus_files)
    if not corpus_text.strip():
        raise ValueError(f"No training data found in paths: {corpus_files}")

    # 2. Tokenizer setup
    vocab_path = paths_cfg.get("vocab_file", "tokenizer/vocab.json")
    tokenizer = SimpleTokenizer()
    if os.path.exists(vocab_path):
        tokenizer.load(vocab_path)
    # Ensure vocabulary covers new tokens in the corpus
    tokenizer.fit(corpus_text)
    tokenizer.save(vocab_path)

    vocab_size = tokenizer.get_vocab_size()

    # 3. Dataloaders
    block_size = student_cfg.get("block_size", 128)
    batch_size = dist_cfg.get("batch_size", 32)
    train_loader, val_loader = prepare_dataloaders(
        corpus_text, tokenizer, block_size=block_size, batch_size=batch_size
    )

    # 4. Student Model
    model = SoulTransformer(
        vocab_size=vocab_size,
        embed_dim=student_cfg.get("d_model", 256),
        num_heads=student_cfg.get("n_head", 4),
        num_layers=student_cfg.get("n_layer", 4),
        ff_dim=student_cfg.get("ff_dim", 512),
        max_seq_len=block_size,
        dropout=student_cfg.get("dropout", 0.1)
    ).to(device)

    # 5. Optimizer & Hyperparameters
    lr = float(dist_cfg.get("learning_rate", 3e-4))
    weight_decay = float(dist_cfg.get("weight_decay", 0.01))
    total_epochs = epochs or dist_cfg.get("epochs", 25)
    grad_clip = float(dist_cfg.get("grad_clip", 1.0))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Checkpoint saving path
    save_dir = paths_cfg.get("checkpoints_dir", "model/soul-llm")
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best_model.pt")

    best_val_loss = float("inf")
    print(f"\n[Training] Beginning {total_epochs} training epochs...")

    for epoch in range(1, total_epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / max(1, train_batches)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, targets=y)
                total_val_loss += loss.item()
                val_batches += 1
        avg_val_loss = total_val_loss / max(1, val_batches)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
                "vocab_size": vocab_size,
                "config": student_cfg
            }
            torch.save(ckpt_payload, best_ckpt_path)
            # Also keep checkpoints/best_model.pt synchronized
            alt_ckpt_dir = os.path.join(ROOT_DIR, "checkpoints")
            os.makedirs(alt_ckpt_dir, exist_ok=True)
            alt_ckpt_path = os.path.join(alt_ckpt_dir, "best_model.pt")
            torch.save(ckpt_payload, alt_ckpt_path)

        star = " * (Best Saved)" if is_best else ""
        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"  Epoch [{epoch:02d}/{total_epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{star}", flush=True)

    print(f"\n[OK] Training complete! Best validation loss: {best_val_loss:.4f}", flush=True)
    print(f"[OK] Best model checkpoint saved to: {best_ckpt_path}", flush=True)
    return best_ckpt_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SOUL-LLM from scratch")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    train(config_path=args.config, epochs=args.epochs)
