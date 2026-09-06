"""
SOUL-LLM Training Script (distillation)
=======================================
Trains the SOUL-LLM Transformer on the Qwen-teacher dataset using the REAL
Trainer (forward -> loss -> backward -> optimizer.step). Reuses the same
dataset builder and tokenizer as the background worker, so training and
inference always agree on the tokenizer/vocabulary.

Usage:
    python training/train.py
    python training/train.py --epochs 20 --batch-size 32
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, DROPOUT, EMBED_DIM, NUM_HEADS,
    NUM_LAYERS, FF_DIM, MAX_SEQ_LEN, GRAD_CLIP, CHECKPOINT_DIR,
)
from model import SoulTransformer
from tokenizer import SimpleTokenizer
from training.trainer import Trainer
from data.build_dataset import build_dataset

# Honest statement about the architecture (shown in logs + UI).
ARCHITECTURE_NOTICE = (
    "SOUL-LLM is an educational decoder-only Transformer (~2M params) trained "
    "from scratch. It cannot match a 7B teacher. It learns best-effort "
    "character-level language from the distilled Q&A; quality improves with "
    "more teacher data but stays far below Qwen."
)


def main():
    parser = argparse.ArgumentParser(description="Train SOUL-LLM on Qwen-teacher data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Skip dataset rebuild (reuse data/train.pt, data/val.pt)")
    args = parser.parse_args()

    print("=" * 62)
    print("   SOUL-LLM Student Training  (teacher: local Qwen 2.5 7B)")
    print("=" * 62)
    print(f"   Device: {DEVICE}   (cuda available: {torch.cuda.is_available()})")
    print(f"   Epochs: {args.epochs}   Batch size: {args.batch_size}   LR: {args.lr}")
    print(ARCHITECTURE_NOTICE)
    print("=" * 62)

    if args.no_rebuild:
        train_data = torch.load("data/train.pt", weights_only=True)
        val_data = torch.load("data/val.pt", weights_only=True)
        tok = SimpleTokenizer()
        tok.load("tokenizer/vocab.json")
        info = {"vocab_size": tok.get_vocab_size()}
        # corpus hash unknown when not rebuilding -> disable resume
        info["corpus_hash"] = None
    else:
        info = build_dataset()
        train_data = torch.load("data/train.pt", weights_only=True)
        val_data = torch.load("data/val.pt", weights_only=True)

    vocab_size = info["vocab_size"]
    print(f"\n[Dataset] {info.get('dataset_size', '?')} teacher examples | "
          f"vocab {vocab_size} | sequences {info.get('sequences', train_data.shape[0])}")

    model = SoulTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    trainer = Trainer(model, device=DEVICE, lr=args.lr,
                      grad_clip=GRAD_CLIP, log_fn=print)

    if not args.no_rebuild:
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        if os.path.exists(best_path):
            ckpt = Trainer.load_checkpoint(best_path, DEVICE)
            trainer.resume(ckpt, info["corpus_hash"], vocab_size)

    train_loader, val_loader = Trainer.make_data_loaders(
        train_data, val_data, args.batch_size
    )
    print(f"\n[Training] {len(train_loader)} batches/epoch, {len(val_loader)} val batches")

    summary = trainer.run(
        train_loader=train_loader, val_loader=val_loader,
        corpus_hash=info.get("corpus_hash"), vocab_size=vocab_size,
        num_epochs=args.epochs,
        save_dir=CHECKPOINT_DIR,
        save_every_steps=1000,
    )

    print("\n" + "=" * 62)
    print(f"   Training complete. Best val loss: {summary['best_val_loss']:.4f}")
    print(f"   Steps: {summary['steps']}   Saved: {summary['checkpoints']}")
    print("=" * 62)


if __name__ == "__main__":
    main()