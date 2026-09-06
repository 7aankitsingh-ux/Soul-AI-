"""
Data Preparation for SOUL-LLM
==============================
Reads raw text, builds the tokenizer vocabulary, 
creates training sequences, and saves train/val splits.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, MAX_SEQ_LEN
from tokenizer import SimpleTokenizer


def prepare_data():
    """
    Pipeline:
      1. Read raw text file
      2. Build tokenizer vocabulary
      3. Encode entire text into token IDs
      4. Create overlapping sequences of length MAX_SEQ_LEN
      5. Split into train (90%) and validation (10%)
      6. Save as .pt files
    """
    print("=" * 50)
    print("SOUL-LLM Data Preparation")
    print("=" * 50)

    # Step 1: Read raw text
    raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", RAW_DATA_PATH)
    # Normalize the path
    raw_path = os.path.normpath(raw_path)
    
    # Also try relative to soul-llm dir
    if not os.path.exists(raw_path):
        raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "sample.txt")
    
    print(f"\n[1/5] Reading raw text from: {raw_path}")
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"      Text length: {len(text):,} characters")

    # Step 2: Build tokenizer
    print(f"\n[2/5] Building tokenizer vocabulary...")
    tokenizer = SimpleTokenizer()
    tokenizer.fit(text)

    # Save tokenizer
    tok_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokenizer", "vocab.json")
    tok_path = os.path.normpath(tok_path)
    tokenizer.save(tok_path)

    # Step 3: Encode text
    print(f"\n[3/5] Encoding text...")
    token_ids = tokenizer.encode(text)
    print(f"      Total tokens: {len(token_ids):,}")

    # Step 4: Create sequences
    print(f"\n[4/5] Creating training sequences (seq_len={MAX_SEQ_LEN})...")
    sequences = []
    stride = MAX_SEQ_LEN // 2  # 50% overlap for more training data
    for i in range(0, len(token_ids) - MAX_SEQ_LEN, stride):
        seq = token_ids[i : i + MAX_SEQ_LEN + 1]  # +1 for target
        sequences.append(seq)

    print(f"      Created {len(sequences)} sequences (stride={stride})")

    # Step 5: Train/Val split
    print(f"\n[5/5] Splitting into train/val...")
    data = torch.tensor(sequences, dtype=torch.long)
    n = len(data)
    split_idx = int(0.9 * n)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"      Train: {len(train_data)} sequences")
    print(f"      Val:   {len(val_data)} sequences")

    # Save
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "train.pt")
    val_path = os.path.join(base_dir, "val.pt")
    
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    print(f"\n      Saved: {train_path}")
    print(f"      Saved: {val_path}")

    # Also save vocab size info
    info = {
        "vocab_size": tokenizer.get_vocab_size(),
        "num_train": len(train_data),
        "num_val": len(val_data),
        "seq_len": MAX_SEQ_LEN,
    }
    print(f"\n{'=' * 50}")
    print(f"Data preparation complete!")
    print(f"Vocab size: {info['vocab_size']}")
    print(f"{'=' * 50}")

    return info


if __name__ == "__main__":
    prepare_data()
