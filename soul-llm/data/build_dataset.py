"""
Build SOUL-LLM training data from Qwen-teacher Q&A (distillation).
===============================================================
Reads the teacher dataset (data/qwen_teacher.jsonl) written by the main
SOUL AI agent, formats each example as a causal-LM block:

    User: <prompt>
    Assistant: <response>

re-builds the tokenizer vocabulary from the full corpus, creates the
train/val sequence tensors, and saves them to data/train.pt / val.pt.

This module is shared by the background worker (training/background_worker.py),
the CLI (training/train.py) and evaluation (evaluation/evaluate.py) so that
training and inference always use the EXACT same tokenizer (tokenizer/vocab.json).
"""

import hashlib
import json
import os
import sys
import torch

# Add parent directory to path so `config` and `tokenizer` import cleanly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import CORPUS_FILE, DATA_DIR, MAX_SEQ_LEN
from tokenizer import SimpleTokenizer

QWEN_TEACHER_JSONL = os.path.join(DATA_DIR, "qwen_teacher.jsonl")
BASE_TEXT_FILE = os.path.join(DATA_DIR, "raw", "base.txt")
VOCAB_PATH = os.path.join(BASE_DIR, "tokenizer", "vocab.json")
TRAIN_PATH = os.path.join(DATA_DIR, "train.pt")
VAL_PATH = os.path.join(DATA_DIR, "val.pt")


def load_examples(path: str = QWEN_TEACHER_JSONL):
    """Read teacher Q&A examples. Accepts both new (prompt/response) and
    legacy (question/answer) field names."""
    if not os.path.exists(path):
        return []
    examples = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = (obj.get("prompt") or obj.get("question") or "").strip()
            response = (obj.get("response") or obj.get("answer") or "").strip()
            if not prompt or not response:
                continue
            examples.append({
                "prompt": prompt,
                "response": response,
                "timestamp": obj.get("timestamp") or obj.get("capturedAt"),
                "model": obj.get("model") or "unknown",
            })
    return examples


def format_example(prompt: str, response: str) -> str:
    """One causal-LM training block. The model learns to continue after
    'Assistant:' given the user prompt."""
    return f"User: {prompt}\nAssistant: {response}\n"


def load_base_text() -> str:
    # Pristine, human-written seed corpus (kept so a fresh model can learn
    # basic spelling/syntax even before any teacher Q&A has accumulated).
    if os.path.exists(BASE_TEXT_FILE):
        with open(BASE_TEXT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def build_corpus_text(examples) -> str:
    blocks = [format_example(e["prompt"], e["response"]) for e in examples]
    base = load_base_text()
    text = base.rstrip()
    if blocks:
        text += "\n\n" + "\n\n".join(blocks) + "\n"
    return text


def build_dataset(write_corpus: bool = True):
    """
    Returns a dict describing the built dataset, or raises ValueError when
    there is nothing to train on (empty corpus).
    """
    examples = load_examples()
    text = build_corpus_text(examples)
    corpus_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    if not text.strip():
        raise ValueError(
            "Training corpus is empty. Collect teacher examples first "
            "(send messages to Qwen with collection enabled)."
        )

    if write_corpus:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CORPUS_FILE, "w", encoding="utf-8") as f:
            f.write(text)

    # Tokenizer must produce the SAME vocab for training and inference.
    tokenizer = SimpleTokenizer()
    tokenizer.fit(text)
    tokenizer.save(VOCAB_PATH)

    token_ids = tokenizer.encode(text)
    if len(token_ids) <= MAX_SEQ_LEN + 1:
        raise ValueError(
            f"Corpus ({len(token_ids)} tokens) is too short for seq_len={MAX_SEQ_LEN}. "
            "Collect more teacher examples."
        )

    sequences = []
    stride = max(MAX_SEQ_LEN // 2, 1)
    for i in range(0, len(token_ids) - MAX_SEQ_LEN, stride):
        sequences.append(token_ids[i: i + MAX_SEQ_LEN + 1])

    data = torch.tensor(sequences, dtype=torch.long)
    n = len(data)
    if n < 8:
        raise ValueError(f"Only {n} sequences — need at least 8. Collect more examples.")

    split_idx = int(0.9 * n)
    train_data, val_data = data[:split_idx], data[split_idx:]
    if len(val_data) == 0:
        val_data = train_data[-1:].clone()

    torch.save(train_data, TRAIN_PATH)
    torch.save(val_data, VAL_PATH)

    info = {
        "dataset_size": len(examples),
        "corpus_chars": len(text),
        "corpus_hash": corpus_hash,
        "token_count": len(token_ids),
        "vocab_size": tokenizer.get_vocab_size(),
        "sequences": int(n),
        "num_train": int(len(train_data)),
        "num_val": int(len(val_data)),
        "seq_len": MAX_SEQ_LEN,
    }

    print("[Dataset]", json.dumps(info))
    return info


if __name__ == "__main__":
    build_dataset()