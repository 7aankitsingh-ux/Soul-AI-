"""
SOUL-LLM Auto-Retrain
=====================
Distills captured chat Q&A (from the main SOUL AI agent) into SOUL-LLM.

Pipeline:
  1. Read pending conversations from ../data/collected/conversations.jsonl
  2. Append them to the cumulative growth corpus
  3. Rebuild the training corpus as: base.txt (pristine) + corpus_growth.txt (learned)
  4. Run the standard data prepare + train pipeline
  5. Ask the running API (localhost:8000) to reload the new checkpoint

Usage:
    python training/retrain.py
    python training/retrain.py --epochs 20
"""

import argparse
import json
import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLLECTED_DIR = os.path.join(BASE_DIR, "data", "collected")
COLLECTED_FILE = os.path.join(COLLECTED_DIR, "conversations.jsonl")
GROWTH_FILE = os.path.join(COLLECTED_DIR, "corpus_growth.txt")
BASE_TEXT_FILE = os.path.join(BASE_DIR, "data", "raw", "base.txt")
RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "sample.txt")
API_URL = "http://localhost:8000/api/reload"


def load_pending_conversations():
    if not os.path.exists(COLLECTED_FILE):
        return []
    entries = []
    with open(COLLECTED_FILE, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def absorb_conversations(entries):
    """Append pending Q&A to the cumulative corpus and clear the pending file."""
    total_chars = 0
    with open(GROWTH_FILE, "a", encoding="utf-8") as g:
        for entry in entries:
            question = (entry.get("question") or "").strip()
            answer = (entry.get("answer") or "").strip()
            if not question or not answer:
                continue
            block = f"\n\nQUESTION: {question}\nANSWER: {answer}"
            g.write(block)
            total_chars += len(block)
    if entries:
        open(COLLECTED_FILE, "w", encoding="utf-8").close()
    print(f"[Retrain] Absorbed {len(entries)} conversations ({total_chars} chars)")
    return total_chars


def build_corpus():
    """Rebuild sample.txt = base.txt (pristine) + corpus_growth.txt (learned)."""
    base_text = ""
    if os.path.exists(BASE_TEXT_FILE):
        with open(BASE_TEXT_FILE, "r", encoding="utf-8") as f:
            base_text = f.read()
    growth_text = ""
    if os.path.exists(GROWTH_FILE):
        with open(GROWTH_FILE, "r", encoding="utf-8") as f:
            growth_text = f.read()
    blended = base_text.rstrip() + "\n\n" + growth_text.strip()
    with open(RAW_FILE, "w", encoding="utf-8") as f:
        f.write(blended)
    print(f"[Retrain] Corpus rebuilt: {len(base_text):,} base + {len(growth_text):,} learned = {len(blended):,} chars")


def request_reload():
    try:
        req = urllib.request.Request(API_URL, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[Retrain] API reload status: {resp.status}")
            print(resp.read().decode("utf-8", errors="ignore"))
    except Exception as err:
        print(f"[Retrain] API reload failed (is SOUL-LLM running?): {err}")


def main():
    parser = argparse.ArgumentParser(description="Retrain SOUL-LLM on captured conversations")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--after", type=int, default=10)
    args = parser.parse_args()

    entries = load_pending_conversations()
    if not entries:
        print(f"[Retrain] No pending conversations. Will auto-run again after {args.after} are captured.")
        return

    print("=" * 60)
    print("   SOUL-LLM Distillation Retrain")
    print(f"   Pending conversations: {len(entries)}")
    print(f"   Epochs: {args.epochs}")
    print("=" * 60)

    absorb_conversations(entries)
    build_corpus()

    # Run the standard data preparation + training pipeline
    sys.path.insert(0, BASE_DIR)
    sys.path.insert(0, os.path.join(BASE_DIR, "data"))
    sys.path.insert(0, os.path.join(BASE_DIR, "training"))

    import prepare_data
    prepare_data.prepare_data()

    import train
    sys.argv = ["retrain.py", "--epochs", str(args.epochs), "--batch-size", str(32)]
    train.main()

    # Load the freshly trained weights into the running API
    request_reload()
    print("[Retrain] Done. SOUL-LLM has learned from the captured conversations.")


if __name__ == "__main__":
    main()