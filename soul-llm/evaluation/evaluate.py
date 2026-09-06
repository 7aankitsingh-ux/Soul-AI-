"""
SOUL-LLM Evaluation
====================
Value beyond loss: evaluates the STUDENT with fixed prompts and shows the
ACTUAL generated text. Real cross-entropy loss is computed on the validation
set; generated responses are produced by the student's own weights.

Usage:
    python evaluation/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    DEVICE, DROPOUT, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM,
    MAX_SEQ_LEN, CHECKPOINT_DIR, TRAINING_STATE,
)
from model import SoulTransformer
from tokenizer import SimpleTokenizer


@torch.no_grad()
def compute_val_loss(model, tokenizer, num_batches=8, batch_size=8):
    """Real validation loss over data/val.pt (capped batches)."""
    val_path = os.path.join(BASE_DIR, "data", "val.pt")
    if not os.path.exists(val_path):
        return None
    val_data = torch.load(val_path, weights_only=True)
    if len(val_data) < 2:
        return None
    x, y = val_data[:, :-1], val_data[:, 1:]
    total, n = 0.0, 0
    for i in range(0, min(len(x), num_batches * batch_size), batch_size):
        bx, by = x[i:i + batch_size].to(DEVICE), y[i:i + batch_size].to(DEVICE)
        logits, loss = model(bx, by)
        total += loss.item()
        n += 1
    return round(total / max(n, 1), 4)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=80, temperature=0.6, top_k=20, top_p=0.9):
    """Greedy-ish sampling decoder shared with inference (same tokenizer)."""
    model.eval()
    input_ids = tokenizer.encode(prompt)[:-1]  # drop EOS, keep BOS
    if len(input_ids) == 0:
        input_ids = [tokenizer.char_to_id[tokenizer.bos_token]]
    input_ids = input_ids[-MAX_SEQ_LEN:]
    tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    generated = tensor.tolist()[0]

    for _ in range(max_new_tokens):
        logits, _ = model(tensor[:, -MAX_SEQ_LEN:])
        next_logits = logits[:, -1, :] / temperature

        if top_k > 0:
            k = min(top_k, next_logits.size(-1))
            thr = torch.topk(next_logits, k).values[:, -1].unsqueeze(-1)
            next_logits = torch.where(next_logits < thr,
                                      torch.full_like(next_logits, float("-inf")), next_logits)
        if top_p < 1.0:
            sorted_logits, idx = torch.sort(next_logits, descending=True)
            cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
            next_logits = torch.zeros_like(next_logits).scatter(-1, idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == tokenizer.char_to_id.get(tokenizer.eos_token, -1):
            break
        generated.append(next_token.item())
        tensor = torch.cat([tensor, next_token], dim=1)

    return tokenizer.decode(generated)


def default_prompts():
    return [
        "hello",
        "what is your name?",
        "how are you?",
        "can you help me?",
    ]


def evaluate(model=None, tokenizer=None, checkpoint_path=None,
             max_new_tokens=80, prompts=None,
             device=DEVICE):
    """Evaluate a SOUL-LLM checkpoint. Returns a dict of results.
    Supplies its own fresh model when (model, tokenizer) are not given."""
    own_model = model is None
    if own_model:
        tok = SimpleTokenizer()
        tok.load(os.path.join(BASE_DIR, "tokenizer", "vocab.json"))
        model = SoulTransformer(
            vocab_size=tok.get_vocab_size(), embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS, num_layers=NUM_LAYERS, ff_dim=FF_DIM,
            max_seq_len=MAX_SEQ_LEN, dropout=DROPOUT,
        ).to(device)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        tokenizer = tok

    val_loss = compute_val_loss(model, tokenizer)
    prompts = prompts or default_prompts()
    samples = []
    for p in prompts:
        # Train format is "User: ...\nAssistant: ..." — prompt the same way.
        formatted = f"User: {p}\nAssistant:"
        out = generate(model, tokenizer, formatted, max_new_tokens=max_new_tokens)
        response = out.split("Assistant:", 1)[-1].strip() if "Assistant:" in out else out
        samples.append({"prompt": p, "prompt_formatted": formatted,
                        "response": response, "generated_full": out})

    result = {
        "device": device,
        "val_loss": val_loss,
        "checkpoint": checkpoint_path or "active",
        "samples": samples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOUL-LLM")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()

    if args.checkpoint is None:
        active = os.path.join(BASE_DIR, "logs", "active_checkpoint.txt")
        if os.path.exists(active):
            with open(active, encoding="utf-8") as f:
                args.checkpoint = f.read().strip()
    if args.checkpoint is None or not os.path.exists(args.checkpoint):
        args.checkpoint = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    print("=" * 62)
    print(f"   SOUL-LLM Evaluation   device={DEVICE}")
    print(f"   Checkpoint: {args.checkpoint}")
    print("=" * 62)

    result = evaluate(checkpoint_path=args.checkpoint)

    print(f"Validation loss: {result['val_loss']}")
    print("-" * 62)
    for s in result["samples"]:
        print(f"PROMPT    : {s['prompt']}")
        print(f"RESPONSE  : {s['response']}")
        print("-" * 62)

    # Persist the last evaluation for the UI (logs/training_state.json)
    try:
        state = {}
        with open(TRAINING_STATE, "r", encoding="utf-8") as f:
            state = json.load(f)
        state["last_eval"] = result
        tmp = TRAINING_STATE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, TRAINING_STATE)
    except Exception:
        pass


if __name__ == "__main__":
    main()