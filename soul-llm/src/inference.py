"""
SOUL-LLM Independent Inference Engine
=====================================
Strictly loads ONLY the local SOUL-LLM model checkpoint and tokenizer.

FINAL INDEPENDENCE GUARANTEES:
- Zero imports of Qwen, HuggingFace, transformers, or external model code.
- Zero network requests or internet connectivity required.
- Zero Qwen model weights or adapters loaded.
- 100% self-contained PyTorch decoder-only Transformer.
"""

import os
import sys
import torch

# Ensure local package imports work cleanly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import SoulTransformer
from tokenizer.tokenizer import SimpleTokenizer

_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_CACHED_DEVICE = None


def load_independent_soul_llm(
    checkpoint_path: str = None,
    vocab_path: str = None,
    device: str = None
):
    """
    Loads SOUL-LLM student model and character tokenizer.
    No Qwen dependencies, no external APIs, fully offline.
    """
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _CACHED_DEVICE = device

    # 1. Resolve paths
    if vocab_path is None:
        vocab_path = os.path.join(ROOT_DIR, "tokenizer", "vocab.json")
    if checkpoint_path is None:
        candidate_paths = [
            os.path.join(ROOT_DIR, "model", "soul-llm", "best_model.pt"),
            os.path.join(ROOT_DIR, "checkpoints", "best_model.pt")
        ]
        checkpoint_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"[SOUL-LLM] Tokenizer vocab not found: {vocab_path}")

    # 2. Load Tokenizer
    tokenizer = SimpleTokenizer(vocab_path)
    _CACHED_TOKENIZER = tokenizer

    # 3. Load Checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[SOUL-LLM] Model checkpoint not found at: {checkpoint_path}\n"
            "Train or distill the model first by running: python distill.py"
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Determine vocabulary size from model weights
    if "lm_head.weight" in state_dict:
        vocab_size = state_dict["lm_head.weight"].shape[0]
    elif "token_embedding.weight" in state_dict:
        vocab_size = state_dict["token_embedding.weight"].shape[0]
    else:
        vocab_size = tokenizer.get_vocab_size()

    # Instantiate SOUL-LLM architecture
    model = SoulTransformer(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        max_seq_len=128,
        dropout=0.0
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    epoch_info = ckpt.get("epoch", "distilled")
    loss_info = ckpt.get("loss", ckpt.get("val_loss", 0.0))
    print(f"[SOUL-LLM] Successfully loaded independent model from {checkpoint_path}")
    print(f"[SOUL-LLM] Status: Offline • Self-contained • Epoch: {epoch_info} • Loss: {loss_info:.4f}")

    _CACHED_MODEL = model
    return model, tokenizer


def generate(
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    deterministic: bool = False,
    checkpoint_path: str = None
) -> str:
    """
    Generate text using only the local SOUL-LLM student model.
    Zero Qwen involvement.
    """
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE

    if _CACHED_MODEL is None or _CACHED_TOKENIZER is None:
        load_independent_soul_llm(checkpoint_path=checkpoint_path)

    model = _CACHED_MODEL
    tokenizer = _CACHED_TOKENIZER
    device = _CACHED_DEVICE

    # Encode prompt into integer token IDs
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        input_ids = [tokenizer.bos_id]

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate token sequence
    out_ids = model.generate(
        idx=x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        deterministic=deterministic
    )

    # Decode back into string
    generated_ids = out_ids[0].tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


if __name__ == "__main__":
    test_prompt = "Artificial intelligence is"
    print(f"Prompt: {test_prompt}")
    output = generate(test_prompt, max_new_tokens=80)
    print(f"Response: {output}")
