"""
SOUL-LLM Text Generation (Inference)
======================================
Generates text from a trained SOUL-LLM model.
Supports temperature sampling and top-k filtering.
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import DEVICE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, MAX_SEQ_LEN, DROPOUT
from model import SoulTransformer
from tokenizer import SimpleTokenizer


def load_model_and_tokenizer(checkpoint_path: str = None):
    """Load a trained SOUL-LLM model and its tokenizer."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    # Load tokenizer
    tok_path = os.path.join(base_dir, "tokenizer", "vocab.json")
    tokenizer = SimpleTokenizer()
    tokenizer.load(tok_path)

    # Create model
    model = SoulTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Generate] Loaded model from {checkpoint_path}")
        print(f"[Generate] Trained for {checkpoint['epoch']} epochs, loss={checkpoint['loss']:.4f}")
    else:
        print(f"[WARNING] No checkpoint found at {checkpoint_path}. Using random weights!")

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str = "", max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 40) -> str:
    """
    Generate text using the trained model.
    
    Args:
        model: Trained SoulTransformer
        tokenizer: SimpleTokenizer
        prompt: Starting text (can be empty)
        max_new_tokens: Number of new tokens to generate
        temperature: Controls randomness (higher = more random)
        top_k: Only sample from top-k most likely tokens
    
    Returns:
        Generated text string
    """
    model.eval()

    # Encode the prompt
    if prompt:
        input_ids = tokenizer.encode(prompt)
        # Remove the EOS token from encoded prompt
        input_ids = input_ids[:-1]
    else:
        # Start with BOS token
        input_ids = [tokenizer.char_to_id[tokenizer.bos_token]]

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    generated = input_ids.tolist()[0]

    for _ in range(max_new_tokens):
        # Truncate to max sequence length
        context = input_ids[:, -MAX_SEQ_LEN:]

        # Get model predictions
        logits, _ = model(context)

        # Take logits for the last position
        next_logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            top_k_val = min(top_k, next_logits.size(-1))
            values, _ = torch.topk(next_logits, top_k_val)
            min_val = values[:, -1].unsqueeze(-1)
            next_logits = torch.where(
                next_logits < min_val,
                torch.full_like(next_logits, float("-inf")),
                next_logits
            )

        # Sample from the distribution
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Check for EOS
        if next_token.item() == tokenizer.char_to_id.get(tokenizer.eos_token, -1):
            break

        # Append to sequence
        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode
    return tokenizer.decode(generated)


def main():
    """Interactive generation mode."""
    print("=" * 50)
    print("   SOUL-LLM Text Generator")
    print("=" * 50)

    model, tokenizer = load_model_and_tokenizer()

    print("\nType a prompt and press Enter to generate text.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break

            text = generate(model, tokenizer, prompt=prompt, max_new_tokens=200)
            print(f"\nSOUL-LLM: {text}\n")

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
