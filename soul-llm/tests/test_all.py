"""
SOUL-LLM Test Suite
====================
Tests to verify all components work correctly.
Run: python tests/test_all.py
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def test_tokenizer():
    """Test the SimpleTokenizer."""
    print("\n[TEST] Tokenizer...")
    from tokenizer import SimpleTokenizer

    tok = SimpleTokenizer()
    tok.fit("hello world! 123")

    # Test encode/decode roundtrip
    text = "hello"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text, f"Roundtrip failed: '{text}' → {encoded} → '{decoded}'"

    # Test vocab size
    assert tok.get_vocab_size() > 4, "Vocab too small (should include special tokens + chars)"

    # Test unknown characters
    encoded_unk = tok.encode("ñ")  # Not in training text
    assert 1 in encoded_unk, "Unknown char should map to UNK token (id=1)"

    print("   ✓ Tokenizer: PASSED")
    return True


def test_attention():
    """Test CausalSelfAttention."""
    print("\n[TEST] Causal Self-Attention...")
    from model.attention import CausalSelfAttention

    attn = CausalSelfAttention(embed_dim=64, num_heads=4, max_seq_len=32)

    # Test forward pass shape
    x = torch.randn(2, 16, 64)  # (batch=2, seq_len=16, embed_dim=64)
    out = attn(x)
    assert out.shape == (2, 16, 64), f"Output shape wrong: {out.shape}"

    # Test causal mask exists
    assert hasattr(attn, 'causal_mask'), "Causal mask not registered"

    print("   ✓ Attention: PASSED")
    return True


def test_transformer():
    """Test the full SoulTransformer."""
    print("\n[TEST] SoulTransformer...")
    from model import SoulTransformer

    model = SoulTransformer(
        vocab_size=100,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        max_seq_len=32,
        dropout=0.1,
    )

    # Test forward pass
    input_ids = torch.randint(0, 100, (2, 16))  # (batch=2, seq_len=16)
    targets = torch.randint(0, 100, (2, 16))

    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 16, 100), f"Logits shape wrong: {logits.shape}"
    assert loss is not None, "Loss should not be None when targets provided"
    assert loss.item() > 0, "Loss should be positive"

    # Test without targets
    logits2, loss2 = model(input_ids)
    assert loss2 is None, "Loss should be None without targets"

    # Test parameter count
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, "Model has no parameters"
    print(f"   Model has {n_params:,} parameters")

    print("   ✓ Transformer: PASSED")
    return True


def test_generation():
    """Test text generation (with random weights)."""
    print("\n[TEST] Generation (random weights)...")
    from model import SoulTransformer
    from tokenizer import SimpleTokenizer
    from inference.generate import generate

    # Build a small test model
    tok = SimpleTokenizer()
    tok.fit("abcdefghijklmnopqrstuvwxyz .!?")

    model = SoulTransformer(
        vocab_size=tok.get_vocab_size(),
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        max_seq_len=32,
    )

    # Generate some text
    text = generate(model, tok, prompt="hello", max_new_tokens=20,
                    temperature=1.0, top_k=0)

    assert isinstance(text, str), "Output should be a string"
    assert len(text) > 0, "Output should not be empty"
    print(f"   Generated: '{text[:50]}...'")

    print("   ✓ Generation: PASSED")
    return True


def test_data_shapes():
    """Test that data preparation produces correct shapes."""
    print("\n[TEST] Data shapes...")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    train_path = os.path.join(data_dir, "train.pt")

    if not os.path.exists(train_path):
        print("   ⚠ Skipped (run data/prepare_data.py first)")
        return True

    train_data = torch.load(train_path, weights_only=True)
    assert len(train_data.shape) == 2, f"Expected 2D tensor, got {train_data.shape}"
    assert train_data.shape[1] > 1, "Sequences too short"
    print(f"   Train data shape: {train_data.shape}")

    print("   ✓ Data shapes: PASSED")
    return True


def main():
    print("=" * 50)
    print("   SOUL-LLM Test Suite")
    print("=" * 50)

    tests = [
        test_tokenizer,
        test_attention,
        test_transformer,
        test_generation,
        test_data_shapes,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"   Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
