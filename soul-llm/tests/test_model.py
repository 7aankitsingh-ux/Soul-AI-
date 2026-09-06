"""
Unit Tests for SoulLLM Transformer
"""

import os
import sys
import unittest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SoulConfig
from model.transformer import SoulLLM


class TestSoulLLM(unittest.TestCase):

    def setUp(self):
        self.config = SoulConfig(
            vocab_size=50,
            d_model=64,
            n_layer=2,
            n_head=2,
            block_size=16,
            dropout=0.0
        )
        self.model = SoulLLM(self.config)

    def test_forward_pass_without_targets(self):
        B, T = 2, 8
        idx = torch.randint(0, self.config.vocab_size, (B, T))
        logits, loss = self.model(idx)
        self.assertEqual(logits.shape, (B, T, self.config.vocab_size))
        self.assertIsNone(loss)

    def test_forward_pass_with_targets(self):
        B, T = 2, 8
        idx = torch.randint(0, self.config.vocab_size, (B, T))
        targets = torch.randint(0, self.config.vocab_size, (B, T))
        logits, loss = self.model(idx, targets)
        self.assertEqual(logits.shape, (B, T, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

    def test_parameter_count(self):
        num_params = self.model.get_num_params()
        self.assertGreater(num_params, 1000)

    def test_generation_shape(self):
        idx = torch.tensor([[1, 2, 3]])
        max_new = 5
        generated = self.model.generate(idx, max_new_tokens=max_new, deterministic=True)
        self.assertEqual(generated.shape, (1, 3 + max_new))


if __name__ == "__main__":
    unittest.main()
