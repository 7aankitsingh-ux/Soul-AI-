"""
Unit Tests for CausalSelfAttention
"""

import os
import sys
import unittest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SoulConfig
from model.attention import CausalSelfAttention


class TestCausalSelfAttention(unittest.TestCase):

    def setUp(self):
        self.config = SoulConfig(
            d_model=64,
            n_head=2,
            block_size=16,
            dropout=0.0
        )
        self.attn = CausalSelfAttention(self.config)

    def test_output_shape(self):
        B, T, C = 2, 8, self.config.d_model
        x = torch.randn(B, T, C)
        out = self.attn(x)
        self.assertEqual(out.shape, (B, T, C))

    def test_causality_masking(self):
        """
        Verify that changing future inputs does NOT affect past outputs.
        This is the defining property of causal autoregressive attention!
        """
        B, T, C = 1, 4, self.config.d_model
        x1 = torch.randn(B, T, C)
        x2 = x1.clone()

        # Modify only the last token at position index 3
        x2[:, 3, :] = torch.randn(1, C)

        out1 = self.attn(x1)
        out2 = self.attn(x2)

        # Positions 0, 1, and 2 should be exactly identical between out1 and out2!
        diff_past = (out1[:, :3, :] - out2[:, :3, :]).abs().max().item()
        self.assertAlmostEqual(diff_past, 0.0, places=5)

        # Position 3 should differ
        diff_future = (out1[:, 3, :] - out2[:, 3, :]).abs().max().item()
        self.assertGreater(diff_future, 1e-4)


if __name__ == "__main__":
    unittest.main()
