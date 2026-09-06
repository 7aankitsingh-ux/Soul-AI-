"""
Unit Tests for Model Optimization & Backpropagation
"""

import os
import sys
import unittest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SoulConfig
from model.transformer import SoulLLM


class TestTrainingLoop(unittest.TestCase):

    def test_overfit_single_batch(self):
        """
        The classic sanity check: A neural network must easily overfit
        a single tiny batch, proving backpropagation and optimizer work.
        """
        config = SoulConfig(
            vocab_size=20,
            d_model=32,
            n_layer=2,
            n_head=2,
            block_size=8,
            dropout=0.0
        )
        model = SoulLLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        x = torch.randint(0, config.vocab_size, (2, 8))
        y = torch.randint(0, config.vocab_size, (2, 8))

        # Initial loss
        _, initial_loss = model(x, y)
        initial_val = initial_loss.item()

        # Run 25 optimization steps on this single batch
        for _ in range(25):
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

        final_val = loss.item()

        # Loss should have decreased significantly
        self.assertLess(final_val, initial_val)


if __name__ == "__main__":
    unittest.main()
