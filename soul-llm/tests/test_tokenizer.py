"""
Unit Tests for SimpleTokenizer
"""

import os
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.tokenizer import SimpleTokenizer


class TestSimpleTokenizer(unittest.TestCase):

    def setUp(self):
        self.text = "Hello, World! Python 123."
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.fit(self.text)

    def test_vocab_size(self):
        # 4 special tokens + unique characters
        expected_unique_chars = len(set(self.text))
        self.assertEqual(self.tokenizer.vocab_size, expected_unique_chars + 4)

    def test_encode_decode_roundtrip(self):
        encoded = self.tokenizer.encode(self.text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, self.text)

    def test_unknown_token(self):
        # Character not in vocabulary
        unseen_text = "Hello, World! 🚀"
        encoded = self.tokenizer.encode(unseen_text)
        self.assertIn(self.tokenizer.unk_id, encoded)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_file = os.path.join(tmpdir, "vocab.json")
            self.tokenizer.save(vocab_file)
            self.assertTrue(os.path.exists(vocab_file))

            loaded_tokenizer = SimpleTokenizer(vocab_file)
            self.assertEqual(loaded_tokenizer.vocab_size, self.tokenizer.vocab_size)
            self.assertEqual(loaded_tokenizer.decode(self.tokenizer.encode("Hello")), "Hello")


if __name__ == "__main__":
    unittest.main()
