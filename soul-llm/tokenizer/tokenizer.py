"""
Simple Character-Level Tokenizer for SOUL-LLM
=============================================
A minimal tokenizer that maps characters to integer IDs.
This is intentionally simple for educational clarity.
No external tokenizer libraries are used.
"""


class SimpleTokenizer:
    """
    Character-level tokenizer.
    
    Special tokens:
        <PAD> = 0
        <UNK> = 1
        <BOS> = 2  (beginning of sequence)
        <EOS> = 3  (end of sequence)
    """

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

        # Reserve special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(self.special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token
        self.vocab_size = len(self.special_tokens)

    def fit(self, text: str):
        """
        Build vocabulary from a text corpus.
        Scans every unique character and assigns it an integer ID.
        """
        unique_chars = sorted(set(text))
        for ch in unique_chars:
            if ch not in self.char_to_id:
                idx = self.vocab_size
                self.char_to_id[ch] = idx
                self.id_to_char[idx] = ch
                self.vocab_size += 1
        print(f"[Tokenizer] Vocabulary built: {self.vocab_size} tokens "
              f"({self.vocab_size - len(self.special_tokens)} characters + "
              f"{len(self.special_tokens)} special tokens)")

    def encode(self, text: str) -> list:
        """Convert a string into a list of integer token IDs."""
        bos_id = self.char_to_id[self.bos_token]
        eos_id = self.char_to_id[self.eos_token]
        unk_id = self.char_to_id[self.unk_token]
        ids = [bos_id]
        for ch in text:
            ids.append(self.char_to_id.get(ch, unk_id))
        ids.append(eos_id)
        return ids

    def decode(self, ids: list) -> str:
        """Convert a list of integer token IDs back into a string."""
        chars = []
        for idx in ids:
            token = self.id_to_char.get(idx, self.unk_token)
            if token in self.special_tokens:
                continue  # Skip special tokens in output
            chars.append(token)
        return "".join(chars)

    def get_vocab_size(self) -> int:
        """Return the total vocabulary size including special tokens."""
        return self.vocab_size

    def save(self, path: str):
        """Save vocabulary mapping to a file."""
        import json
        data = {
            "char_to_id": self.char_to_id,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] Saved to {path}")

    def load(self, path: str):
        """Load vocabulary mapping from a file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.vocab_size = data["vocab_size"]
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        print(f"[Tokenizer] Loaded from {path} ({self.vocab_size} tokens)")
