# 🧠 SOUL-LLM

> **A small educational language model trained entirely from scratch using PyTorch.**
> No pretrained models. No GPT. No Llama. No Qwen. Pure learning.

---

## ⚠️ Disclaimer

This is a **small educational project** built to understand how LLMs work internally.
It is **NOT** a production-quality language model. The output will be basic and
repetitive — that's expected for a tiny model trained on minimal data.

The value is in **understanding the architecture**, not in the output quality.

---

## 🏗️ Architecture

SOUL-LLM is a **decoder-only Transformer** with:

| Component | Details |
|---|---|
| Attention | Multi-Head Causal Self-Attention |
| Normalization | Pre-LayerNorm (more stable training) |
| Feed-Forward | 2-layer MLP with GELU activation |
| Embeddings | Learned token + positional embeddings |
| Weight Tying | Shared embedding ↔ LM head weights |
| Tokenizer | Character-level (custom, no external libs) |
| Parameters | ~1M (intentionally small) |

---

## 📁 Project Structure

```
soul-llm/
├── config.py                # All hyperparameters
├── requirements.txt         # Dependencies
├── model/
│   ├── attention.py         # Multi-Head Causal Self-Attention
│   └── transformer.py       # Full Transformer model
├── tokenizer/
│   └── tokenizer.py         # Character-level tokenizer
├── data/
│   ├── raw/sample.txt       # Training text corpus
│   └── prepare_data.py      # Data preparation pipeline
├── training/
│   └── train.py             # Training loop with checkpointing
├── inference/
│   └── generate.py          # Text generation with sampling
├── api/
│   └── main.py              # FastAPI REST API
├── frontend/
│   └── index.html           # Web chat interface
├── tests/
│   └── test_all.py          # Test suite
└── checkpoints/             # Saved model weights
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Training Data
```bash
python data/prepare_data.py
```

### 3. Run Tests
```bash
python tests/test_all.py
```

### 4. Train the Model
```bash
python training/train.py --epochs 50 --batch-size 32
```

### 5. Generate Text (CLI)
```bash
python inference/generate.py
```

### 6. Start Web Interface
```bash
cd soul-llm
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Then open http://localhost:8000

---

## 🎓 For BCA Students

This project demonstrates:
1. **Tokenization** — How text becomes numbers
2. **Embeddings** — How numbers become meaningful vectors
3. **Self-Attention** — How the model relates words to each other
4. **Transformer Blocks** — How layers transform representations
5. **Training** — How the model learns from data
6. **Generation** — How the model produces text

Each file is heavily commented to explain every step.

---

## 📊 Technology Stack

- **Python 3.11+**
- **PyTorch** — Deep learning framework
- **NumPy** — Numerical computing
- **FastAPI** — REST API
- **Uvicorn** — ASGI server

**Zero pretrained models. Zero external LLM APIs.**

---

*Built with ❤️ for learning — by a human, for humans.*
