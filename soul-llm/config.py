"""
SOUL-LLM Configuration
======================
All hyperparameters and settings for the SOUL-LLM project.
This is a small educational decoder-only Transformer — no pretrained models used.

SOUL-LLM is the STUDENT. Qwen (via Ollama) is only the TEACHER that
generates high-quality Q&A which we distill into SOUL-LLM's own weights.
"""

import os
import torch

# ─── Device Configuration ───────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Model Architecture ─────────────────────────────────────────────
VOCAB_SIZE = 512          # Small vocabulary for educational purposes
EMBED_DIM = 256           # Embedding dimension
NUM_HEADS = 4             # Number of attention heads
NUM_LAYERS = 4            # Number of transformer blocks
FF_DIM = 512              # Feed-forward hidden dimension
MAX_SEQ_LEN = 128         # Maximum sequence length
DROPOUT = 0.1             # Dropout rate

# ─── Training ────────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
GRAD_CLIP = 1.0

# ─── Data ────────────────────────────────────────────────────────────
DATA_DIR = "data"
RAW_DATA_PATH = "data/raw/sample.txt"
TRAIN_DATA_PATH = "data/train.pt"
VAL_DATA_PATH = "data/val.pt"

# ─── Checkpoints ─────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 10           # Save checkpoint every N epochs

# ─── Generation ──────────────────────────────────────────────────────
MAX_GEN_LEN = 200         # Maximum generation length
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.9               # Nucleus sampling cutoff (added to tame a tiny vocab)

# ─── Distillation Trainer (Qwen Teacher -> SOUL-LLM Student) ─────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Teacher dataset written by the main SOUL AI agent (agent/core.js):
#   {"prompt": ..., "response": ..., "timestamp": ..., "model": ...}
QWEN_TEACHER_JSONL = os.path.join(DATA_DIR, "qwen_teacher.jsonl")

# Rebuilt student corpus: pristine base.txt + formatted Q&A blocks
CORPUS_FILE = os.path.join(DATA_DIR, "student_corpus.txt")

TRAINING_LOG = os.path.join(LOGS_DIR, "training.log")
TRAINING_STATE = os.path.join(LOGS_DIR, "training_state.json")
TRAINING_CONTROL = os.path.join(LOGS_DIR, "training_control.json")
TRAINER_CONFIG = os.path.join(LOGS_DIR, "trainer_config.json")
ACTIVE_CHECKPOINT_FILE = os.path.join(LOGS_DIR, "active_checkpoint.txt")

# Worker behaviour. Runtime overrides can be written to TRAINER_CONFIG.
TRAIN_AFTER = 5           # train after this many pending (untrained) examples
WORKER_EPOCHS = 10        # epochs per training run
WORKER_SAVE_EVERY_STEPS = 100
WORKER_POLL_SECONDS = 2.0 # how often the worker re-reads its control file

# Optimizer for distillation runs
DISTILL_LR = 5e-4
WEIGHT_DECAY = 0.01
