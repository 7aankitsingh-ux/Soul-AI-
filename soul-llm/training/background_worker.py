"""
SOUL-LLM Background Training Worker (separate process)
======================================================
A long-running worker process that runs REAL PyTorch training in the
background while the chat application stays usable.

  Communication:
    - Control file  logs/training_control.json   {command: RUN|PAUSE|STOP}
                    written by the API (Start/Pause/Resume/Stop).
    - State file    logs/training_state.json     live progress (step/loss/...)
                    written by the worker, read by the API and the UI.
    - Config file   logs/trainer_config.json     {train_after, worker_epochs, ...}
                    written by the API, re-read live each loop.
    - Log file      logs/training.log            timestamped event log.

  Behaviour:
    - Watches data/qwen_teacher.jsonl (the Qwen-teacher dataset).
    - When (total examples - already-trained examples) >= train_after,
      it rebuilds the student corpus, tokenizes with the SAME vocab used at
      inference, and runs `worker_epochs` real training epochs (forward ->
      loss -> backward -> optimizer.step -> zero_grad).
    - Saves checkpoints (weights + optimizer state + epoch/step/loss).
    - Cooperative PAUSE/STOP is checked between every training step/epoch.
    - After a successful run it asks the SOUL-LLM API to reload the new
      weights so chats use the updated student checkpoint.

Usage (as a separate process):
    python training/background_worker.py
"""

import json
import os
import sys
import time
import urllib.request
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    DEVICE,
    CHECKPOINT_DIR,
    QWEN_TEACHER_JSONL,
    TRAINING_LOG,
    TRAINING_STATE,
    TRAINING_CONTROL,
    TRAINER_CONFIG,
    ACTIVE_CHECKPOINT_FILE,
    TRAIN_AFTER,
    WORKER_EPOCHS,
    WORKER_SAVE_EVERY_STEPS,
    WORKER_POLL_SECONDS,
    DISTILL_LR,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    FF_DIM,
    MAX_SEQ_LEN,
    DROPOUT,
    BATCH_SIZE,
)
from data.build_dataset import build_dataset, load_examples
from model import SoulTransformer
from tokenizer import SimpleTokenizer
from training.trainer import Trainer

API_URL = "http://localhost:8000"
RELOAD_URL = API_URL + "/api/reload"


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + str(msg)
    os.makedirs(os.path.dirname(TRAINING_LOG), exist_ok=True)
    with open(TRAINING_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except Exception:
        pass


def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_control():
    return read_json(TRAINING_CONTROL, {"command": "RUN", "at": time.time()})


def load_config():
    cfg = read_json(TRAINER_CONFIG, {})
    return {
        "train_after": int(cfg.get("train_after", TRAIN_AFTER)),
        "worker_epochs": int(cfg.get("worker_epochs", WORKER_EPOCHS)),
        "batch_size": int(cfg.get("batch_size", BATCH_SIZE)),
        "lr": float(cfg.get("lr", DISTILL_LR)),
        "save_every_steps": int(cfg.get("save_every_steps", WORKER_SAVE_EVERY_STEPS)),
    }


def load_state():
    state = read_json(TRAINING_STATE, {})
    state.setdefault("processed", 0)
    state.setdefault("dataset_size", 0)
    return state


def update_state(**fields):
    state = load_state()
    state.update(fields)
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["device"] = DEVICE
    write_json(TRAINING_STATE, state)
    return state


def request_reload():
    try:
        req = urllib.request.Request(RELOAD_URL, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            log("[Worker] API reload -> " + str(body)[:160])
            return True
    except Exception as e:
        log(f"[Worker] API reload failed (SOUL-LLM API running?): {e}")
        return False


def load_tokenizer():
    tok = SimpleTokenizer()
    tok.load(os.path.join(BASE_DIR, "tokenizer", "vocab.json"))
    return tok


def run_training_run(cfg):
    """Build the dataset, then run real training. Returns summary dict."""
    info = build_dataset()
    vocab_size = info["vocab_size"]
    corpus_hash = info["corpus_hash"]

    model = SoulTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    trainer = Trainer(model, device=DEVICE, lr=cfg["lr"], log_fn=log)

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_model.pt")
    if os.path.exists(latest_ckpt):
        ckpt = Trainer.load_checkpoint(latest_ckpt, DEVICE)
        trainer.resume(ckpt, corpus_hash, vocab_size)

    train_data = torch.load(os.path.join(BASE_DIR, "data", "train.pt"), weights_only=True)
    val_data = torch.load(os.path.join(BASE_DIR, "data", "val.pt"), weights_only=True)
    train_loader, val_loader = Trainer.make_data_loaders(train_data, val_data, cfg["batch_size"])

    log(f"[Worker] Training on {info['dataset_size']} teacher examples | "
        f"corpus {info['corpus_chars']} chars | vocab {vocab_size} | "
        f"sequences {info['sequences']} | device {DEVICE}")
    update_state(
        status="TRAINING", dataset_size=info["dataset_size"],
        pending=max(info["dataset_size"] - load_state().get("processed", 0), 0),
        corpus_chars=info["corpus_chars"], vocab_size=vocab_size,
        sequences=info["sequences"], message=f"Training on {info['dataset_size']} examples"
    )

    def control_check():
        cmd = load_control().get("command")
        if cmd == "STOP":
            update_state(status="STOPPED", message="Stop requested")
        return cmd in ("STOP",)

    def pause_check():
        return load_control().get("command") == "PAUSE"

    last_state_write = 0.0

    def on_step(epoch, step, loss):
        nonlocal last_state_write
        now = time.time()
        if now - last_state_write >= 1.0:
            update_state(status="TRAINING", epoch=epoch, step=step, loss=round(loss, 4))
            last_state_write = now

    def on_epoch(info_epoch):
        update_state(status="TRAINING", epoch=info_epoch["epoch"],
                     step=info_epoch["step"], loss=round(info_epoch["train_loss"], 4),
                     val_loss=round(info_epoch["val_loss"], 4))

    summary = trainer.run(
        train_loader=train_loader, val_loader=val_loader,
        corpus_hash=corpus_hash, vocab_size=vocab_size,
        num_epochs=cfg["worker_epochs"],
        save_dir=CHECKPOINT_DIR,
        save_every_steps=cfg["save_every_steps"],
        should_pause=pause_check, should_stop=control_check,
        on_step=on_step, on_epoch=on_epoch,
    )

    update_state(
        status="STOPPED" if load_control().get("command") == "STOP" else "RUNNING",
        epoch=summary["end_epoch"], step=summary["steps"],
        loss=None, val_loss=round(summary["best_val_loss"], 4),
        message="Checkpoint saved",
    )
    return summary


def main():
    update_state(status="RUNNING", pid=os.getpid(),
                 message="Worker started", error=None)
    log(f"[Worker] Started (pid={os.getpid()}, device={DEVICE})")

    while True:
        try:
            control = load_control()
            cmd = control.get("command", "RUN")

            if cmd == "STOP":
                log("[Worker] STOP command received — exiting.")
                update_state(status="STOPPED", message="Stopped",
                             started_at=None)
                break

            if cmd == "PAUSE":
                update_state(status="PAUSED", message="Paused")
                time.sleep(WORKER_POLL_SECONDS)
                continue

            # RUN / RESUME
            if cmd == "RESUME":
                write_json(TRAINING_CONTROL, {"command": "RUN", "at": time.time()})

            state = load_state()
            dataset_size = len(load_examples())
            processed = min(state.get("processed", 0), dataset_size)
            pending = dataset_size - processed

            update_state(status="RUNNING", dataset_size=dataset_size,
                         processed=processed, pending=pending,
                         message=f"Watching for new examples ({pending} pending)")

            cfg = load_config()
            if pending >= 1 and pending >= cfg["train_after"]:
                run_training_run(cfg)
                state = load_state()
                update_state(processed=dataset_size, pending=0,
                             message="Training run finished; weights active after reload")
                request_reload()
            else:
                time.sleep(WORKER_POLL_SECONDS)

        except SystemExit:
            raise
        except Exception as e:
            import traceback
            log("[Worker] ERROR: " + traceback.format_exc())
            update_state(status="ERROR", message=f"Error: {e}")
            time.sleep(WORKER_POLL_SECONDS * 2)


if __name__ == "__main__":
    main()