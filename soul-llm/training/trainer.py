"""
SOUL-LLM Trainer (student distillation training)
================================================
The REAL training loop. Every call performs an actual PyTorch update:

    logits, loss = model(x, y)     # forward pass
    loss.backward()                # backward pass
    optimizer.step()               # weight update
    optimizer.zero_grad()          # reset gradients

No fake progress, no simulated losses. Loss values are real cross-entropy
values produced by the model on real data.

Checkpoints store:
    model_state_dict      (SOUL-LLM's OWN weights — never Qwen's)
    optimizer_state_dict  (AdamW momentum/variance => resumable training)
    epoch / step          (current training position)
    train_loss / val_loss (latest real losses)
    corpus_hash / vocab_size (validate skip/resume compatibility)
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(self, model, device, lr=5e-4, weight_decay=0.01,
                 grad_clip=1.0, log_fn=print):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.grad_clip = grad_clip
        self.log = log_fn

        # Current position in training (used for resume + reporting)
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float("inf")
        self.total_params = sum(p.numel() for p in model.parameters())

    # ─── Checkpoint helpers ─────────────────────────────────────────
    @staticmethod
    def load_checkpoint(path, map_location):
        """Robust load that also works for older non-resumable checkpoints."""
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            return torch.load(path, map_location=map_location, weights_only=False)

    def resume(self, checkpoint, corpus_hash, vocab_size):
        """Try to resume training from a checkpoint. Returns bool.
        Resume stays valid as the corpus grows (weights are the student's own
        and should be retained/rehearsed). A VOCAB change forces a fresh start
        because embedding/lm_head tensors would no longer match."""
        sd = checkpoint.get("model_state_dict")
        if sd is None:
            self.log(f"[Trainer] Checkpoint has no model_state_dict — starting fresh.")
            return False
        if checkpoint.get("vocab_size") != vocab_size:
            self.log("[Trainer] Vocab size differs from checkpoint — starting fresh "
                     "(tokenizer corpus changed).")
            return False
        try:
            self.model.load_state_dict(sd)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            self.log(f"[Trainer] Could not restore optimizer/weights ({e}) — starting fresh.")
            return False
        self.epoch = int(checkpoint.get("epoch", 0))
        self.step = int(checkpoint.get("step", 0))
        self.best_val_loss = float(checkpoint.get("val_loss", checkpoint.get("loss", float("inf"))))
        self.log(f"[Trainer] Resumed at epoch {self.epoch}, step {self.step}, "
                 f"best_val={self.best_val_loss:.4f}")
        return True

    def save(self, path, train_loss, val_loss, is_best=False):
        payload = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": train_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "params": self.total_params,
            "device": self.device,
            "corpus_hash": self._corpus_hash,
            "vocab_size": self._vocab_size,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(payload, path)
        # Sidecar metadata so the API can list checkpoints without loading the
        # full weights/optimizer tensors.
        try:
            meta = {k: v for k, v in payload.items()
                    if k not in ("model_state_dict", "optimizer_state_dict")}
            with open(path + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass
        if is_best and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        self.log(f"[Trainer] Saved checkpoint -> {os.path.basename(path)} "
                 f"(epoch {self.epoch}, step {self.step}, loss {train_loss:.4f})")
        return path

    # ─── Data loaders ──────────────────────────────────────────────
    @staticmethod
    def make_data_loaders(train_data, val_data, batch_size):
        tx, ty = train_data[:, :-1], train_data[:, 1:]
        vx, vy = val_data[:, :-1], val_data[:, 1:]
        train_loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size,
                                  shuffle=True, drop_last=False)
        val_loader = DataLoader(TensorDataset(vx, vy), batch_size=batch_size,
                                shuffle=False, drop_last=False)
        return train_loader, val_loader

    # ─── Training ──────────────────────────────────────────────────
    def train_epoch(self, loader, should_pause=None, should_stop=None,
                    on_step=None):
        """Run one epoch over the training loader. Returns average loss."""
        self.model.train()
        total = 0.0
        num = 0
        for batch_x, batch_y in loader:
            if should_stop and should_stop():
                self.log("[Trainer] Stop requested — finishing epoch early.")
                return total / max(num, 1)
            # Pause between steps so the worker can be halted cooperatively.
            while should_pause and should_pause():
                time.sleep(0.5)
                if should_stop and should_stop():
                    self.log("[Trainer] Stop requested while paused — aborting.")
                    return total / max(num, 1)

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # 1. Forward pass
            logits, loss = self.model(batch_x, batch_y)
            # 2. Reset gradients
            self.optimizer.zero_grad()
            # 3. Backward pass (real gradients)
            loss.backward()
            # 4. Clip then step optimizer (real weight updates)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            self.step += 1
            total += loss.item()
            num += 1

            if on_step:
                on_step(self.epoch, self.step, loss.item())

        return total / max(num, 1)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total = 0.0
        num = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            _, loss = self.model(batch_x, batch_y)
            total += loss.item()
            num += 1
        return total / max(num, 1)

    def run(self, train_loader, val_loader, corpus_hash, vocab_size,
            num_epochs, save_dir, save_every_steps=100,
            should_pause=None, should_stop=None, on_step=None, on_epoch=None):
        """
        Run num_epochs epochs of REAL training. Returns summary dict.
        """
        self._corpus_hash = corpus_hash
        self._vocab_size = vocab_size
        os.makedirs(save_dir, exist_ok=True)

        start_epoch = self.epoch
        history = []
        checkpoints = []
        latest_path = os.path.join(save_dir, "latest_model.pt")
        best_path = os.path.join(save_dir, "best_model.pt")

        for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
            self.epoch = epoch
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, should_pause, should_stop, on_step)
            val_loss = self.validate(val_loader)
            elapsed = time.time() - t0
            self.log(f"[Trainer] epoch {epoch}/{start_epoch + num_epochs} | "
                     f"train={train_loss:.4f} val={val_loss:.4f} | {elapsed:.1f}s")

            history.append({
                "epoch": epoch,
                "step": self.step,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
            })
            if on_epoch:
                on_epoch({"epoch": epoch, "step": self.step,
                          "train_loss": train_loss, "val_loss": val_loss})

            # Best (by validation loss) + latest checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoints.append(self.save(best_path, train_loss, val_loss, is_best=True))
            checkpoints.append(self.save(latest_path, train_loss, val_loss))

            # Periodic checkpoint with an unambiguous name
            if save_every_steps > 0 and self.step % save_every_steps < 1:
                ckpt = os.path.join(save_dir, f"checkpoint_epoch{self.epoch}_step{self.step}.pt")
                checkpoints.append(self.save(ckpt, train_loss, val_loss))

            if should_stop and should_stop():
                self.log("[Trainer] STOPPED (checkpoint already saved for this epoch).")
                break

        summary = {
            "start_epoch": start_epoch,
            "end_epoch": self.epoch,
            "steps": self.step,
            "best_val_loss": self.best_val_loss,
            "history": history,
            "checkpoints": checkpoints[-3:] if checkpoints else [],
        }
        return summary