"""
SOUL-LLM FastAPI Backend
========================
Exposes the trained SOUL-LLM student model AND the background distillation
trainer (Qwen teacher -> SOUL-LLM student) controls.

Teacher notes:
  - SOUL-LLM owns its weights; it NEVER calls Qwen during inference.
  - Training is done by training/background_worker.py (a separate process):
      /api/training/start|pause|resume|stop control it via logs/training_control.json
      /api/training/status   reads live progress from logs/training_state.json
  - The active served checkpoint can be chosen with /api/model/select.
"""

import glob
import os
import subprocess
import sys
import json
import time
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    DEVICE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, MAX_SEQ_LEN,
    DROPOUT, LOGS_DIR, TRAINING_CONTROL, TRAINING_STATE, TRAINER_CONFIG,
    ACTIVE_CHECKPOINT_FILE, TRAIN_AFTER, WORKER_EPOCHS, BATCH_SIZE,
    WORKER_SAVE_EVERY_STEPS,
    DISTILL_LR, TEMPERATURE, TOP_K, TOP_P,
)
from model import SoulTransformer
from tokenizer import SimpleTokenizer

ARCHITECTURE_NOTE = (
    "SOUL-LLM is a from-scratch educational decoder-only Transformer "
    "(~2M params, character-level vocab). It is the STUDENT. Qwen 2.5 7B is "
    "only the TEACHER that generates training examples. SOUL-LLM inference "
    "never calls Qwen. Its language quality is limited by its small size and "
    "training data; it improves as more teacher examples are distilled in."
)

app = FastAPI(title="SOUL-LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
TOKENIZER = None
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
VOCAB_PATH = os.path.join(BASE_DIR, "tokenizer", "vocab.json")


def ensure_logs():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─── Small JSON helpers (mirrors of the worker's, no shared code needed) ──
def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    ensure_logs()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def active_checkpoint_path():
    """Path the API should serve + reload. Defaults to best_model.pt."""
    if os.path.exists(ACTIVE_CHECKPOINT_FILE):
        try:
            with open(ACTIVE_CHECKPOINT_FILE, encoding="utf-8") as f:
                p = f.read().strip()
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    best = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if os.path.exists(best):
        return best
    final = os.path.join(CHECKPOINT_DIR, "final_model.pt")
    return final if os.path.exists(final) else None


def list_checkpoints():
    out = []
    for path in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt"))):
        meta = {}  # name from sidecar if present
        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        else:
            try:
                ck = torch.load(path, map_location="cpu", weights_only=True)
                meta = {k: ck[k] for k in ("epoch", "step", "loss", "val_loss",
                                           "params", "corpus_hash", "vocab_size",
                                           "saved_at") if k in ck}
            except Exception:
                pass
        out.append({
            "path": path,
            "name": os.path.basename(path),
            "size_mb": round(os.path.getsize(path) / 1e6, 2),
            **{k: v for k, v in meta.items()},
        })
    return out


def worker_alive(pid=None):
    if pid is None:
        pid = read_json(TRAINING_STATE, {}).get("pid")
    if not pid:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        return str(pid) in out and "No tasks" not in out
    except Exception:
        return False


def spawn_worker():
    worker_py = os.path.join(BASE_DIR, "training", "background_worker.py")
    ensure_logs()
    out_log = open(os.path.join(LOGS_DIR, "worker.out.log"), "a", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [sys.executable, worker_py],
        cwd=BASE_DIR, stdout=out_log, stderr=out_log, creationflags=flags,
    )
    return proc.pid


def set_control(command):
    write_json(TRAINING_CONTROL, {"command": command, "at": time.time()})


# ─── Model loading ───────────────────────────────────────────────────────
def load_model():
    global MODEL, TOKENIZER
    if MODEL is not None and TOKENIZER is not None:
        return

    TOKENIZER = SimpleTokenizer()
    TOKENIZER.load(VOCAB_PATH)

    MODEL = SoulTransformer(
        vocab_size=TOKENIZER.get_vocab_size(),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    ckpt_path = active_checkpoint_path()
    if ckpt_path:
        try:
            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            MODEL.load_state_dict(checkpoint["model_state_dict"])
            loss = checkpoint.get("loss", checkpoint.get("val_loss", "?"))
            print(f"[SOUL-LLM] Loaded checkpoint {os.path.basename(ckpt_path)} "
                  f"(epoch {checkpoint.get('epoch', '?')}, loss {loss})")
        except Exception as e:
            # Vocab drift (corpus/tokenizer rebuild) makes an old checkpoint
            # structurally incompatible. Never crash: serve random init and tell
            # the user to retrain so checkpoints match the current vocabulary.
            print(f"[SOUL-LLM] WARNING: checkpoint {os.path.basename(ckpt_path)} is "
                  f"incompatible with current vocab ({TOKENIZER.get_vocab_size()}): {e}")
            print("[SOUL-LLM] Using freshly initialized weights. "
                  "Run training to rebuild matching checkpoints.")
    else:
        print("[SOUL-LLM] No checkpoint found, using random weights")

    MODEL.eval()


@app.on_event("startup")
def startup():
    ensure_logs()
    load_model()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = TEMPERATURE
    top_k: int = TOP_K
    top_p: float = TOP_P


class GenerateResponse(BaseModel):
    generated_text: str


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200,
             temperature=0.7, top_k=40, top_p=0.9):
    """Temperature + top-k + top-p sampling decoder (shared by all API paths)."""
    input_ids = tokenizer.encode(prompt)[:-1]
    if not input_ids:
        input_ids = [tokenizer.char_to_id[tokenizer.bos_token]]
    input_ids = input_ids[-MAX_SEQ_LEN:]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    generated = input_tensor.tolist()[0]
    eos_id = tokenizer.char_to_id.get(tokenizer.eos_token, -1)

    for _ in range(max_new_tokens):
        logits, _ = model(input_tensor[:, -MAX_SEQ_LEN:])
        next_logits = logits[:, -1, :] / temperature

        if top_k > 0:
            k = min(top_k, next_logits.size(-1))
            thr = torch.topk(next_logits, k).values[:, -1].unsqueeze(-1)
            next_logits = torch.where(
                next_logits < thr,
                torch.full_like(next_logits, float("-inf")), next_logits)
        if top_p < 1.0:
            sorted_logits, idxs = torch.sort(next_logits, descending=True)
            cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = torch.where(
                mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
            next_logits = torch.zeros_like(next_logits).scatter(-1, idxs, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eos_id:
            break
        generated.append(next_token.item())
        input_tensor = torch.cat([input_tensor, next_token], dim=1)

    return tokenizer.decode(generated)


# ─── Core endpoints ──────────────────────────────────────────────────────
@app.get("/health")
@app.get("/api/health")
def health():
    state = read_json(TRAINING_STATE, {})
    return {
        "status": "healthy",
        "model_name": "SOUL-LLM (Scratch PyTorch)",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": MODEL is not None,
        "checkpoint": os.path.basename(active_checkpoint_path() or "none"),
        "training_status": state.get("status", "STOPPED"),
        "architecture_note": ARCHITECTURE_NOTE,
    }


@app.post("/api/reload")
def reload_model():
    """Reload the ACTIVE checkpoint (and tokenizer) from disk."""
    global MODEL, TOKENIZER
    MODEL = None
    TOKENIZER = None
    ckpt = active_checkpoint_path()
    load_model()
    return {
        "status": "reloaded",
        "model_loaded": MODEL is not None,
        "checkpoint": ckpt,
        "vocab_size": TOKENIZER.get_vocab_size() if TOKENIZER else None,
    }


@app.post("/api/generate", response_model=GenerateResponse)
def api_generate(req: GenerateRequest):
    text = generate(MODEL, TOKENIZER, prompt=req.prompt,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k, top_p=req.top_p)
    return GenerateResponse(generated_text=text)


@app.get("/api/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "soul-llm",
                "name": "SOUL-LLM (Scratch PyTorch Transformer)",
                "object": "model",
                "created": 1700000000,
                "owned_by": "soul-ai",
                "device": DEVICE,
                "checkpoint": os.path.basename(active_checkpoint_path() or "none"),
            }
        ]
    }


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "soul-llm"
    messages: list[ChatMessage]
    temperature: float = TEMPERATURE
    max_tokens: int = 100
    stream: bool = False
    top_k: int = TOP_K
    top_p: float = TOP_P


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    last_prompt = user_msgs[-1] if user_msgs else "Hello"

    completion = generate(
        MODEL, TOKENIZER,
        prompt=last_prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k, top_p=req.top_p,
    )

    clean_resp = completion[len(last_prompt):] if completion.startswith(last_prompt) else completion

    if req.stream:
        def stream_generator():
            words = clean_resp.split(" ")
            for i, word in enumerate(words):
                chunk_text = word + (" " if i < len(words) - 1 else "")
                chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "soul-llm",
                    "choices": [{"index": 0, "delta": {"content": chunk_text},
                                 "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                time.sleep(0.02)
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "soul-llm",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": clean_resp},
             "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": len(last_prompt),
            "completion_tokens": len(clean_resp),
            "total_tokens": len(last_prompt) + len(clean_resp),
        },
    }


# ─── Trainer controls ────────────────────────────────────────────────────
class TrainerConfigRequest(BaseModel):
    train_after: int | None = None
    worker_epochs: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    save_every_steps: int | None = None


def trainer_config():
    cfg = read_json(TRAINER_CONFIG, {})
    return {
        "train_after": int(cfg.get("train_after", TRAIN_AFTER)),
        "worker_epochs": int(cfg.get("worker_epochs", WORKER_EPOCHS)),
        "batch_size": int(cfg.get("batch_size", BATCH_SIZE)),
        "lr": float(cfg.get("lr", DISTILL_LR)),
        "save_every_steps": int(cfg.get("save_every_steps", WORKER_SAVE_EVERY_STEPS)),
    }


@app.get("/api/training/config")
def get_trainer_config():
    return {"success": True, "config": trainer_config()}


@app.post("/api/training/config")
def set_trainer_config(req: TrainerConfigRequest):
    cfg = trainer_config()
    for k, v in req.model_dump(exclude_none=True).items():
        cfg[k] = v
    write_json(TRAINER_CONFIG, cfg)
    return {"success": True, "config": cfg}


@app.get("/api/training/status")
def training_status():
    state = read_json(TRAINING_STATE, {})
    alive = worker_alive(state.get("pid"))
    ckpts = list_checkpoints()
    return {
        "success": True,
        "status": state.get("status", "STOPPED") if alive else (
            state.get("status", "STOPPED") if not state else
            "ERROR_Y" if state.get("status") == "ERROR" else "STOPPED"),
        "worker_alive": alive,
        "pid": state.get("pid"),
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "architecture_note": ARCHITECTURE_NOTE,
        "n_params": state.get("params"),
        "dataset_size": state.get("dataset_size", 0),
        "processed": state.get("processed", 0),
        "pending": state.get("pending", 0),
        "epoch": state.get("epoch"),
        "step": state.get("step"),
        "loss": state.get("loss"),
        "val_loss": state.get("val_loss"),
        "corpus_chars": state.get("corpus_chars"),
        "vocab_size": state.get("vocab_size"),
        "sequences": state.get("sequences"),
        "message": state.get("message"),
        "last_eval": state.get("last_eval"),
        "active_checkpoint": os.path.basename(active_checkpoint_path() or "none"),
        "checkpoints": ckpts,
        "config": trainer_config(),
    }


@app.post("/api/training/start")
def training_start():
    state = read_json(TRAINING_STATE, {})
    if worker_alive(state.get("pid")):
        set_control("RUN")
        return {"success": True, "message": "Worker already running — resumed.",
                "pid": state.get("pid")}
    set_control("RUN")
    pid = spawn_worker()
    write_json(TRAINING_STATE, {**state, "pid": pid})
    return {"success": True, "message": "Training worker started.", "pid": pid}


@app.post("/api/training/pause")
def training_pause():
    set_control("PAUSE")
    return {"success": True, "message": "Pause requested."}


@app.post("/api/training/resume")
def training_resume():
    state = read_json(TRAINING_STATE, {})
    if not worker_alive(state.get("pid")):
        set_control("RUN")
        pid = spawn_worker()
        write_json(TRAINING_STATE, {**state, "status": "RUNNING", "pid": pid})
        return {"success": True, "message": "Worker restarted.", "pid": pid}
    set_control("RUN")
    return {"success": True, "message": "Resumed."}


@app.post("/api/training/stop")
def training_stop():
    set_control("STOP")
    return {"success": True, "message": "Stop requested."}


class SelectModelRequest(BaseModel):
    checkpoint: str


@app.post("/api/model/select")
def select_checkpoint(req: SelectModelRequest):
    """Select which SOUL-LLM checkpoint the API serves + reload it."""
    path = os.path.abspath(req.checkpoint)
    if not os.path.isfile(path):
        # Allow a bare filename resolved against the checkpoints dir
        candidate = os.path.join(CHECKPOINT_DIR, req.checkpoint)
        if os.path.isfile(candidate):
            path = os.path.abspath(candidate)
        else:
            return {"success": False, "error": f"Checkpoint not found: {req.checkpoint}"}
    ensure_logs()
    with open(ACTIVE_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        f.write(path)
    reload_model()
    return {"success": True, "checkpoint": path}


@app.post("/api/training/evaluate")
def training_evaluate():
    """Run a real evaluation (loss on val set + generated text) synchronously."""
    from evaluation import evaluate as eval_mod
    result = eval_mod.evaluate(model=MODEL, tokenizer=TOKENIZER,
                               checkpoint_path=active_checkpoint_path(),
                               device=DEVICE)
    # Persist for UI
    state = read_json(TRAINING_STATE, {})
    state["last_eval"] = result
    write_json(TRAINING_STATE, state)
    return {"success": True, "result": result}


# ─── Frontend ────────────────────────────────────────────────────────────
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def read_root():
    frontend_path = os.path.join(BASE_DIR, "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "SOUL-LLM API is running. Use /v1/chat/completions"}