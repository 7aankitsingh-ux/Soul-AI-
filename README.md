# Soul AI — Offline Personal AI Companion

A private, offline AI agent that runs entirely on your own computer. It uses a web dashboard
(and CLI) powered by local open-weights models via **Ollama**, and includes a built-in
distillation pipeline that teaches a small from-scratch Transformer (**SOUL-LLM**) from your
Qwen conversations using real PyTorch training.

> 100% local. No cloud, no telemetry, no data leaves your machine.

---

## Features

- **ReAct agent loop** – thinks, picks tools, executes, and iterates until the task is done.
- **Offline tool suite** – file read/write/search, safe shell commands, math evaluation,
  system/GPU telemetry, local BM25 RAG, and cross-session memory.
- **Black & white web dashboard** – searchable chat history (saved even as a guest),
  Sign In / Sign Up accounts, SSE streaming with thinking + tool progress, model hub.
- **SOUL-LLM Distillation Trainer (Settings tab)**:
  - Qwen 2.5 7B (teacher) answers are captured to `soul-llm/data/qwen_teacher.jsonl`.
  - A background worker runs **real PyTorch training** (`train.py`) on SOUL-LLM (student).
  - Start / Pause / Resume / Stop, live loss progress, checkpoint save + select, evaluate.
  - SOUL-LLM keeps its own weights and never calls Qwen during chat.
- **CLI companion** – fast terminal sessions (`npm run cli`).

## Quickstart

Prerequisites: [Node.js](https://nodejs.org) 18+, [Python](https://python.org) 3.10+,
[Ollama](https://ollama.com).

```powershell
# 1. Pull a model
ollama pull qwen2.5:7b

# 2. Install web deps
npm install

# 3. Optional: Python deps for the SOUL-LLM trainer
pip install -r soul-llm/requirements.txt

# 4. Start the web dashboard
npm start
```

Open **http://localhost:3000**.

Optional background services (for the trainer):

```powershell
# Start the SOUL-LLM API (port 8000)
python -m uvicorn api.main:app --port 8000   # from the soul-llm/ folder

# Start the background training worker
npm run trainer
```

To point the app at a remote SOUL-LLM API (e.g. a cloud instance), set `SOUL_LLM_API`:
```powershell
$env:SOUL_LLM_API = "https://your-api.example.com"; npm start
```

## Deploy to Render (free public URL)

Get a public link anyone can open — like `https://soul-ai.onrender.com` — in a few clicks.

1. Create a free account at [render.com](https://render.com) (GitHub sign-in works).
2. Dashboard → **New** → **Blueprint** → pick this repo (`7aankitsingh-ux/Soul-AI-`).
3. `render.yaml` is detected automatically → hit **Apply**. Render builds both services:
   - `soul-ai` — the web app (Node/Express).
   - `soul-llm-api` — the Python + PyTorch SOUL-LLM service (CPU build).
4. When both show **Live**, open the `soul-ai` URL. Done.

Notes:
- Free-tier servers are limited (512 MB RAM, sleep after ~15 min idle; wake on first visit).
- Chat uses **SOUL-LLM** when the Python service is up. If that service is slow or off
  (large model download, RAM limits), the app still works via the built-in offline agent,
  plus the workspace and RAG tools.
- User accounts and chat history live in server-side storage and reset on redeploy — treat
  the public URL as a demonstration instance.

## Tests

```powershell
npm test                 # agent + tools + RAG
python soul-llm/tests/test_all.py
```

## Upload to GitHub (once)

The repo is pre-configured: `.gitignore` keeps private data out
(user accounts, chat history, distilled conversations, checkpoints, logs, generated files).

```powershell
# Create an empty repo on github.com (no README needed), then:
scripts\github-upload.ps1 -Repo https://github.com/YOUR_USERNAME/soul-ai
```

After that, updates are the normal flow:

```powershell
git add -A
git commit -m "message"
git push
```

## License

MIT — see [LICENSE](LICENSE).