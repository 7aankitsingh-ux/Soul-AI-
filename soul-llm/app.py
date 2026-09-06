"""
SOUL-LLM Independent Web Application
====================================
FastAPI web application and REST API serving the independent SOUL-LLM model.
Zero external LLM imports, zero Qwen dependencies. Runs 100% offline.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.inference import generate, load_independent_soul_llm

app = FastAPI(
    title="SOUL-LLM Independent Engine",
    description="Offline Decoder-Only Transformer built from scratch and trained via Knowledge Distillation",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text to generate completion from")
    max_tokens: int = Field(default=80, ge=1, le=500)
    temperature: float = Field(default=0.8, ge=0.01, le=2.0)
    top_k: int = Field(default=40, ge=0, le=100)
    deterministic: bool = Field(default=False)


class ChatCompletionRequest(BaseModel):
    model: str = "soul-llm"
    messages: list = []
    temperature: float = 0.8
    max_tokens: int = 100
    top_k: int = 40
    stream: bool = False


@app.on_event("startup")
def startup_event():
    """Pre-load SOUL-LLM weights on startup."""
    try:
        load_independent_soul_llm()
        print("[OK] SOUL-LLM student model loaded and ready.")
    except Exception as e:
        print(f"[WARN] Model load warning on startup: {e}")


@app.get("/health")
@app.get("/api/health")
def health():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "healthy",
        "model": "SOUL-LLM (Independent Student Transformer)",
        "teacher_dependency": "NONE (Fully Detached)",
        "offline": True,
        "device": dev
    }


@app.get("/v1/models")
@app.get("/api/tags")
@app.get("/api/models")
def list_models():
    """Lists available models for OpenAI clients and Soul AI Web Cockpit."""
    return {
        "object": "list",
        "data": [
            {
                "id": "soul-llm",
                "name": "SOUL-LLM (Scratch PyTorch Transformer)",
                "object": "model",
                "created": 1700000000,
                "owned_by": "soul-ai"
            }
        ],
        "models": [
            {
                "name": "soul-llm",
                "model": "soul-llm",
                "modified_at": "2026-09-06T00:00:00Z",
                "size": 26240869
            }
        ]
    }


# Soul AI Agent Integration
import soul_ai
from soul_ai import SoulAIEngine, execute_tool, get_tool_definitions, get_system_info, rag_engine

soul_agent_engine = SoulAIEngine(model_generator=lambda p: generate(prompt=p, max_new_tokens=100, temperature=0.7))


class AgentChatRequest(BaseModel):
    message: str = ""
    temperature: float = 0.7


@app.post("/api/agent/chat")
def agent_chat(req: AgentChatRequest):
    """Runs Soul AI autonomous reasoning loop with tool execution in Python."""
    result = soul_agent_engine.run_agent_loop(req.message)
    return {
        "success": True,
        "response": result["response"],
        "iterations": result.get("iterations", 1),
        "tool_calls": result.get("tool_calls", [])
    }


@app.get("/api/tools")
def get_tools():
    """Lists available offline tools in Soul AI."""
    return {"tools": get_tool_definitions()}


@app.get("/api/system")
def get_system():
    """Returns local system telemetry."""
    return {"system": get_system_info()}


@app.post("/generate")
@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    try:
        completion = generate(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            deterministic=req.deterministic
        )
        return {
            "prompt": req.prompt,
            "response": completion,
            "model": "SOUL-LLM",
            "teacher_involved": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
@app.post("/api/chat")
def chat_completions(req: ChatCompletionRequest):
    # Extract latest user message
    user_msgs = []
    for m in req.messages:
        if isinstance(m, dict):
            if m.get("role") == "user":
                user_msgs.append(m.get("content", ""))
        elif hasattr(m, "content"):
            user_msgs.append(m.content)
        elif isinstance(m, str):
            user_msgs.append(m)

    last_prompt = user_msgs[-1] if user_msgs else "Hello"

    full_output = generate(
        prompt=last_prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k
    )

    clean_resp = full_output[len(last_prompt):] if full_output.startswith(last_prompt) else full_output

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
                    "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]
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
            {
                "index": 0,
                "message": {"role": "assistant", "content": clean_resp},
                "finish_reason": "stop"
            }
        ]
    }


# Frontend static files
frontend_dir = os.path.join(ROOT_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
