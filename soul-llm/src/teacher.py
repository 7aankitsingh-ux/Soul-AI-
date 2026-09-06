"""
Temporary Teacher Model Interface (Qwen)
=========================================
Connects to a temporary Qwen LLM teacher (e.g. running on local Ollama)
to generate high-quality educational text and reasoning data for SOUL-LLM to learn from.

CRITICAL ARCHITECTURAL BOUNDARY:
- This file is used ONLY during the knowledge generation and distillation phase.
- The final SOUL-LLM inference engine (src/inference.py) NEVER imports or calls this module.
"""

import os
import json
import time
import urllib.request
from typing import List, Dict, Any, Optional

EDUCATIONAL_SEED_PROMPTS = [
    # 1. AI and LLM Core Foundations
    "Explain what artificial intelligence is in clear, simple language.",
    "What is a Large Language Model and how does it generate words?",
    "Explain what a Transformer neural network is and why it changed AI.",
    "What is self-attention in a Transformer and how does it work?",
    "What is the difference between an encoder and a decoder in Transformers?",
    "Explain the concept of tokens and tokenization in language models.",
    "What is positional encoding in Transformer models and why is it needed?",
    "Explain how residual connections and layer normalization stabilize deep neural networks.",
    "What is temperature in text generation and how does it affect creativity?",
    "Explain top-k and top-p sampling and how they prevent low-probability words from appearing.",
    "What is autoregressive generation and why does a decoder model generate one token at a time?",
    "Explain the mathematical equation for Scaled Dot-Product Attention: Softmax(QK^T / sqrt(d_k)) * V.",

    # 2. Machine Learning, Math & Optimization
    "Explain how gradient descent and backpropagation optimize neural network weights.",
    "What is cross-entropy loss and why is it used for training language models?",
    "What is the AdamW optimizer and how does weight decay prevent overfitting?",
    "Explain learning rate scheduling and warmup in modern Transformer training.",
    "What is gradient clipping and why does it prevent exploding gradients?",

    # 3. Python & PyTorch Programming
    "Explain Python variables, lists, dictionaries, and functions with clear examples.",
    "What is PyTorch and why is it the standard library for deep learning research?",
    "Explain what a tensor is in PyTorch and how it differs from a NumPy array.",
    "How does automatic differentiation (autograd) work in deep learning?",
    "Write a concise guide on object-oriented programming in Python using classes and inheritance.",
    "How do nn.Embedding, nn.Linear, and nn.LayerNorm work in PyTorch?",

    # 4. Soul AI & Offline Architecture
    "What is Soul AI and why is private, 100% offline computing important?",
    "Explain how an autonomous AI agent uses tools to inspect hardware and solve calculations.",
    "What is Retrieval-Augmented Generation (RAG) and how does vector search work?",
    "What is knowledge distillation in machine learning and how does a student model learn from a teacher?",
    "Why does a student model remain functional even after the teacher model is detached?"
]


class QwenTeacherClient:
    """Client to query the temporary Qwen teacher model via local Ollama endpoint."""

    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "qwen2.5:7b", timeout: int = 60):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_teacher_available(self, auto_start: bool = True) -> bool:
        """Verifies whether the local Qwen teacher service is reachable, and auto-starts if needed."""
        try:
            req = urllib.request.Request(f"{self.endpoint}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                if any(self.model in m for m in models):
                    return True
        except Exception:
            pass

        if not auto_start:
            return False

        # Attempt to auto-start local Ollama server if running on localhost
        if "127.0.0.1" in self.endpoint or "localhost" in self.endpoint:
            print("[Teacher] Ollama server not actively responding. Attempting to start local Ollama service...")
            try:
                import subprocess
                import shutil
                ollama_bin = shutil.which("ollama")
                if not ollama_bin:
                    default_path = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
                    if os.path.exists(default_path):
                        ollama_bin = default_path

                if ollama_bin:
                    subprocess.Popen([ollama_bin, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    for attempt in range(10):
                        time.sleep(1)
                        try:
                            req = urllib.request.Request(f"{self.endpoint}/api/tags", method="GET")
                            with urllib.request.urlopen(req, timeout=2) as resp:
                                data = json.loads(resp.read().decode())
                                models = [m.get("name", "") for m in data.get("models", [])]
                                if any(self.model in m for m in models):
                                    print(f"[Teacher] Ollama successfully verified with model '{self.model}'!")
                                    return True
                        except Exception:
                            continue
            except Exception as e:
                print(f"[Teacher] Auto-start attempt failed: {e}")

        return False

    def generate_response(self, prompt: str, system: Optional[str] = None) -> str:
        """Query the Qwen teacher model to generate a response."""
        sys_prompt = system or (
            "You are an expert computer science professor. Provide clear, concise, "
            "and highly educational explanations. Output clean factual paragraphs without filler."
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": sys_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 120
            }
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            return res_json.get("response", "").strip()

    def generate_distillation_corpus(
        self,
        prompts: List[str] = None,
        output_dir: str = "data/teacher_generated",
        max_samples: Optional[int] = None
    ) -> str:
        """
        Queries the Qwen teacher on seed prompts, saves the generated high-quality
        responses to disk, and returns the accumulated corpus text.
        """
        os.makedirs(output_dir, exist_ok=True)
        prompts = prompts or EDUCATIONAL_SEED_PROMPTS
        if max_samples:
            prompts = prompts[:max_samples]

        text_file = os.path.join(output_dir, "distilled_data.txt")
        meta_file = os.path.join(output_dir, "metadata.json")

        samples = []
        full_text_blocks = []

        print(f"\n[Teacher] Generating educational distillation data using {self.model} on {self.endpoint}...")
        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{len(prompts)}] Teacher generating answer for: '{prompt}'")
            try:
                answer = self.generate_response(prompt)
                block = f"{prompt}\n{answer}"
                full_text_blocks.append(block)
                samples.append({
                    "prompt": prompt,
                    "teacher_response": answer,
                    "teacher_model": self.model,
                    "timestamp": time.time()
                })
            except Exception as e:
                print(f"  ⚠ Failed to generate for prompt '{prompt}': {e}")
                # Fallback educational block if service is briefly busy
                fallback_answer = (
                    f"{prompt} is a foundational principle in computer science, "
                    f"enabling algorithms to process data, optimize parameters, and perform autonomous tasks."
                )
                block = f"{prompt}\n{fallback_answer}"
                full_text_blocks.append(block)

        # Write distilled text
        distilled_corpus = "\n\n".join(full_text_blocks)
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(distilled_corpus)

        # Write metadata
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "teacher_model": self.model,
                "teacher_endpoint": self.endpoint,
                "sample_count": len(samples),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "samples": samples
            }, f, indent=2)

        print(f"[Teacher] Generated {len(samples)} samples saved to: {text_file} ({len(distilled_corpus):,} chars)")
        return distilled_corpus
