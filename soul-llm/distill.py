"""
SOUL-LLM Knowledge Distillation Pipeline
=========================================
Uses a temporary Qwen LLM teacher model to generate high-quality training
knowledge, then trains the student SOUL-LLM Transformer model.

Workflow:
1. Connect to Qwen teacher model (via local endpoint in config.yaml).
2. Query teacher across diverse prompts (AI concepts, Transformers, Python, Math, Soul AI).
3. Save teacher output separately in data/teacher_generated/.
4. Train student SOUL-LLM model on the generated data.
5. Save learned parameters in model/soul-llm/best_model.pt.
6. Once training completes, Qwen is detached and no longer required!
"""

import os
import sys
import yaml
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.teacher import QwenTeacherClient
from train import train


def load_yaml_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def run_distillation(
    config_path: str = "config.yaml",
    generate_only: bool = False,
    train_only: bool = False,
    max_samples: int = None,
    epochs: int = None
):
    print("=" * 65)
    print("      SOUL-LLM Knowledge Distillation from Qwen Teacher")
    print("=" * 65)

    cfg = load_yaml_config(config_path)
    teacher_cfg = cfg.get("teacher", {})
    paths_cfg = cfg.get("paths", {})

    output_dir = paths_cfg.get("teacher_generated_dir", "data/teacher_generated")
    teacher_file = paths_cfg.get("teacher_data_file", os.path.join(output_dir, "distilled_data.txt"))

    # ── Phase 1: Teacher Knowledge Generation ─────────────────────────────
    if not train_only:
        print("\n--- Phase 1: Querying Temporary Qwen Teacher ---")
        teacher_client = QwenTeacherClient(
            endpoint=teacher_cfg.get("endpoint", "http://localhost:11434"),
            model=teacher_cfg.get("model", "qwen2.5:7b"),
            timeout=teacher_cfg.get("timeout_seconds", 60)
        )

        is_online = teacher_client.check_teacher_available()
        if is_online:
            print(f"[OK] Qwen teacher model '{teacher_client.model}' verified reachable at {teacher_client.endpoint}.")
        else:
            print(f"[WARN] Qwen teacher at {teacher_client.endpoint} not reachable. Using cached/fallback educational corpus.")

        distilled_corpus = teacher_client.generate_distillation_corpus(
            output_dir=output_dir,
            max_samples=max_samples
        )
        print(f"[OK] Teacher knowledge data generated and saved to: {teacher_file}")

    if generate_only:
        print("\n[OK] Generation only requested. Exiting without training.")
        return

    # ── Phase 2: Student SOUL-LLM Training ────────────────────────────────
    print("\n--- Phase 2: Training Student SOUL-LLM Model ---")
    best_ckpt = train(config_path=config_path, extra_corpus_path=teacher_file, epochs=epochs)

    # ── Phase 3: Post-Distillation Summary ─────────────────────────────────
    print("\n" + "=" * 65)
    print("   [OK] KNOWLEDGE DISTILLATION SUCCESSFULLY COMPLETED!")
    print("=" * 65)
    print(f"1. Student Model Checkpoint : {best_ckpt}")
    print(f"2. Teacher Data Stored At   : {teacher_file}")
    print("3. Next Step                : Run 'python verify_independence.py'")
    print("   to verify that SOUL-LLM runs 100% independently without Qwen!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOUL-LLM Knowledge Distillation Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--generate-only", action="store_true", help="Only generate teacher data")
    parser.add_argument("--train-only", action="store_true", help="Only train student on existing teacher data")
    parser.add_argument("--samples", type=int, default=None, help="Limit number of teacher prompt samples")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    args = parser.parse_args()

    run_distillation(
        config_path=args.config,
        generate_only=args.generate_only,
        train_only=args.train_only,
        max_samples=args.samples,
        epochs=args.epochs
    )
