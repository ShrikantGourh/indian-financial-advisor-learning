#!/usr/bin/env python3
"""Merge a trained PEFT adapter into its original Hugging Face base model."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=PROJECT_DIR / "artifacts/adapter")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "artifacts/merged")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not (args.adapter / "adapter_config.json").is_file():
        parser.error(f"no PEFT adapter found at {args.adapter}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.output_dir)
    print(f"Merged Hugging Face model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
