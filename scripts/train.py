#!/usr/bin/env python3
"""LoRA fine-tuning for Qwen2.5-0.5B-Instruct with assistant-only loss."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROJECT_DIR = Path(__file__).resolve().parents[1]


class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tokenizer: Any, max_length: int) -> None:
        self.records = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    messages = json.loads(line)["messages"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(f"{path}:{line_number}: invalid conversation record") from error

                input_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )
                labels = [-100] * len(input_ids)
                for index, message in enumerate(messages):
                    if message["role"] != "assistant":
                        continue
                    prefix = tokenizer.apply_chat_template(
                        messages[:index], tokenize=True, add_generation_prompt=True
                    )
                    assistant_turn = tokenizer.apply_chat_template(
                        messages[: index + 1], tokenize=True, add_generation_prompt=False
                    )
                    if assistant_turn[: len(prefix)] != prefix:
                        raise ValueError(
                            f"{path}:{line_number}: chat template does not preserve the "
                            "assistant generation prefix"
                        )
                    labels[len(prefix) : len(assistant_turn)] = assistant_turn[len(prefix) :]

                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
                if not any(label != -100 for label in labels):
                    raise ValueError(
                        f"{path}:{line_number}: no assistant tokens remain; increase --max-length"
                    )
                self.records.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        "labels": labels,
                    }
                )

        if not self.records:
            raise ValueError(f"{path}: no training records found")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.records[index]


class CausalLMCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_masks, labels = [], [], []
        for feature in features:
            padding = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=PROJECT_DIR / "data/train.jsonl")
    parser.add_argument("--validation", type=Path, default=PROJECT_DIR / "data/validation.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "artifacts/adapter")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size < 1 or args.gradient_accumulation_steps < 1 or args.max_length < 1:
        parser.error("batch size, gradient accumulation steps, and max length must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = ConversationDataset(args.train, tokenizer, args.max_length)
    validation_dataset = ConversationDataset(args.validation, tokenizer, args.max_length)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=torch.cuda.is_available() and dtype == torch.float16,
        bf16=torch.cuda.is_available() and dtype == torch.bfloat16,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
