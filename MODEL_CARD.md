---
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- lora
- transformers
- indian-markets
- educational
- experimental
---

# indian_financial_advisor:pilot-cpu

An educational LoRA fine-tuning pilot for Indian stock-market and gold
research, based on Qwen2.5-0.5B-Instruct. The PyTorch training pipeline
completed on CPU; **this adapter failed a basic arithmetic inference check and
is not a reliable financial-analysis model.**

## Intended use

This artifact is shared to teach a small end-to-end PyTorch/Transformers/PEFT
workflow on older CPU hardware. It is experimental and for local learning only.
Do not use it to make investment, trading, or other financial decisions.

## Model and training

- Base checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`
- Method: PEFT LoRA; rank 8, alpha 16, dropout 0.05
- Trainable targets: attention and feed-forward projection layers
- Device: CPU, float32; gradient checkpointing enabled
- Host: user-identified mid-2012 MacBook running Ubuntu; observed CPU
  Intel Core i5-3210M at 2.50 GHz, 4 logical CPUs, 15 GiB RAM
- Data: 36 training and 8 validation chat scenarios, mostly hypothetical;
  not expert-reviewed. No captured live market quotes were used for training.
- Run: one epoch, max sequence length 512, batch size 1, gradient
  accumulation 8, learning rate 2e-4
- Training loss: 3.4241
- Validation loss: 3.0805
- Recorded training time: about 16 minutes

The loss values show that optimization completed; they do not prove answer
quality. In a manual test with hypothetical revenue of ₹500 crore and net
profit of ₹50 crore, the correct net margin is 10%. This adapter returned
90% and used an incorrect formula. **The quality check failed.** Do not treat
this as a successful financial adviser or export it for real-world use.

## How the training works

Qwen2.5 is a decoder-only transformer that predicts the next token from the
preceding context. The training script formats JSONL messages with the model's
chat template and minimizes token-level cross-entropy on assistant responses.
Prompt and padding tokens are masked from the loss. LoRA learns small
low-rank weight updates while the much larger base-model weights remain frozen.
The saved artifact is an adapter, not a complete standalone model.

## Loading for experimentation

Install compatible versions of `torch`, `transformers`, and `peft`, then load
the base checkpoint and adapter:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_id = "shrikantgrh/indian_financial_advisor"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id)
model = PeftModel.from_pretrained(base, adapter_id)
```

Model: https://huggingface.co/shrikantgrh/indian_financial_advisor

Source project:
https://github.com/ShrikantGourh/indian-financial-advisor-learning

## Limitations and risk

- The dataset is tiny, synthetic/hypothetical, and not an expert-reviewed
  financial corpus.
- The adapter failed the stated arithmetic check and can produce incorrect
  calculations, unsupported claims, or misleading analysis.
- It has no live market-data feed and cannot know current prices unless a
  separate, dated data source is provided.
- It is not a registered financial adviser and must not provide personalized
  recommendations, price targets, or guaranteed returns.
- Validate all outputs independently; this artifact is not suitable for
  real-world financial decisions.

## Evaluation

Only a basic manual arithmetic smoke check was recorded. It failed as described
above. The 8-example validation loss is not a robust benchmark. No claim of
financial accuracy, prediction skill, or production readiness is made.
