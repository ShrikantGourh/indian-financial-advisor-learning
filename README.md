# Indian Financial Advisor (research assistant)

This learning project successfully completed a one-epoch PyTorch/LoRA training
run for Qwen2.5-0.5B-Instruct on CPU on a machine identified as a mid-2012
MacBook running Ubuntu. The resulting adapter is an experimental pilot, not a
successful financial adviser: it failed a basic arithmetic inference check
(90% returned where 10% was correct). This limitation is documented rather
than hidden. No production Ollama model has been exported.

See [build_log.html](./build_log.html) for the collapsible plan, project
structure, implementation notes, and current progress.
See [MODEL_CARD.md](./MODEL_CARD.md) for the pilot's training details, intended
use, limitations, and evaluation result.

> **Important:** This is an experimental market-analysis model, not a
> registered financial adviser. It can analyze dated data supplied to it, but
> this project currently has no live market-data connector. Fine-tuning alone
> cannot make its knowledge current or its forecasts reliable. It must not
> provide individualized trade instructions or execute trades. Verify market
> facts and applicable Indian regulations independently.

## Why training uses Hugging Face weights

The Ollama copy (`qwen2.5:0.5b`) is a runtime model, generally stored in a
quantized inference format. Ollama does not provide a fine-tuning interface for
it. The training script therefore loads the compatible open-weight
`Qwen/Qwen2.5-0.5B-Instruct` checkpoint through Transformers. This requires an
internet connection the first time and accepts the model publisher's license
and usage terms. The Ollama copy remains useful for comparing the base model
before and after fine-tuning.

The starter examples are illustrative and intentionally contain no live
prices. Replace or extend them only with appropriately licensed, timestamped
data and reviewed market-analysis examples. Keep a separate holdout set and
assess accuracy, risk coverage, and hallucinations before any use.

The current starter corpus has 36 training records and 8 validation records.
Most numeric cases are explicitly hypothetical; these records are suitable for
a pipeline smoke test, not as expert-validated financial research or a
production-quality model. The two captured live-page snapshots are kept
separate and are not used as training examples. Their screenshots and raw
snapshot file remain local and are excluded from the public source repository
because redistribution rights were not verified.

Copilot browser captures and their provenance are kept separately in
`data/captures/` and `data/snapshots-2026-11-23.jsonl`. They are one-time
observations, not a continuous feed or fine-tuning examples. The Google Finance
quote's exchange delay was not independently verified; the MCX figure is a
dated futures contract, not a retail/spot gold rate.

## Setup

Use Python 3.10–3.14 and a virtual environment. This project was checked on
Python 3.14.4 with CPU-only PyTorch 2.15.0. CPU fine-tuning is very slow; PyTorch
installation can differ by OS and CUDA version, so select a matching build from
the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

```bash
cd indian_financial_advisor
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# CPU-only Linux example; choose the PyTorch build appropriate for your machine.
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple 'torch==2.15.0+cpu'
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/validate_data.py
```

The 0.5B model can be fine-tuned with LoRA on a modest GPU; CPU training is
supported but can be very slow. Tune batch size and sequence length for the
available memory. Training downloads the Hugging Face checkpoint if it is not
already cached. The current workspace has no NVIDIA GPU; its first pilot ran
on CPU. The observed host reports an Intel Core i5-3210M at 2.50 GHz, four
logical CPUs, 15 GiB of memory, and Ubuntu x86_64. Training took about 16
minutes for the small pilot.

## What this project teaches

Qwen2.5-0.5B-Instruct is a decoder-only transformer. It generates text
autoregressively: each next-token distribution is conditioned on earlier
tokens. Supervised fine-tuning formats each example as a chat conversation,
then minimizes token-level cross-entropy on the assistant's target tokens.
Prompt and padding labels are set to `-100`, so PyTorch's loss ignores them.

The code loads the Hugging Face model checkpoint into PyTorch; Ollama's
quantized runtime copy is not the training input. PEFT inserts small LoRA
matrices into selected attention and feed-forward projections. The original
model weights are frozen while the much smaller adapter is optimized, reducing
training memory and the size of the saved update.

For the recorded run, the script used rank 8, alpha 16, dropout 0.05, one
epoch, maximum sequence length 512, batch size 1, gradient accumulation 8,
and learning rate 2e-4. It processed 36 training examples and evaluated on 8
held-out examples. Train loss was 3.4241 and validation loss was 3.0805.
Losses show that the training loop ran and provide a debugging signal; they do
not establish factuality, arithmetic competence, safety, or financial utility.

The meaningful success is learning and completing the CPU training pipeline on
older hardware: dataset validation, tokenization, assistant-only labels,
gradient-checkpointed LoRA training, adapter saving, and local inference. The
model-quality gate did not pass. The small synthetic corpus and the observed
wrong calculation mean this adapter must not be used for market decisions.

## Fine-tune

```bash
.venv/bin/python scripts/train.py \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --output-dir artifacts/adapter
```

The script supervises assistant turns only and saves a PEFT adapter. The
included tiny sample set is for a smoke test, not a useful training corpus.
For meaningful results, use a substantially larger, reviewed, properly
licensed dataset and evaluate on examples not used during training.

### Current CPU pilot

A one-epoch pilot was run on the 36/8 starter split and saved to
`artifacts/pilot-cpu/`. Training loss was 3.4241 and validation loss was
3.0805. A basic arithmetic inference check returned an incorrect net-margin
formula/result, so this adapter is **not suitable for use or Ollama export**.
The run confirms the training pipeline can complete; it does not demonstrate
model quality. Improve the dataset and evaluation before another run.

## Export to Ollama

Install and build [llama.cpp](https://github.com/ggml-org/llama.cpp), then
merge the adapter into the original Hugging Face model and convert/quantize it:

```bash
.venv/bin/python scripts/merge_adapter.py \
  --adapter artifacts/adapter \
  --output-dir artifacts/merged

bash scripts/export_ollama.sh /path/to/llama.cpp
```

The export script creates an Ollama model from the GGUF file and checks that
the target name is available:

```bash
ollama run indian_financial_advisor
```

To try inference without creating a merged model, continue using
`ollama run qwen2.5:0.5b` as the base model. This project does not automatically
download model weights, install system tools, or start a training run.

You can also open `build_log.html` and use its **Try a local model**
panel. It discovers installed models from Ollama, sends the prompt to the
selected model on `localhost`, and displays the response. The pilot adapter is
not imported into Ollama, so this panel currently tests an installed base
model, not the fine-tuned adapter. Ollama must be running.

To select the experimental LoRA pilot in the same panel, start its loopback
inference server in another terminal:

```bash
.venv/bin/python scripts/serve_adapter.py
```

Refresh the HTML page and choose `indian_financial_advisor:pilot-cpu`. The
server loads the local adapter and base checkpoint, accepts only local browser
origins, and is separate from Ollama. The pilot already failed an arithmetic
quality check; use it for experimentation only, not financial decisions.

## Dataset format

Each UTF-8 JSONL line contains a `messages` array with `system`, `user`, and
`assistant` roles, following the conversational format in `data/`. Do not
include confidential client information, personal account details, or
unlicensed market data. The system prompt is a training aid, not a substitute
for product-level safeguards or professional review.
