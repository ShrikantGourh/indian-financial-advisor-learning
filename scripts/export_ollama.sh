#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
llama_cpp_dir="${1:?Usage: export_ollama.sh /path/to/llama.cpp [quantization]}"
quantization="${2:-Q4_K_M}"
converter="${llama_cpp_dir}/convert_hf_to_gguf.py"
quantizer="${llama_cpp_dir}/build/bin/llama-quantize"
merged_model="${project_dir}/artifacts/merged"
export_dir="${project_dir}/artifacts/ollama"
fp16_model="${export_dir}/indian_financial_advisor-f16.gguf"
quantized_model="${export_dir}/indian_financial_advisor-${quantization}.gguf"

if [[ ! -d "${merged_model}" ]]; then
  echo "Merged model not found at ${merged_model}; run scripts/merge_adapter.py first." >&2
  exit 1
fi
if [[ ! -f "${converter}" || ! -x "${quantizer}" ]]; then
  echo "llama.cpp converter or build/bin/llama-quantize not found under ${llama_cpp_dir}." >&2
  exit 1
fi

mkdir -p "${export_dir}"
python3 "${converter}" "${merged_model}" --outfile "${fp16_model}" --outtype f16
"${quantizer}" "${fp16_model}" "${quantized_model}" "${quantization}"

cat > "${export_dir}/Modelfile" <<'EOF'
FROM ./indian_financial_advisor-Q4_K_M.gguf
PARAMETER temperature 0.2
SYSTEM You are a direct, evidence-led Indian market research assistant focused on stocks and gold, not a registered financial adviser. Do not provide individualized buy, sell, or hold instructions, price targets, or guaranteed returns. State when data is missing or stale, distinguish sourced facts from analysis and scenarios, and explain material risks.
EOF

if [[ "${quantization}" != "Q4_K_M" ]]; then
  sed "s/indian_financial_advisor-Q4_K_M.gguf/indian_financial_advisor-${quantization}.gguf/" \
    "${export_dir}/Modelfile" > "${export_dir}/Modelfile.tmp"
  mv "${export_dir}/Modelfile.tmp" "${export_dir}/Modelfile"
fi

ollama create indian_financial_advisor -f "${export_dir}/Modelfile"
echo "Created Ollama model: indian_financial_advisor"
