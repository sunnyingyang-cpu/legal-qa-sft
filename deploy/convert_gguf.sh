#!/usr/bin/env bash
# 阶段 5：把 merge 后的 HF 权重转成 GGUF（Q8_0，1.5B 约 1.6GB，几乎无损）。
# 用法：bash deploy/convert_gguf.sh [merged模型目录] [输出gguf路径]
set -euo pipefail

MODEL_DIR=${1:-models/qwen2.5-1.5b-legal-merged}
OUT=${2:-models/qwen2.5-1.5b-legal-q8_0.gguf}

cd /content
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp
fi
pip install -q -r llama.cpp/requirements.txt

python llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" --outfile "$OUT" --outtype q8_0
ls -lh "$OUT"
