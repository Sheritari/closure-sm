#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${CUDA_ARCH:-}"
if [[ -z "$ARCH" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    CCAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d " ")
    if [[ -n "$CCAP" ]]; then
      MAJOR=${CCAP%%.*}
      MINOR=${CCAP#*.}
      ARCH="sm_${MAJOR}${MINOR}"
      echo "Detected GPU compute capability ${CCAP} -> -arch=${ARCH}"
    fi
  fi
fi
if [[ -z "$ARCH" ]]; then
  ARCH="native"
  echo "CUDA_ARCH not set; using -arch=native"
fi

nvcc -O3 -std=c++17 --use_fast_math -Xptxas -O3 \
  -arch="${ARCH}" \
  closure_sm_batch.cu -o closure_sm_batch

echo "Built: $(pwd)/closure_sm_batch"
