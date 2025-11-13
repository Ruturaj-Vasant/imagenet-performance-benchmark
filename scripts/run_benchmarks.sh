#!/usr/bin/env bash
set -euo pipefail

# Run 1-epoch ImageNet/Tiny-ImageNet benchmarks for 3 models and save logs.
# Usage:
#   bash scripts/run_benchmarks.sh [DATASET_DIR] [BATCH_SIZE] [NUM_WORKERS] [--dummy]
# Defaults:
#   DATASET_DIR=tiny-imagenet-200, BATCH_SIZE=128, NUM_WORKERS=8

DATASET_DIR=${1:-tiny-imagenet-200}
BATCH_SIZE=${2:-128}
NUM_WORKERS=${3:-8}
EXTRA="${4:-}"

MODELS=(
  resnet18
  mobilenet_v2
  vit_b_16
)

timestamp=$(date +"%Y%m%d_%H%M%S")
outdir="results/${timestamp}"
mkdir -p "${outdir}"

echo "Benchmark start: ${timestamp}" | tee "${outdir}/meta.txt"
echo "Dataset: ${DATASET_DIR}" | tee -a "${outdir}/meta.txt"
echo "Batch size: ${BATCH_SIZE}, Workers: ${NUM_WORKERS}" | tee -a "${outdir}/meta.txt"
echo "Extra flags: ${EXTRA}" | tee -a "${outdir}/meta.txt"
echo "System: $(uname -a)" | tee -a "${outdir}/meta.txt"

# Record Python/torch env
{
  echo "Python: $(python3 -V 2>&1)"
  python3 - <<'PY'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("accelerator available:", getattr(torch, "accelerator").is_available() if hasattr(torch, "accelerator") else False)
print("cuda available:", torch.cuda.is_available())
print("mps available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY
} | tee "${outdir}/env.txt"

combined_csv="${outdir}/metrics_all.csv"

model_available() {
  local name="$1"
  python3 - "$name" <<'PY'
import sys, torchvision.models as M
name = sys.argv[1]
ok = name in [n for n in M.__dict__ if n.islower() and callable(M.__dict__[n])]
print('YES' if ok else 'NO')
PY
}

run_one() {
  local model="$1"
  local log="${outdir}/${model}.log"
  local csv_model="${outdir}/metrics_${model}.csv"

  if [ "$(model_available "$model")" != "YES" ]; then
    echo "[SKIP] ${model} not available in this torchvision build" | tee -a "${outdir}/meta.txt"
    return 0
  fi

  echo "\n==> Running model: ${model}" | tee -a "${outdir}/meta.txt"
  if python3 imagenet/main.py "${DATASET_DIR}" -a "${model}" --epochs 1 -b "${BATCH_SIZE}" -j "${NUM_WORKERS}" --print-freq 10 \
       --log-csv "${csv_model}" ${EXTRA} 2>&1 | tee "${log}" ; then
    echo "[OK] ${model} log: ${log} | csv: ${csv_model}" | tee -a "${outdir}/meta.txt"
    # Append to combined CSV
    if [ -f "${csv_model}" ]; then
      if [ ! -f "${combined_csv}" ]; then
        cp "${csv_model}" "${combined_csv}"
      else
        tail -n +2 "${csv_model}" >> "${combined_csv}"
      fi
    fi
  else
    echo "[FAIL] ${model} — see ${log}" | tee -a "${outdir}/meta.txt"
  fi
}

for m in "${MODELS[@]}"; do
  run_one "$m"
done

echo "\nBenchmark logs saved to: ${outdir}" | tee -a "${outdir}/meta.txt"
