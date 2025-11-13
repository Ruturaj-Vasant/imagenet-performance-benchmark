#!/usr/bin/env bash
set -euo pipefail

# ============================================
#   FULL AUTOMATED IMAGENET BENCHMARK PIPELINE
#   GCP VM / Any Linux with Python3
# ============================================

# Config (edit as needed)
REPO_URL="https://github.com/Ruturaj-Vasant/imagenet-performance-benchmark.git"
REPO_DIR="imagenet-performance-benchmark"
DATA_DIR="data"
TINYTAR_URL="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINYTAR_ZIP="$DATA_DIR/tiny.zip"
TINY_DIR="$DATA_DIR/tiny-imagenet-200"
RESULTS_DIR="results"
MODELS=(resnet18 mobilenet_v2 resnet50 vgg11_bn)
BATCH_SIZE=${BATCH_SIZE:-128}
WORKERS=${WORKERS:-4}
MAX_BATCHES=${MAX_BATCHES:-50}
PRINT_FREQ=${PRINT_FREQ:-10}

echo "Python3: $(python3 -V 2>&1)" || true
python3 - <<'PY' || true
import torch
print("PyTorch version:", getattr(torch, "__version__", "unknown"))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "\n==> Installing Python dependencies"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q pandas matplotlib numpy thop psutil || true

if [ ! -d "$REPO_DIR" ]; then
  echo "\n==> Cloning repo: $REPO_URL"
  git clone "$REPO_URL"
fi

cd "$REPO_DIR"

echo "\n==> Repo requirements"
python3 -m pip install -q -r imagenet/requirements.txt || true

echo "\n==> Preparing Tiny-ImageNet"
mkdir -p "$DATA_DIR"
if [ ! -d "$TINY_DIR" ]; then
  echo "Downloading Tiny-ImageNet..."
  curl -L "$TINYTAR_URL" -o "$TINYTAR_ZIP"
  unzip -q "$TINYTAR_ZIP" -d "$DATA_DIR"
  rm -f "$TINYTAR_ZIP"
fi

python3 imagenet/prepare_tiny_imagenet.py "$TINY_DIR"

echo "\n==> Running benchmarks"
mkdir -p "$RESULTS_DIR"
for model in "${MODELS[@]}"; do
  echo "\n====================================="
  echo "   RUNNING MODEL: ${model}"
  echo "=====================================\n"
  python3 imagenet/main.py "$TINY_DIR" \
    -a "$model" \
    --epochs 1 \
    -b "$BATCH_SIZE" \
    -j "$WORKERS" \
    --max-batches "$MAX_BATCHES" \
    --print-freq "$PRINT_FREQ" \
    --log-csv "${RESULTS_DIR}/metrics_${model}.csv"
done

echo "\n==> Combining CSVs"
python3 - <<'PY'
import glob, pandas as pd
paths = sorted(glob.glob('results/metrics_*.csv'))
if paths:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df.to_csv('results/metrics_all.csv', index=False)
    print('Wrote results/metrics_all.csv with', len(df), 'rows')
else:
    print('No per-model CSVs found in results/')
PY

if [ -f scripts/plot_roofline.py ]; then
  echo "\n==> Generating roofline plot"
  python3 scripts/plot_roofline.py || true
fi

echo "\n====================================="
echo "   ALL DONE! Check results folder."
echo "====================================="

