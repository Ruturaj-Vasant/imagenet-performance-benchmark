# ImageNet Performance Benchmark

This repo contains a reproducible micro‑benchmark suite to compare deep learning training performance across environments (local Apple M‑series, cloud CPU, and NVIDIA GPUs) on a standardized Tiny‑ImageNet workload. It also includes roofline modeling and a LaTeX report template with figures and analysis.

## Contents
- `imagenet/main.py` — PyTorch ImageNet trainer (short representative runs; FP32 by default).
- `imagenet/prepare_tiny_imagenet.py` — Convert Tiny‑ImageNet into ImageFolder layout.
- `scripts/run_benchmarks.sh` — Run 1‑epoch benchmarks for several models and log to CSV.
- `scripts/gcp_benchmark.sh` — Turnkey pipeline for a fresh GCP VM (data prep + runs + CSVs).
- `scripts/performance_monitor.py` — Collect runtime, FLOPs (via thop), memory/process info.
- `scripts/plot_roofline.py` — Generate device and model roofline figures from CSVs.
- `Report/report.tex` — LaTeX report with tables/figures and ready‑to‑fill narrative.

## Quick Start

### 1) Environment
- Python 3.11+ recommended.
- Install base deps:
  ```bash
  pip install -r imagenet/requirements.txt
  ```
- GPU hosts (CUDA): install torch/torchvision wheels that match your CUDA/driver:
  ```bash
  # Example for CUDA 12.1 (adjust per your system)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

### 2) Data (Tiny‑ImageNet)
```bash
curl -L -o tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip
python imagenet/prepare_tiny_imagenet.py tiny-imagenet-200
```

### 3) Single Run (example)
```bash
python imagenet/main.py tiny-imagenet-200 \
  -a resnet18 --epochs 1 -b 128 -j 8 --print-freq 10 \
  --log-csv results/metrics_resnet18.csv
```

### 4) Batch Script (3 models)
```bash
chmod +x scripts/run_benchmarks.sh
bash scripts/run_benchmarks.sh tiny-imagenet-200 128 8 "--max-batches 50 --log-csv results/metrics_all.csv"
```
This produces per‑model logs under `results/<timestamp>/` and a combined CSV (if provided via `--log-csv`).

### 5) Roofline Plots
Generate per‑device and per‑model rooflines from curated CSVs (defaults to `*git*.csv` and SqueezeNet batch CSVs):
```bash
python scripts/plot_roofline.py --mode both --which train
```
Figures are saved in `results/`:
- `roofline_device_{mps,cpu,cuda}.png`
- `roofline_model_{resnet18,mobilenet_v2,resnet50,squeezenet1_1}.png`

### 6) Build the Report
```bash
pdflatex Report/report.tex && pdflatex Report/report.tex
```
The report includes: experiment design, complexity/AI, results tables, rooflines, and analysis.

## Models and Metrics
- Supported models (via torchvision): `resnet18`, `mobilenet_v2`, `resnet50`, `squeezenet1_1` (and many others).
- Core metrics (CSV): throughput (img/s), avg batch/data time, achieved GFLOPs/s, TFLOP/s (if FLOPs available), eval accuracy, process RSS, device metadata.

## Notes per Environment
- Apple M‑series (MPS): no pinned memory and no GPU util/VRAM counters; CSV marks these as N/A.
- CUDA (NVIDIA): consider enabling AMP experiments (not on by default) if you add autocast+GradScaler.
- CPU: expect lower throughput; keep `-j` moderate and consider `--max-batches` for quick runs.

## Repro Tips
- Keep batch sizes identical across devices when stable; if instability occurs, choose the smallest stable batch across runs and document it.
- Run with `--print-freq 1` for per‑batch throughput during quick validation.
- For rooflines, default peaks are: M3 (5.7 TFLOPs / 150 GB/s), T4 (8.1 TFLOPs / 320 GB/s), CPU (160 GFLOPs / 30 GB/s). Override via CLI flags if needed.

## Repo Conventions
- `results/` is git‑ignored to avoid large artifacts. Curated CSVs for the report are kept as `*_git.csv` and versioned.
- The LaTeX report uses UTF‑8; avoid smart punctuation if your TeX engine errors.

## GCP One‑liner
```bash
chmod +x scripts/gcp_benchmark.sh
./scripts/gcp_benchmark.sh
```
Prepares data, runs selected models, and writes CSVs to `results/<timestamp>/` (optionally auto‑commit with `GIT_PUSH=1`, if credentials are configured).

