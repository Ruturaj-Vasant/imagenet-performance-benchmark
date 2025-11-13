import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# --- Load all result CSVs ---
files = glob.glob("results/metrics_*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# --- Hardware theoretical limits (adjust for each device) ---
peak_compute = 6000  # GFLOPs/s (Apple M3 Pro ≈ 6 TFLOPs)
mem_bandwidth = 200  # GB/s

# --- Prepare Roofline limits ---
x_vals = np.logspace(-3, 3, 200)
compute_roof = np.full_like(x_vals, peak_compute)
memory_roof = mem_bandwidth * x_vals
roof = np.minimum(compute_roof, memory_roof)

# --- Plot setup ---
plt.figure(figsize=(9, 7))
plt.xscale("log")
plt.yscale("log")
plt.plot(x_vals, roof, 'k--', label="Roofline limit")

# --- Plot your data points ---
for _, row in df.iterrows():
    plt.scatter(float(row["arithmetic_intensity"]),
                float(row["train_achieved_gflops_s"]),
                s=80,
                label=f"{row['arch']} ({row['device_type']})")
    plt.text(float(row["arithmetic_intensity"]) * 1.05,
             float(row["train_achieved_gflops_s"]) * 1.05,
             row["arch"], fontsize=8)

# --- Labels and aesthetics ---
plt.xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=12)
plt.ylabel("Achieved Performance (GFLOPs/s)", fontsize=12)
plt.title("Roofline Model — Tiny-ImageNet Benchmarks", fontsize=14)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()