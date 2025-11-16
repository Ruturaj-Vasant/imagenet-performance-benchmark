import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="Generate roofline plots from metrics CSVs")
    ap.add_argument("--csv-glob", nargs="*", default=[
        "results/*git*.csv",
        "results/metrics_squeezenet1_2.csv",
        "results/metrics_squeezenet1_3.csv",
    ], help="Glob(s) for input CSV files")
    ap.add_argument("--which", choices=["train", "val"], default="train",
                    help="Which achieved GFLOPs column to use")
    ap.add_argument("--mode", choices=["device", "model", "both"], default="both",
                    help="Which figures to generate")
    # Peak numbers (GFLOPs/s, GB/s)
    ap.add_argument("--mps-compute", type=float, default=5700.0)
    ap.add_argument("--mps-bw", type=float, default=150.0)
    ap.add_argument("--t4-compute", type=float, default=8100.0)
    ap.add_argument("--t4-bw", type=float, default=320.0)
    ap.add_argument("--cpu-compute", type=float, default=160.0)
    ap.add_argument("--cpu-bw", type=float, default=30.0)
    ap.add_argument("--outdir", default="results", help="Directory to write figures")
    return ap.parse_args()


def load_csvs(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        raise SystemExit("No CSV files matched. Provide --csv-glob paths.")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            pass
    if not frames:
        raise SystemExit("No CSVs could be read.")
    df = pd.concat(frames, ignore_index=True)
    # Coerce numeric columns we rely on
    for col in [
        "arithmetic_intensity",
        "train_achieved_gflops_s", "val_achieved_gflops_s",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalize device_type
    if "device_type" in df.columns:
        df["device_type"] = df["device_type"].str.lower()
    return df


def get_peaks(args, device_type):
    if device_type == "cuda":
        return args.t4_compute, args.t4_bw
    if device_type == "mps":
        return args.mps_compute, args.mps_bw
    # cpu fallback
    return args.cpu_compute, args.cpu_bw


def roofline_curve(ai_vals, compute_peak, bw):
    return np.minimum(compute_peak, bw * ai_vals)


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def plot_device(df, device, args, which_col):
    sub = df[df["device_type"] == device].copy()
    sub = sub.dropna(subset=["arithmetic_intensity", which_col])
    if sub.empty:
        return None

    ai = sub["arithmetic_intensity"].values
    perf = sub[which_col].values
    compute_peak, bw = get_peaks(args, device)

    x_vals = np.logspace(-4, 2, 400)
    roof = roofline_curve(x_vals, compute_peak, bw)

    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x_vals, roof, "k--", label=f"Roofline ({compute_peak:.0f} GF/s, {bw:.0f} GB/s)")

    # Color by model arch
    arches = sorted(sub["arch"].astype(str).unique())
    for arch in arches:
        d = sub[sub["arch"] == arch]
        plt.scatter(d["arithmetic_intensity"], d[which_col], s=60, label=arch)

    plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
    plt.ylabel("Achieved Performance (GFLOPs/s)")
    plt.title(f"Roofline — Device: {device.upper()}")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc="lower right", fontsize=8)
    ensure_outdir(args.outdir)
    out = os.path.join(args.outdir, f"roofline_device_{device}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_model(df, arch, args, which_col):
    sub = df[df["arch"].astype(str) == arch].copy()
    sub = sub.dropna(subset=["arithmetic_intensity", which_col, "device_type"]) 
    if sub.empty:
        return None

    # Determine AI span for curves
    ai_min = max(1e-4, sub["arithmetic_intensity"].min() / 10.0)
    ai_max = max(1.0, sub["arithmetic_intensity"].max() * 10.0)
    x_vals = np.logspace(np.log10(ai_min), np.log10(ai_max), 400)

    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.yscale("log")

    # Plot rooflines for each device present
    for dev in ["mps", "cpu", "cuda"]:
        if dev in sub["device_type"].values:
            peak, bw = get_peaks(args, dev)
            plt.plot(x_vals, roofline_curve(x_vals, peak, bw), label=f"{dev.upper()} roof ({peak:.0f}, {bw:.0f})", linestyle="--")

    # Plot points per device
    markers = {"mps": "o", "cpu": "s", "cuda": "^"}
    for dev in ["mps", "cpu", "cuda"]:
        d = sub[sub["device_type"] == dev]
        if d.empty:
            continue
        plt.scatter(d["arithmetic_intensity"], d[which_col], s=70, marker=markers.get(dev, "o"), label=f"{dev.upper()} points")

    plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
    plt.ylabel("Achieved Performance (GFLOPs/s)")
    plt.title(f"Roofline — Model: {arch}")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc="lower right", fontsize=8)
    ensure_outdir(args.outdir)
    out = os.path.join(args.outdir, f"roofline_model_{arch}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def main():
    args = parse_args()
    df = load_csvs(args.csv_glob)
    which_col = "train_achieved_gflops_s" if args.which == "train" else "val_achieved_gflops_s"

    paths = []
    if args.mode in ("device", "both"):
        for dev in ["mps", "cpu", "cuda"]:
            p = plot_device(df, dev, args, which_col)
            if p:
                print("Wrote:", p)
                paths.append(p)

    if args.mode in ("model", "both"):
        for arch in sorted(df["arch"].astype(str).unique()):
            p = plot_model(df, arch, args, which_col)
            if p:
                print("Wrote:", p)
                paths.append(p)

    if not paths:
        print("No plots generated (check filters and CSV content)")


if __name__ == "__main__":
    main()
