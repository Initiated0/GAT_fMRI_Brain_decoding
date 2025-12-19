#!/usr/bin/env python3
"""
Compute subject-wise FC matrices, threshold + binarize adjacency, and save results + plots.
HPC-safe (headless): uses matplotlib Agg backend; saves figures to disk (no plt.show()).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend for HPC
import matplotlib.pyplot as plt


# =========================
# Paths / Config
# =========================
FMRI_BOLD_DATA_DIR = Path("/work/nayeem/Huth_deepfMRI/processed_data_transformed_atlas_float32/")
OUT_DIR = Path("/work/nayeem/Huth_deepfMRI/results/GAT_mul_subj/")

SUB_IDS = ["UTS01", "UTS03", "UTS07"]

# You used 0.50 in the loop; keep it as default for the saved outputs.
THRESHOLDS = [0.50]

# If you also want 0.25 outputs, do: THRESHOLDS = [0.25, 0.50]


# =========================
# Utilities
# =========================
def ensure_dirs(base_out: Path) -> dict:
    base_out.mkdir(parents=True, exist_ok=True)
    dirs = {
        "npz": base_out / "npz",
        "figs": base_out / "figs",
        "logs": base_out / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_zscore(X: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score with numerical safety for near-constant signals.
    """
    mu = X.mean(axis=axis, keepdims=True)
    sd = X.std(axis=axis, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)  # avoid divide-by-zero
    return (X - mu) / sd


def compute_fc_matrix(roi_time_series: np.ndarray) -> np.ndarray:
    """
    roi_time_series: shape (TRs, ROIs)
    returns: FC matrix (ROIs x ROIs) using Pearson correlation
    """
    # ROIs x Time
    X = roi_time_series.T.astype(np.float32, copy=False)

    # Z-score along time dimension
    X = safe_zscore(X, axis=1)

    # Correlation matrix
    fc = (X @ X.T) / X.shape[1]

    # Numerical clip
    fc = np.clip(fc, -1.0, 1.0)

    return fc.astype(np.float32, copy=False)


def threshold_and_binarize(fc_matrix: np.ndarray, threshold: float):
    """
    Applies |fc| >= threshold, zeros diagonal, returns:
    - fc_filtered (float32)
    - adjacency (uint8)
    """
    fc = fc_matrix.copy()
    np.fill_diagonal(fc, 0.0)

    fc_filtered = np.where(np.abs(fc) >= threshold, fc, 0.0).astype(np.float32, copy=False)
    adjacency = (fc_filtered != 0.0).astype(np.uint8, copy=False)

    return fc_filtered, adjacency


# =========================
# Plotting (save-to-disk)
# =========================
def plot_fc_heatmap(fc_matrix: np.ndarray, out_png: Path, title: str):
    plt.figure(figsize=(10, 8))
    im = plt.imshow(fc_matrix, vmin=-1, vmax=1, aspect="equal")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("ROI Index")
    plt.ylabel("ROI Index")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_fc_histogram(fc_matrix: np.ndarray, out_png: Path, title: str):
    # upper triangle without diagonal
    vals = fc_matrix[np.triu_indices_from(fc_matrix, k=1)]
    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=80, density=True)
    plt.title(title)
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_adjacency_sparsity(adj_matrix: np.ndarray, out_png: Path, title: str):
    plt.figure(figsize=(7, 6))
    plt.spy(adj_matrix, markersize=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================
# Main
# =========================
def main():
    dirs = ensure_dirs(OUT_DIR)

    summary_rows = []

    for sub_id in SUB_IDS:
        in_path = FMRI_BOLD_DATA_DIR / f"{sub_id}_combined.npy"
        if not in_path.exists():
            print(f"[WARN] Missing file: {in_path} (skipping)")
            continue

        print(f"\n[INFO] Loading {in_path}")
        fmri_data = np.load(in_path)  # expected shape (TRs, ROIs)

        print(f"[INFO] Computing FC for {sub_id} ...")
        fc_matrix = compute_fc_matrix(fmri_data)

        # Save raw FC once per subject (optional but useful)
        raw_fc_npz = dirs["npz"] / f"{sub_id}_FC_raw.npz"
        np.savez_compressed(raw_fc_npz, fc_matrix=fc_matrix)
        print(f"[SAVE] {raw_fc_npz}")

        # Plots for raw FC
        plot_fc_heatmap(
            fc_matrix,
            dirs["figs"] / f"{sub_id}_FC_raw_heatmap.png",
            title=f"Functional Connectivity (Pearson) - {sub_id} (raw)"
        )
        plot_fc_histogram(
            fc_matrix,
            dirs["figs"] / f"{sub_id}_FC_raw_hist.png",
            title=f"Distribution of FC Values - {sub_id} (raw)"
        )

        # Threshold-specific outputs
        for thr in THRESHOLDS:
            fc_filtered, adjacency = threshold_and_binarize(fc_matrix, thr)

            # Edge count: adjacency is symmetric; adjacency.sum() counts both directions
            edges_directed = int(adjacency.sum())
            edges_undirected = int(edges_directed // 2)

            out_npz = dirs["npz"] / f"{sub_id}_FC_thr{thr:.2f}.npz"
            np.savez_compressed(
                out_npz,
                fc_filtered=fc_filtered,
                adjacency=adjacency,
                threshold=np.array(thr, dtype=np.float32),
            )
            print(f"[SAVE] {out_npz}  | edges(directed)={edges_directed}, edges(undirected)~={edges_undirected}")

            # Plots for thresholded adjacency
            plot_adjacency_sparsity(
                adjacency,
                dirs["figs"] / f"{sub_id}_adj_thr{thr:.2f}_spy.png",
                title=f"Adjacency Sparsity - {sub_id} (|r| ≥ {thr:.2f})"
            )

            summary_rows.append({
                "sub_id": sub_id,
                "input_file": str(in_path),
                "TRs": int(fmri_data.shape[0]),
                "ROIs": int(fmri_data.shape[1]),
                "threshold": float(thr),
                "edges_directed": edges_directed,
                "edges_undirected_approx": edges_undirected,
                "raw_fc_npz": str(raw_fc_npz),
                "thr_npz": str(out_npz),
            })

    # Save summary CSV
    summary_csv = OUT_DIR / "fc_threshold_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\n[SAVE] Summary: {summary_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()
