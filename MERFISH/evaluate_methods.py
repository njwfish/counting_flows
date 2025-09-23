#!/usr/bin/env python3
"""
Evaluate methods across multiple cell type counts and plot metric curves.

Inputs:
- --celltypes-list: comma-separated list like 5,10,15
- --outputs-dir: directory containing per-k subfolders like <basename>_5, <basename>_10, ...
- --ground-truth-dir: directory containing ground-truth CSVs named like *_<k>.csv (e.g., S1R1_5.csv)

For each k, this script:
  - loads ground truth CTP (*_<k>.csv) from ground-truth-dir
  - scans outputs-dir/<*>_<k>/ for *.csv predictions
  - computes metrics (JSD scipy, JSD paper, RMSE)
Aggregates metrics over k and plots line charts (x=k, y=metric) per method.
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import evaluation functions
from eval import mean_jsd, rmse, jsd_ct


def load_csv_as_array(csv_path):
    """Load CSV file and return as numpy array and row indices."""
    df = pd.read_csv(csv_path, index_col=0)
    return df.values, df.index.values


def main():
    parser = argparse.ArgumentParser(description="Evaluate methods across cell type counts and plot metrics")
    parser.add_argument("--celltypes-list", required=True, help="Comma-separated list of cell type counts, e.g. 5,10,15")
    parser.add_argument("--outputs-dir", required=True, help="Directory containing per-k subfolders like <basename>_k with method CSVs")
    parser.add_argument("--ground-truth-dir", required=True, help="Directory containing ground-truth CSVs named like *_<k>.csv")
    parser.add_argument("--figs-out", required=True, help="Directory to write metric plots")
    parser.add_argument("--basename", help="Optional basename to match, e.g., S1R1 (filters *_<k> and *_<k>.csv)")
    args = parser.parse_args()

    ks = [int(x) for x in args.celltypes_list.split(',') if x.strip()]
    os.makedirs(args.figs_out, exist_ok=True)

    # metrics_by_method: method -> { 'k': [], 'jsd': [], 'jsd_ct': [], 'rmse': [] }
    metrics_by_method = {}

    for k in ks:
        # locate ground truth file
        pattern = f"{args.basename}_{k}.csv" if args.basename else f"*_{k}.csv"
        gt_candidates = glob.glob(os.path.join(args.ground_truth_dir, pattern))
        if not gt_candidates:
            print(f"[WARN] No ground truth file found for k={k}; skipping")
            continue
        true_csv_path = gt_candidates[0]
        true_array, true_row_indices = load_csv_as_array(true_csv_path)

        # locate outputs subdir for this k
        kdir_pattern = f"{args.basename}_{k}" if args.basename else f"*_{k}"
        k_dir_candidates = [d for d in glob.glob(os.path.join(args.outputs_dir, kdir_pattern)) if os.path.isdir(d)]
        if not k_dir_candidates:
            print(f"[WARN] No outputs subdir found for k={k}; skipping")
            continue
        k_dir = k_dir_candidates[0]
        predicted_files = glob.glob(os.path.join(k_dir, "*.csv"))
        if not predicted_files:
            print(f"[WARN] No predicted CSVs in {k_dir}; skipping")
            continue

        for pred_file in predicted_files:
            method_name = Path(pred_file).stem
            pred_array, pred_row_indices = load_csv_as_array(pred_file)

            # align rows by intersection if needed
            if pred_array.shape != true_array.shape or not np.array_equal(pred_row_indices, true_row_indices):
                common_rows = np.intersect1d(true_row_indices, pred_row_indices)
                if len(common_rows) == 0:
                    print(f"[WARN] No common rows for {method_name} at k={k}; skipping")
                    continue
                true_map = {r: i for i, r in enumerate(true_row_indices)}
                pred_map = {r: i for i, r in enumerate(pred_row_indices)}
                t_idx = [true_map[r] for r in common_rows]
                p_idx = [pred_map[r] for r in common_rows]
                eval_true = true_array[t_idx]
                eval_pred = pred_array[p_idx]
            else:
                eval_true = true_array
                eval_pred = pred_array

            # compute metrics
            _, jsd_mean = mean_jsd(eval_pred, eval_true)
            _, jsd_paper_mean = jsd_ct(eval_pred, eval_true)
            mse_val = rmse(eval_pred, eval_true)

            store = metrics_by_method.setdefault(method_name, {"k": [], "jsd": [], "jsd_ct": [], "rmse": []})
            store["k"].append(k)
            store["jsd"].append(jsd_mean)
            store["jsd_ct"].append(jsd_paper_mean)
            store["rmse"].append(mse_val)

    # plot three graphs: one per metric
    def _plot_metric(metric_key, title, ylabel, fname):
        plt.figure(figsize=(8, 6))
        for method, vals in metrics_by_method.items():
            # order by k
            order = np.argsort(vals["k"]) if len(vals["k"]) else []
            ks_sorted = np.array(vals["k"])[order]
            ys = np.array(vals[metric_key])[order]
            if len(ks_sorted) == 0:
                continue
            plt.plot(ks_sorted, ys, marker='o', label=method)
        plt.title(title)
        plt.xlabel('Number of cell types (k)')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = os.path.join(args.figs_out, fname)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved {title} to {out_path}")
        plt.close()

    _plot_metric("jsd", "JSD (Scipy) vs Cell Types", "Mean JSD", "metric_jsd_scipy.png")
    _plot_metric("jsd_ct", "JSD (Paper) vs Cell Types", "Mean JSD (paper)", "metric_jsd_paper.png")
    _plot_metric("rmse", "RMSE vs Cell Types", "RMSE", "metric_rmse.png")


if __name__ == "__main__":
    main()
