"""
eval.py
-------
Recompute summary metrics and figures from saved CSV results without
re-running inference. Useful for reproducing plots after the fact.

Usage:
    python scripts/eval.py --results artifacts/logs/ua_acd_all_results.csv
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_rel

from src.evaluation.metrics import compute_summary_metrics


METHOD_COLORS = {
    "Vanilla LLM":       "#E74C3C",
    "Standard RAG":      "#F39C12",
    "Static Constraint": "#3498DB",
    "UA-ACD":            "#2ECC71",
    "UA-ACD-Dec-Mod":    "#8E44AD",
}
METHOD_ORDER = ["Vanilla LLM", "Standard RAG", "Static Constraint",
                "UA-ACD", "UA-ACD-Dec-Mod"]


def plot_main_results(df_all: pd.DataFrame, out_dir: str):
    methods = [m for m in METHOD_ORDER if m in df_all["method"].unique()]
    means   = [df_all[df_all["method"] == m]["factuality"].mean() for m in methods]
    stds    = [df_all[df_all["method"] == m]["factuality"].std()  for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, means, yerr=stds, capsize=5,
                  color=[METHOD_COLORS[m] for m in methods],
                  edgecolor="black", linewidth=0.8)
    ax.set_ylabel("FactScore", fontsize=12)
    ax.set_title("FactScore Comparison Across Methods", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure1_main_results.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_efficiency_frontier(df_all: pd.DataFrame, out_dir: str):
    methods = [m for m in METHOD_ORDER if m in df_all["method"].unique()]
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in methods:
        grp = df_all[df_all["method"] == m]
        ax.scatter(grp["ver_calls"].mean(), grp["factuality"].mean(),
                   s=200, color=METHOD_COLORS[m], label=m,
                   edgecolors="black", linewidth=1)
        ax.annotate(m, (grp["ver_calls"].mean(), grp["factuality"].mean()),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Average Verification Calls", fontsize=11)
    ax.set_ylabel("FactScore", fontsize=11)
    ax.set_title("Efficiency-Factuality Trade-off", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure3_pareto_frontier.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="artifacts/logs/ua_acd_all_results.csv")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df_all = pd.read_csv(args.results)

    summary = compute_summary_metrics(df_all)
    print(summary.to_string(index=False))
    summary.to_csv("artifacts/logs/summary_table.csv", index=False)

    plot_main_results(df_all, args.out_dir)
    plot_efficiency_frontier(df_all, args.out_dir)

    # Significance test: UA-ACD vs Static Constraint
    if "UA-ACD" in df_all["method"].unique() and "Static Constraint" in df_all["method"].unique():
        uaacd_f  = df_all[df_all["method"] == "UA-ACD"].set_index("entity")["factuality"]
        static_f = df_all[df_all["method"] == "Static Constraint"].set_index("entity")["factuality"]
        common   = uaacd_f.index.intersection(static_f.index)
        if len(common) >= 5:
            t, p = ttest_rel(uaacd_f[common].values, static_f[common].values)
            delta = uaacd_f[common].mean() - static_f[common].mean()
            print(f"\nUA-ACD vs Static Constraint: delta={delta:+.4f}, t={t:.3f}, p={p:.4f}")

    print(f"\nFigures saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
