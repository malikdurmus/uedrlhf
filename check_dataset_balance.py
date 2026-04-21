"""
Check class balance of a compiled preference dataset.

Usage:
    python check_dataset_balance.py --dataset data/learnability/compiled_gemini_2_5_flash_dijkstra_15k
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from pathlib import Path

script_path = Path(__file__).resolve()
base_dir    = script_path.parent


def check_balance(dataset_path: str):

    print(f"Loading dataset from: {dataset_path}")
    ds = load_from_disk(dataset_path)

    print(f"Total samples: {len(ds)}\n")

    preferences = np.array(ds["preference"])

    n_env0  = (preferences == 0.0).sum()
    n_env1  = (preferences == 1.0).sum()
    n_tie   = (preferences == 0.5).sum()
    n_total = len(preferences)

    print(f"Class distribution:")
    print(f"  env_0 preferred (0.0): {n_env0:>6}  ({100*n_env0/n_total:.1f}%)")
    print(f"  env_1 preferred (1.0): {n_env1:>6}  ({100*n_env1/n_total:.1f}%)")
    if n_tie > 0:
        print(f"  tie             (0.5): {n_tie:>6}  ({100*n_tie/n_total:.1f}%)")
    print()

    if n_env0 > 0 and n_env1 > 0:
        ratio = min(n_env0, n_env1) / max(n_env0, n_env1)
        print(f"Balance ratio (min/max): {ratio:.3f}  (1.0 = perfect, <0.8 = concerning)")

    # collect split info before plotting so we can reuse it in the text box
    split_lines = []
    if hasattr(ds, "keys"):
        for split in ds.keys():
            split_prefs = np.array(ds[split]["preference"])
            n0 = (split_prefs == 0.0).sum()
            n1 = (split_prefs == 1.0).sum()
            split_lines.append(
                f"Split '{split}': env_0={n0} ({100*n0/len(split_prefs):.1f}%)  "
                f"env_1={n1} ({100*n1/len(split_prefs):.1f}%)"
            )
            print(split_lines[-1])

    # plot
    fig = plt.figure(figsize=(12, 5.5))
    gs  = fig.add_gridspec(2, 2, height_ratios=[4, 1])

    ax0     = fig.add_subplot(gs[0, 0])
    ax1     = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    # bar chart
    labels = ["env_0 (0.0)", "env_1 (1.0)"] + (["tie (0.5)"] if n_tie > 0 else [])
    counts = [n_env0, n_env1]                + ([n_tie]       if n_tie > 0 else [])
    colors = ["steelblue", "coral"]          + (["gray"]      if n_tie > 0 else [])

    ax0.bar(labels, counts, color=colors, edgecolor="white")

    ax0.set_ylabel("Count")
    ax0.set_title("Class distribution")
    ax0.legend()

    # histogram
    ax1.hist(preferences, bins=30, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Preference value")
    ax1.set_ylabel("Count")
    ax1.set_title("Preference value distribution")

    # text box
    summary = (
        f"Total: {n_total}  |  "
        f"env_0: {n_env0} ({100*n_env0/n_total:.1f}%)  |  "
        f"env_1: {n_env1} ({100*n_env1/n_total:.1f}%)  |  "
        f"balance ratio: {min(n_env0, n_env1) / max(n_env0, n_env1):.3f}\n"
    )
    if split_lines:
        summary += "\n".join(split_lines)

    ax_text.text(
        0.0, 0.9, summary,
        transform=ax_text.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    )

    plt.suptitle(Path(dataset_path).name, fontsize=11)
    plt.tight_layout()

    balance_dir = base_dir / "dataset_balances"
    balance_dir.mkdir(parents=True, exist_ok=True)
    plot_name = Path(dataset_path).name + "_balance.png"
    plt.savefig(balance_dir / plot_name, dpi=150)
    print(f"Plot saved to {balance_dir / plot_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/difficulty/compiled_gemini_2_5_flash_20k",
        help="Path to the compiled HuggingFace dataset"
    )
    args = parser.parse_args()
    check_balance(args.dataset)
