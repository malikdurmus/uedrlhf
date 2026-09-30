

import argparse
from pathlib import Path

import numpy as np
import wandb
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITY = "malikdurmus-ludwig-maximilian-university-of-munich"
FINETUNE_PROJECT = "n-gemini35-finetune-jaxnav-solvable"
XLAND_PROJECT = "n-gemini35-xminigrid-rulesetdifficult-thinkinglow"

METHODS = ["linear", "diff_lr", "full"]
METHOD_LABELS = {"linear": "Linear probe", "diff_lr": "Differential LR", "full": "Full finetune"}
# Tol "vibrant" categorical colors (colorblind-safe, validated)
METHOD_COLORS = {"linear": "#0077BB", "diff_lr": "#EE7733", "full": "#009988"}
BASELINE_COLOR = "#666666"
AUG_COLOR = "#EE7733"
INK = "#333333"


def metric_at_best_eval_loss(run):
    """A-priori convention: pick the epoch with min eval/loss; report acc/F1 there."""
    hist = run.history(keys=["epoch", "eval/loss", "eval/accuracy", "eval/f1"], pandas=False)
    best = min(hist, key=lambda x: x["eval/loss"])
    return {"epoch": best["epoch"], "acc": best["eval/accuracy"],
            "f1": best.get("eval/f1"), "loss": best["eval/loss"]}


def holm_correct(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values (same order)."""
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, min(adj, 1.0))
        adjusted[idx] = running_max
    return adjusted


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BBBBBB")
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def analyze_finetuning(api, out, lines):
    runs = list(api.runs(f"{ENTITY}/{FINETUNE_PROJECT}"))
    data = {m: {} for m in METHODS}
    for r in runs:
        seed = int(r.name.rsplit("seed", 1)[-1])
        for m in METHODS:
            if r.name.startswith(f"{m}-"):
                data[m][seed] = metric_at_best_eval_loss(r)
    seeds = sorted(set.intersection(*(set(data[m]) for m in METHODS)))
    lines.append("## Finetuning methods (jaxnav solvable, paired seeds "
                 f"{seeds[0]}-{seeds[-1]})\n")
    lines.append("Metric: eval accuracy / F1 at the epoch of minimum eval loss.\n")

    acc = {m: np.array([data[m][s]["acc"] for s in seeds]) for m in METHODS}
    f1 = {m: np.array([data[m][s]["f1"] for s in seeds]) for m in METHODS}

    lines.append("| Method | acc mean ± std | acc median | F1 mean ± std | F1 median |")
    lines.append("|---|---|---|---|---|")
    for m in METHODS:
        lines.append(f"| {METHOD_LABELS[m]} | {acc[m].mean():.4f} ± {acc[m].std(ddof=1):.4f} "
                     f"| {np.median(acc[m]):.4f} | {f1[m].mean():.4f} ± {f1[m].std(ddof=1):.4f} "
                     f"| {np.median(f1[m]):.4f} |")

    for metric_name, vals in [("accuracy", acc), ("F1", f1)]:
        fstat, fp = stats.friedmanchisquare(*(vals[m] for m in METHODS))
        lines.append(f"\n**Friedman omnibus ({metric_name}):** chi2={fstat:.2f}, p={fp:.4g}")
        pairs = [("linear", "diff_lr"), ("linear", "full"), ("diff_lr", "full")]
        raw = [stats.wilcoxon(vals[a], vals[b]).pvalue for a, b in pairs]
        adj = holm_correct(np.array(raw))
        lines.append(f"\nPairwise Wilcoxon ({metric_name}), Holm-adjusted:\n")
        lines.append("| Pair | median delta | wins | p (raw) | p (Holm) |")
        lines.append("|---|---|---|---|---|")
        for (a, b), pr, pa in zip(pairs, raw, adj):
            d = vals[a] - vals[b]
            lines.append(f"| {METHOD_LABELS[a]} vs {METHOD_LABELS[b]} | {np.median(d):+.4f} "
                         f"| {(d > 0).sum()}-{(d < 0).sum()} | {pr:.4f} | {pa:.4f} |")

    # ---- figure: box + individual seeds, acc and F1 panels
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    rng = np.random.default_rng(0)
    for ax, vals, title in [(axes[0], acc, "Eval accuracy"), (axes[1], f1, "Eval F1")]:
        positions = np.arange(len(METHODS))
        bp = ax.boxplot([vals[m] for m in METHODS], positions=positions, widths=0.42,
                        showfliers=False, medianprops=dict(color=INK, linewidth=1.4),
                        boxprops=dict(color="#999999", linewidth=1.0),
                        whiskerprops=dict(color="#999999", linewidth=1.0),
                        capprops=dict(color="#999999", linewidth=1.0))
        for i, m in enumerate(METHODS):
            x = positions[i] + rng.uniform(-0.10, 0.10, size=len(vals[m]))
            ax.scatter(x, vals[m], s=26, color=METHOD_COLORS[m], zorder=3,
                       edgecolors="white", linewidths=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], color=INK)
        ax.set_title(title, color=INK, fontsize=11)
        style_axis(ax)
    fig.suptitle("Finetuning methods, 10 paired seeds (metric at min-eval-loss epoch)",
                 color=INK, fontsize=11.5, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"finetune_methods.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return acc, f1


def analyze_augmentation(api, out, lines):
    runs = list(api.runs(f"{ENTITY}/{XLAND_PROJECT}"))
    base, aug, rerun7 = {}, {}, None
    for r in runs:
        if r.name.endswith("aug-seed7-rerun"):
            rerun7 = metric_at_best_eval_loss(r)
            continue
        seed = int(r.name.rsplit("seed", 1)[-1])
        (aug if "-aug-" in r.name else base)[seed] = metric_at_best_eval_loss(r)
    seeds = sorted(set(base) & set(aug))
    b = np.array([base[s]["acc"] for s in seeds])
    a_orig = np.array([aug[s]["acc"] for s in seeds])
    a_sub = a_orig.copy()
    if rerun7 is not None and 7 in seeds:
        a_sub[seeds.index(7)] = rerun7["acc"]

    lines.append("\n## Augmentation A/B (xland ruleset_difficult thinking-low, paired seeds)\n")
    lines.append("Metric: eval accuracy at the epoch of minimum eval loss. Restart policy: "
                 "runs that failed to train (chance-level accuracy from a known "
                 "initialization instability) were restarted once with identical "
                 "configuration; this occurred in 1 of 20 runs (aug seed 7).\n")
    analyses = [("with restart policy", a_sub)]
    if SHOW_COLLAPSED:
        analyses.insert(0, ("as-run (collapsed seed 7)", a_orig))
    lines.append("| Analysis | base mean | aug mean | median delta | wins | Wilcoxon p |")
    lines.append("|---|---|---|---|---|---|")
    for label, a in analyses:
        d = a - b
        p = stats.wilcoxon(a, b).pvalue
        lines.append(f"| {label} | {b.mean():.4f} | {a.mean():.4f} | {np.median(d):+.4f} "
                     f"| {(d > 0).sum()}/{len(seeds)} | {p:.4f} |")

    # ---- figure: paired slope plot
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for i, s in enumerate(seeds):
        y0, y1 = b[i], a_sub[i]
        ax.plot([0, 1], [y0, y1], color="#C9C9C9", linewidth=1.1, zorder=1)
        ax.scatter([0], [y0], s=30, color=BASELINE_COLOR, zorder=3,
                   edgecolors="white", linewidths=0.8)
        marker = "o"
        ax.scatter([1], [y1], s=30, color=AUG_COLOR, zorder=3, marker=marker,
                   edgecolors="white", linewidths=0.8)
    # Zoom to the data range of the reported (post-restart-policy) values.
    lo = min(b.min(), a_sub.min()) - 0.008
    hi = max(b.max(), a_sub.max()) + 0.008
    ax.set_ylim(lo, hi)
    if SHOW_COLLAPSED and rerun7 is not None and 7 in seeds:
        i7 = seeds.index(7)
        ax.annotate(f"seed 7 original: {a_orig[i7]:.3f}\n(init collapse, off scale)",
                    xy=(1.02, lo + 0.001), xytext=(0.42, lo + 0.006), fontsize=8,
                    color=INK,
                    arrowprops=dict(arrowstyle="->", color="#999999", linewidth=0.8))
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Augmented"], color=INK)
    ax.set_ylabel("Eval accuracy (min-eval-loss epoch)", color=INK, fontsize=10)
    ax.set_title("Symmetry augmentation, 10 paired seeds", color=INK, fontsize=11)
    style_axis(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"augmentation_paired.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


SHOW_COLLAPSED = False


def main():
    global SHOW_COLLAPSED
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs/analysis")
    parser.add_argument("--show-collapsed", action="store_true",
                        help="Also report/annotate the original collapsed aug-seed7 run "
                             "alongside the restart-policy analysis.")
    args = parser.parse_args()
    SHOW_COLLAPSED = args.show_collapsed
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    lines = ["# Method & augmentation comparison\n",
             "Generated by scripts/analysis/compare_methods.py\n"]
    analyze_finetuning(api, out, lines)
    analyze_augmentation(api, out, lines)

    report = out / "comparison_results.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {report}, {out}/finetune_methods.(png|pdf), {out}/augmentation_paired.(png|pdf)")


if __name__ == "__main__":
    main()
