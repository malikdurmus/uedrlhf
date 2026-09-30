"""Statistical analysis of the TVT campaign (train/val/test protocol).

Pulls finished runs from the tvt-* W&B projects and computes, under the
pre-registered reporting rule:

  PRIMARY:    test accuracy/F1 at the epoch of minimum RAW val loss
  ROBUSTNESS: test accuracy/F1 at the epoch of minimum 3-epoch-centered-average
              val loss (damps the BatchNorm eval oscillation)

Analyses:
  1. Finetuning methods (linear_probe vs differential_learning vs
     full_finetuning, paired by seed): Friedman omnibus, then pairwise Wilcoxon
     signed-rank with Holm correction (Demsar 2006).
  2. Augmentation A/B on xland ruleset_difficult thinking-low (paired by seed):
     Wilcoxon signed-rank.

Figures -> outputs/analysis/tvt_{finetune_methods,augmentation_paired}.{png,pdf}
Numbers -> outputs/analysis/tvt_comparison_results.md

(The superseded train/test-only campaign analysis is preserved in
compare_methods_legacy.py.)

Usage:
    python scripts/analysis/compare_methods.py
"""

import argparse
import re
from pathlib import Path

import numpy as np
import wandb
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITY = "malikdurmus-ludwig-maximilian-university-of-munich"
FINETUNE_PROJECT = "tvt2-n-gemini35-finetune-jaxnav-solvable"  # true-domain rerun (canonical); the tvt- predecessor is superseded
XLAND_PROJECT = "tvt-n-gemini35-xminigrid-rulesetdifficult-thinkinglow"
SINGLE_ARM_PROJECTS = ["tvt-n-gemini35-direct-jaxnav",
                       "tvt-n-gemini35-xminigrid-rulesetmedium-thinkinglow"]

METHODS = ["linear", "diff_lr", "full"]
METHOD_LABELS = {"linear": "Linear probe", "diff_lr": "Differential LR", "full": "Full finetune"}
# Tol "vibrant" categorical colors (colorblind-safe, validated)
METHOD_COLORS = {"linear": "#0077BB", "diff_lr": "#EE7733", "full": "#009988"}
BASELINE_COLOR = "#666666"
AUG_COLOR = "#EE7733"
INK = "#333333"


COLLAPSE_TRAIN_ACC = 0.55


def parse_seed_attempt(name):
    """'...-seed7' -> (7, 0); '...-seed7-restart' -> (7, 1); '-restart2' -> (7, 2)."""
    m = re.search(r"seed(\d+)(?:-restart(\d*))?$", name)
    if m is None:
        return None
    attempt = 0 if m.group(2) is None else (int(m.group(2)) if m.group(2) else 1)
    return int(m.group(1)), attempt


def run_metrics(run):
    """Return {rule: (test_acc, test_f1), 'collapsed': bool} or None if empty."""
    hist = run.history(keys=["epoch", "val/loss", "eval/accuracy", "eval/f1",
                             "train/accuracy"], pandas=False)
    if not hist:
        return None
    hist = sorted(hist, key=lambda x: x["epoch"])
    vl = np.array([h["val/loss"] for h in hist])
    # centered 3-epoch moving average (edges use the available window)
    sm = np.array([vl[max(0, i - 1): i + 2].mean() for i in range(len(vl))])
    out = {"collapsed": hist[-1]["train/accuracy"] < COLLAPSE_TRAIN_ACC,
           "epochs": len(hist), "state": run.state, "name": run.name}
    for rule, series in [("raw", vl), ("smoothed", sm)]:
        i = int(np.argmin(series))
        out[rule] = (hist[i]["eval/accuracy"], hist[i]["eval/f1"])
    return out


def consider(bucket, seed, attempt, run):
    """Keep, per seed, the trained run from the latest restart attempt.

    Restart policy: a run whose network collapsed at initialization (train
    accuracy pinned at chance) is rerun under an identical configuration with a
    '-restart' suffix. Both attempts stay in W&B; the analysis uses the latest
    attempt that actually trained, so a collapsed run only counts if no trained
    replacement exists (which would then be visible in the provenance table).
    """
    met = run_metrics(run)
    if met is None:
        return
    met["attempt"] = attempt
    cur = bucket.get(seed)
    if cur is None or (not met["collapsed"], attempt) > (not cur["collapsed"], cur["attempt"]):
        bucket[seed] = met


def provenance(lines, label, buckets, seeds):
    """Record which W&B run backs each seed, so the numbers stay traceable."""
    lines.append(f"\n<details><summary>Runs used ({label})</summary>\n")
    lines.append("| Seed | " + " | ".join(buckets) + " |")
    lines.append("|---" * (len(buckets) + 1) + "|")
    for s in seeds:
        cells = []
        for b in buckets:
            m = buckets[b][s]
            cells.append(m["name"] + (" **COLLAPSED**" if m["collapsed"] else ""))
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines.append("\n</details>\n")


def holm_correct(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min((m - rank) * pvals[idx], 1.0))
        adjusted[idx] = running_max
    return adjusted


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BBBBBB")
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def analyze_finetuning(api, out, lines, project=FINETUNE_PROJECT, tag="tvt"):
    runs = list(api.runs(f"{ENTITY}/{project}"))
    data = {m: {} for m in METHODS}
    for r in runs:
        parsed = parse_seed_attempt(r.name)
        if parsed is None:
            continue
        seed, attempt = parsed
        for m in METHODS:
            if r.name.startswith(f"{m}-"):
                consider(data[m], seed, attempt, r)
    seeds = sorted(set.intersection(*(set(data[m]) for m in METHODS)))
    lines.append(f"## Finetuning methods ({project}, paired seeds {seeds[0]}-{seeds[-1]})\n")
    provenance(lines, "finetuning", data, seeds)

    for rule, rule_label in [("raw", "PRIMARY: test @ min raw val loss"),
                             ("smoothed", "ROBUSTNESS: test @ min 3-epoch-avg val loss")]:
        acc = {m: np.array([data[m][s][rule][0] for s in seeds]) for m in METHODS}
        f1 = {m: np.array([data[m][s][rule][1] for s in seeds]) for m in METHODS}
        lines.append(f"\n### {rule_label}\n")
        lines.append("| Method | test acc mean ± std | acc median | test F1 mean ± std |")
        lines.append("|---|---|---|---|")
        for m in METHODS:
            lines.append(f"| {METHOD_LABELS[m]} | {acc[m].mean():.4f} ± {acc[m].std(ddof=1):.4f} "
                         f"| {np.median(acc[m]):.4f} | {f1[m].mean():.4f} ± {f1[m].std(ddof=1):.4f} |")
        fstat, fp = stats.friedmanchisquare(*(acc[m] for m in METHODS))
        lines.append(f"\nFriedman omnibus (accuracy): chi2={fstat:.2f}, p={fp:.4g}\n")
        pairs = [("linear", "diff_lr"), ("linear", "full"), ("diff_lr", "full")]
        raw_p = [stats.wilcoxon(acc[a], acc[b]).pvalue for a, b in pairs]
        adj = holm_correct(np.array(raw_p))
        lines.append("| Pair | median delta | wins | p (Holm) |")
        lines.append("|---|---|---|---|")
        for (a, b), pa in zip(pairs, adj):
            d = acc[a] - acc[b]
            lines.append(f"| {METHOD_LABELS[a]} vs {METHOD_LABELS[b]} | {np.median(d):+.4f} "
                         f"| {(d > 0).sum()}-{(d < 0).sum()} | {pa:.4f} |")
        if rule == "raw":
            acc_primary, f1_primary = acc, f1

    # ---- figure (primary rule)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    rng = np.random.default_rng(0)
    for ax, vals, title in [(axes[0], acc_primary, "Test accuracy"), (axes[1], f1_primary, "Test F1")]:
        positions = np.arange(len(METHODS))
        ax.boxplot([vals[m] for m in METHODS], positions=positions, widths=0.42,
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
    fig.suptitle("Finetuning methods, 10 paired seeds (test @ min-val-loss epoch)",
                 color=INK, fontsize=11.5, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{tag}_finetune_methods.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_augmentation(api, out, lines, project=XLAND_PROJECT, tag="tvt_augmentation_paired",
                         title="Symmetry augmentation, 10 paired seeds", name_filter=None):
    runs = list(api.runs(f"{ENTITY}/{project}"))
    base, aug = {}, {}
    for r in runs:
        if name_filter is not None and not name_filter(r.name):
            continue
        parsed = parse_seed_attempt(r.name)
        if parsed is None:
            continue
        seed, attempt = parsed
        consider(aug if "-aug-" in r.name else base, seed, attempt, r)
    seeds = sorted(set(base) & set(aug))
    lines.append(f"\n## Augmentation A/B ({project}, paired seeds {seeds[0]}-{seeds[-1]})\n")
    provenance(lines, f"augmentation:{project}", {"baseline": base, "augmented": aug}, seeds)
    lines.append("| Rule | base mean | aug mean | median delta | wins | Wilcoxon p |")
    lines.append("|---|---|---|---|---|---|")
    for rule, rule_label in [("raw", "primary (min raw val loss)"),
                             ("smoothed", "robustness (min smoothed val loss)")]:
        b = np.array([base[s][rule][0] for s in seeds])
        a = np.array([aug[s][rule][0] for s in seeds])
        d = a - b
        p = stats.wilcoxon(a, b).pvalue
        lines.append(f"| {rule_label} | {b.mean():.4f} | {a.mean():.4f} | {np.median(d):+.4f} "
                     f"| {(d > 0).sum()}/{len(seeds)} | {p:.4f} |")
        if rule == "raw":
            b_primary, a_primary = b, a

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for i in range(len(seeds)):
        ax.plot([0, 1], [b_primary[i], a_primary[i]], color="#C9C9C9", linewidth=1.1, zorder=1)
        ax.scatter([0], [b_primary[i]], s=30, color=BASELINE_COLOR, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.scatter([1], [a_primary[i]], s=30, color=AUG_COLOR, zorder=3,
                   edgecolors="white", linewidths=0.8)
    lo = min(b_primary.min(), a_primary.min()) - 0.008
    hi = max(b_primary.max(), a_primary.max()) + 0.008
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Augmented"], color=INK)
    ax.set_ylabel("Test accuracy (min-val-loss epoch)", color=INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11)
    style_axis(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{tag}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_single_arm(api, lines):
    """Summary stats for the projects without a paired comparison."""
    lines.append("\n## Single-arm summaries (no comparison; same selection rules)\n")
    lines.append("| Project | n | rule | test acc mean ± std | acc median | test F1 mean ± std |")
    lines.append("|---|---|---|---|---|---|")
    for proj in SINGLE_ARM_PROJECTS:
        data = {}
        for r in api.runs(f"{ENTITY}/{proj}"):
            parsed = parse_seed_attempt(r.name)
            if parsed is None:
                continue
            consider(data, parsed[0], parsed[1], r)
        seeds = sorted(data)
        for rule in ("raw", "smoothed"):
            acc = np.array([data[s][rule][0] for s in seeds])
            f1 = np.array([data[s][rule][1] for s in seeds])
            lines.append(f"| {proj} | {len(seeds)} | {rule} | {acc.mean():.4f} ± {acc.std(ddof=1):.4f} "
                         f"| {np.median(acc):.4f} | {f1.mean():.4f} ± {f1.std(ddof=1):.4f} |")
        restarts = [data[s]["name"] for s in seeds if data[s]["attempt"] > 0]
        if restarts:
            lines.append(f"\nRestart runs used in {proj}: {', '.join(restarts)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs/analysis")
    parser.add_argument("--finetune-project", default=FINETUNE_PROJECT,
                        help="override the finetuning W&B project (e.g. the tvt2 true-domain rerun)")
    parser.add_argument("--finetune-tag", default="tvt", help="figure filename tag")
    args = parser.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    lines = ["# TVT campaign: method & augmentation comparison\n",
             "Generated by scripts/analysis/compare_methods.py. Selection on the",
             "validation split; reporting on the untouched test split.\n"]
    analyze_finetuning(api, out, lines, project=args.finetune_project, tag=args.finetune_tag)
    analyze_augmentation(api, out, lines)
    # jaxnav aug A/B: baselines and aug runs share the direct-jaxnav project;
    # exclude nothing (ensemble runs live in their own project)
    analyze_augmentation(api, out, lines, project="tvt-n-gemini35-direct-jaxnav",
                         tag="tvt_augmentation_jaxnav_paired",
                         title="Symmetry augmentation (jaxnav), 10 paired seeds")
    analyze_single_arm(api, lines)

    report = out / "tvt_comparison_results.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {report}, {out}/tvt_finetune_methods.(png|pdf), {out}/tvt_augmentation_paired.(png|pdf)")


if __name__ == "__main__":
    main()
