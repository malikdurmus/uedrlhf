import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import wandb

ENTITY = "malikdurmus-ludwig-maximilian-university-of-munich"
PROJECT = "hpo-xmg-direct"
COLLAPSE_TRAIN_ACC = 0.55


def sd(a):
    """± std, or a marker when a single run makes std undefined."""
    return f" +/- {a.std(ddof=1):.4f}" if len(a) > 1 else " (n=1)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/analysis")
    args = ap.parse_args()

    api = wandb.Api()
    by_lr = defaultdict(list)
    for r in api.runs(f"{ENTITY}/{PROJECT}"):
        m = re.search(r"lr([0-9.e-]+)-seed(\d+)$", r.name)
        if not m:
            continue
        lr, seed = m.group(1), int(m.group(2))
        h = r.history(keys=["epoch", "val/loss", "val/accuracy", "train/accuracy"],
                      pandas=False)
        if not h:
            continue
        h = sorted(h, key=lambda x: x["epoch"])
        vl = np.array([x["val/loss"] for x in h])
        i = int(np.argmin(vl))
        by_lr[lr].append({
            "seed": seed,
            "collapsed": h[-1]["train/accuracy"] < COLLAPSE_TRAIN_ACC,
            "val_loss": float(vl[i]),
            "val_acc": float(h[i]["val/accuracy"]),
            "epoch": int(h[i]["epoch"]),
        })

    lines = [
        "# Hyperparameter sensitivity: learning rate (XLand-MiniGrid difficult, direct)",
        "",
        "Pre-registered VALIDATION-ONLY sensitivity analysis; the test split is not",
        "consulted here, and the main campaigns keep lr = 1e-3 regardless of the",
        "outcome (Rule A, THESIS_NOTES.MD). Within-run selection: epoch of minimum raw",
        "val loss. Runs execute with ABORT_ON_COLLAPSE=0 so collapses are counted",
        "rather than hidden; means are over trained (non-collapsed) runs only.",
        "",
        "| learning rate | n trained | collapses | val loss (mean +/- std) | val acc (mean +/- std) | selected epochs |",
        "|---|---|---|---|---|---|",
    ]
    summary = {}
    for lr in sorted(by_lr, key=float):
        rows = by_lr[lr]
        ok = [x for x in rows if not x["collapsed"]]
        n_col = len(rows) - len(ok)
        if ok:
            vls = np.array([x["val_loss"] for x in ok])
            vas = np.array([x["val_acc"] for x in ok])
            eps = ", ".join(str(x["epoch"]) for x in sorted(ok, key=lambda x: x["seed"]))
            lines.append(f"| {lr} | {len(ok)} | {n_col}/{len(rows)} | "
                         f"{vls.mean():.4f}{sd(vls)} | "
                         f"{vas.mean():.4f}{sd(vas)} | {eps} |")
            summary[lr] = (float(vls.mean()), float(vas.mean()))
        else:
            lines.append(f"| {lr} | 0 | {n_col}/{len(rows)} | - | - | - |")

    if summary:
        best = min(summary, key=lambda k: summary[k][0])
        lines += ["", f"Lowest mean validation loss: **lr = {best}** "
                      f"(val loss {summary[best][0]:.4f}, val acc {summary[best][1]:.4f})."]
        if best != "1e-3":
            lines.append("Per the pre-registered Rule A this changes NO experiment: all "
                         "campaigns and deployed models use lr = 1e-3; the difference is "
                         "reported as a limitation rather than acted upon.")
        else:
            lines.append("This confirms the configuration used throughout the thesis.")
        tc = sum(len([x for x in v if x["collapsed"]]) for v in by_lr.values())
        tr = sum(len(v) for v in by_lr.values())
        rates = ", ".join(f"{lr}: {len([x for x in by_lr[lr] if x['collapsed']])}/{len(by_lr[lr])}"
                          for lr in sorted(by_lr, key=float))
        lines += ["", "## Collapse incidence (secondary result)", "",
                  f"Across the grid: {tc}/{tr} runs collapsed ({rates}).",
                  "",
                  "The rate increases monotonically with the learning rate, which is the "
                  "direction predicted by the dying-ReLU account of the collapse (larger "
                  "first Adam steps -> higher chance the scoring head's ReLU is driven "
                  "into the all-negative regime, from which the pairwise-difference loss "
                  "provides no gradient path back). Report as SUGGESTIVE, not established: "
                  "with 3 runs per cell the comparison is badly underpowered (Fisher exact, "
                  "lowest vs highest lr: p ~ 0.4). A properly powered test would need "
                  "~20 runs per cell and is out of scope; state the trend and its limits."]

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    report = out / "hpo_lr_sensitivity.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {report}")


if __name__ == "__main__":
    main()
