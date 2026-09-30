"""Pin the selection-rule epoch of every canonical run to a static index file.

Rationale: checkpoint dirs often contain BOTH the raw-rule and smoothed-rule
epochs, and nothing on disk says which is which (the raw pick is sometimes the
lower and sometimes the higher epoch number). Consumers — especially the SFL
repo loading deployed ensembles — must not re-derive this from W&B at use time:
that adds a network dependency and risks two consumers resolving differently.

Writes outputs/checkpoints/CHECKPOINT_INDEX.json (machine-readable, the
authority) and prints a summary. Re-run after new runs land; it is idempotent.

Usage:
    python scripts/analysis/pin_checkpoint_epochs.py
"""
import json
from pathlib import Path

import numpy as np
import wandb

from uedrlhf.config import BASE_DIR

ENTITY = "malikdurmus-ludwig-maximilian-university-of-munich"
CKPT = BASE_DIR / "outputs" / "checkpoints"
# project -> directory holding <run_name>/epoch_<N>
LOCATIONS = {
    "tvt-n-gemini35-direct-jaxnav": "tvt_best/tvt-n-gemini35-direct-jaxnav",
    "tvt-n-gemini35-finetune-jaxnav-solvable": "tvt_best/tvt-n-gemini35-finetune-jaxnav-solvable",
    "tvt-n-gemini35-xminigrid-rulesetdifficult-thinkinglow": "tvt_best/tvt-n-gemini35-xminigrid-rulesetdifficult-thinkinglow",
    "tvt-n-gemini35-xminigrid-rulesetmedium-thinkinglow": "tvt_best/tvt-n-gemini35-xminigrid-rulesetmedium-thinkinglow",
    "tvt-ensemble-direct-jaxnav": "ensemble/jaxnav",
    "tvt-ensemble-xmg-difficult": "ensemble/xmg_difficult",
    "tvt-ensemble-xmg-medium": "ensemble/xmg_medium",
    # Post-loader-fix (TRUE-domain) projects — see INPUT_DOMAIN below.
    "tvt2-n-gemini35-finetune-jaxnav-solvable": "tvt_best/tvt2-n-gemini35-finetune-jaxnav-solvable",
    "tvt2-n-gemini35-direct-jaxnav-solvable": "tvt_best/tvt2-n-gemini35-direct-jaxnav-solvable",
    "tvt2-n-gemma4-direct-blockedunlock": "tvt_best/tvt2-n-gemma4-direct-blockedunlock",
}

# Which colour domain a checkpoint expects at inference. Everything trained
# before the 2026-08-29 loader fix learned COLOR-INVERTED images and must be
# served through `fast_data._legacy_invert` (the SFL loop's
# `invert_input_domain`); anything trained after expects TRUE images and must
# NOT be inverted. Consumers should branch on this field, never on a hardcoded
# assumption. See EXPERIMENTS_STATE.MD section 10.
def input_domain(project):
    return "true" if project.startswith("tvt2-") else "inverted"
COLLAPSE_TRAIN_ACC = 0.55


def main():
    api = wandb.Api()
    index, missing_dirs, collapsed = {}, [], []
    for project, subdir in LOCATIONS.items():
        entries = {}
        try:
            runs = list(api.runs(f"{ENTITY}/{project}"))
        except Exception as e:
            print(f"# skip {project}: {e}")
            continue
        for r in runs:
            if r.state != "finished":
                continue
            h = r.history(keys=["epoch", "val/loss", "eval/accuracy", "eval/f1",
                                "train/accuracy"], pandas=False)
            if not h:
                continue
            h = sorted(h, key=lambda x: x["epoch"])
            if h[-1]["train/accuracy"] < COLLAPSE_TRAIN_ACC:
                collapsed.append(f"{project}/{r.name}")
                continue
            vl = np.array([x["val/loss"] for x in h])
            sm = np.array([vl[max(0, i - 1): i + 2].mean() for i in range(len(vl))])
            i_raw, i_sm = int(np.argmin(vl)), int(np.argmin(sm))
            raw_epoch = int(h[i_raw]["epoch"])
            rel = f"{subdir}/{r.name}/epoch_{raw_epoch}"
            on_disk = (CKPT / rel).is_dir()
            if not on_disk:
                missing_dirs.append(rel)
            entries[r.name] = {
                "wandb_run_id": r.id,
                "input_domain": input_domain(project),
                "selected_epoch_raw": raw_epoch,          # PRIMARY rule
                "selected_epoch_smoothed": int(h[i_sm]["epoch"]),
                "val_loss_at_raw": round(float(vl[i_raw]), 6),
                "test_accuracy_at_raw": round(float(h[i_raw]["eval/accuracy"]), 6),
                "test_f1_at_raw": round(float(h[i_raw]["eval/f1"]), 6),
                "path": rel,
                "path_exists": on_disk,
            }
        if entries:
            index[project] = dict(sorted(entries.items()))

    out = {
        "_README": ("Authoritative pin of the selection-rule epoch per run. "
                    "'input_domain' says which colour encoding the checkpoint "
                    "expects: 'inverted' checkpoints MUST be served through the "
                    "legacy inversion (0 if x==0 else 256-x); 'true' checkpoints "
                    "must NOT be inverted. See EXPERIMENTS_STATE.MD section 10. "
                    "PRIMARY rule = selected_epoch_raw (min raw val/loss); "
                    "'path' is relative to outputs/checkpoints/ and points at the "
                    "PRIMARY checkpoint. Do NOT re-derive from W&B — use this file. "
                    "Regenerate with scripts/analysis/pin_checkpoint_epochs.py."),
        "entity": ENTITY,
        "projects": index,
    }
    path = CKPT / "CHECKPOINT_INDEX.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    n = sum(len(v) for v in index.values())
    print(f"Pinned {n} runs across {len(index)} projects -> {path}")
    for p, v in index.items():
        print(f"  {p}: {len(v)}")
    if collapsed:
        print(f"Excluded {len(collapsed)} collapsed run(s): {', '.join(collapsed)}")
    if missing_dirs:
        print(f"WARNING: {len(missing_dirs)} pinned checkpoint dir(s) not on disk:")
        for m in missing_dirs:
            print(f"  MISSING {m}")


if __name__ == "__main__":
    main()
