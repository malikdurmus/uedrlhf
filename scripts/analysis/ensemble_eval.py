import argparse
import json
import os
from pathlib import Path

import numpy as np
import wandb
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from datasets import load_from_disk

from uedrlhf.config import BASE_DIR
from uedrlhf.networks import LearnabilityResNet
from uedrlhf.fast_data import three_way_split, to_device_uint8, to_float, batches

ENTITY = "malikdurmus-ludwig-maximilian-university-of-munich"
ENVS = {
    "jaxnav": {
        "dataset": "data_v3/jaxnav/difficulty_feedback/compiled_gemini_3_5_flash_len_21990_arr",
        "project": "tvt-ensemble-direct-jaxnav",
        "ckpt_dir": "outputs/checkpoints/ensemble/jaxnav",
        "sets": {"no_aug": "ens-jaxnav-init", "aug": "ens-jaxnav-aug-init"},
    },
    "xmg_difficult": {
        "dataset": "data_v3/xland_minigrid/ruleset_difficult/difficulty_feedback_low/compiled_gemini_3_5_flash_len_21973_arr",
        "project": "tvt-ensemble-xmg-difficult",
        "ckpt_dir": "outputs/checkpoints/ensemble/xmg_difficult",
        "sets": {"aug": "ens-xmg-aug-init"},
    },
    "xmg_medium": {
        "dataset": "data_v3/xland_minigrid/ruleset_medium/difficulty_feedback_low/compiled_gemini_3_5_flash_len_21980_arr",
        "project": "tvt-ensemble-xmg-medium",
        "ckpt_dir": "outputs/checkpoints/ensemble/xmg_medium",
        "sets": {"no_aug": "ens-xmgmed-init", "aug": "ens-xmgmed-aug-init"},
    },
}
SPLIT_SEED = 0  # all members share this split by design


MEMBERS_PER_SET = 5


def resolve_members(project, prefix, k=MEMBERS_PER_SET):
    """Pick the k trained members of a set from the pinned checkpoint index.

    The index contains only finished, non-collapsed runs, so members whose init
    draw collapsed (e.g. ens-xmg-aug-init100, which collapsed on all 3 attempts)
    are excluded automatically and substitutes (init105) are picked up in their
    place. Returns [(run_name, epoch, abs_path)] sorted by name.
    """
    idx_path = BASE_DIR / "outputs" / "checkpoints" / "CHECKPOINT_INDEX.json"
    if not idx_path.exists():
        raise SystemExit(f"missing {idx_path}; run scripts/analysis/pin_checkpoint_epochs.py")
    idx = json.loads(idx_path.read_text())["projects"].get(project)
    if not idx:
        raise SystemExit(f"project {project} absent from CHECKPOINT_INDEX.json "
                         "(runs unfinished? rerun pin_checkpoint_epochs.py)")
   
    by_init = {}
    for name, e in sorted(idx.items()):
        if not name.startswith(prefix):
            continue
        by_init.setdefault(name.split("-restart")[0], (name, e))
    members = [(n, e["selected_epoch_raw"],
                BASE_DIR / "outputs" / "checkpoints" / e["path"])
               for n, e in by_init.values()]
    if len(members) < k:
        print(f"  WARNING: only {len(members)} trained members for {prefix} (wanted {k})")
    return members[:k]


def load_member(path):
    model = LearnabilityResNet(rngs=nnx.Rngs(0))
    graphdef, abstract_state = nnx.split(model)
    restored = ocp.StandardCheckpointer().restore(os.path.abspath(path), abstract_state)
    return nnx.merge(graphdef, restored)


def member_scores(model, test_data, batch_size=256):
    model.eval()
    n = test_data[0].shape[0]
    order = jnp.arange(n)
    s0_parts, s1_parts = [], []

    @nnx.jit
    def score(model, img0, img1):
        combined = jnp.concatenate([to_float(img0), to_float(img1)], axis=0)
        s = model(combined, deterministic=True, return_logits=True)
        return jnp.split(s, 2, axis=0)

    for img0, img1, _ in batches(*test_data, order, batch_size):
        a, b = score(model, img0, img1)
        s0_parts.append(np.asarray(a).ravel())
        s1_parts.append(np.asarray(b).ravel())
    return np.concatenate(s0_parts), np.concatenate(s1_parts)


def acc_f1(s0, s1, prefs):
    pred = (s1 - s0 > 0).astype(np.float32)
    acc = float((pred == prefs).mean())
    tp = int(((pred == 1) & (prefs == 1)).sum())
    fp = int(((pred == 1) & (prefs == 0)).sum())
    fn = int(((pred == 0) & (prefs == 1)).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=ENVS, required=True)
    ap.add_argument("--outdir", default="outputs/analysis")
    args = ap.parse_args()
    cfg = ENVS[args.env]
    api = wandb.Api()

    idx = json.loads((BASE_DIR / "outputs" / "checkpoints" / "CHECKPOINT_INDEX.json").read_text())
    domains = {e.get("input_domain") for e in idx["projects"].get(cfg["project"], {}).values()}
    assert len(domains) == 1, f"mixed/unknown input_domain for {cfg['project']}: {domains}"
    domain = domains.pop()
    os.environ["LEGACY_INVERT_INPUTS"] = "1" if domain == "inverted" else "0"
    print(f"input_domain={domain} (from CHECKPOINT_INDEX.json) -> "
          f"LEGACY_INVERT_INPUTS={os.environ['LEGACY_INVERT_INPUTS']}")

    ds = load_from_disk(str(BASE_DIR / cfg["dataset"]))
    _, _, test_split = three_way_split(ds, SPLIT_SEED)
    test_data = to_device_uint8(test_split)
    prefs = np.asarray(test_data[2])
    print(f"{args.env}: {len(prefs)} shared test pairs (split seed {SPLIT_SEED})")

    lines = [f"# Ensemble-vs-single ablation: {args.env}",
             f"Shared split SEED={SPLIT_SEED}; member epoch = min raw val loss (from W&B).",
             "", "| Set | member accs | single mean ± std | ensemble acc | ensemble F1 |",
             "|---|---|---|---|---|"]
    for set_name, prefix in cfg["sets"].items():
        s0s, s1s, single, used = [], [], [], []
        for name, ep, path in resolve_members(cfg["project"], prefix):
            model = load_member(path)
            s0, s1 = member_scores(model, test_data)
            s0s.append(s0); s1s.append(s1)
            single.append(acc_f1(s0, s1, prefs))
            used.append(f"{name}@{ep}")
            print(f"  {name} epoch_{ep}: acc={single[-1][0]:.4f} f1={single[-1][1]:.4f}")
        ens_acc, ens_f1 = acc_f1(np.mean(s0s, axis=0), np.mean(s1s, axis=0), prefs)
        accs = np.array([a for a, _ in single])
        lines.append(f"| {set_name} | {', '.join(f'{a:.4f}' for a in accs)} "
                     f"| {accs.mean():.4f} ± {accs.std(ddof=1):.4f} | **{ens_acc:.4f}** | {ens_f1:.4f} |")
        lines.append("")
        lines.append(f"Members used ({set_name}): {', '.join(used)}")
        lines.append("")
        print(f"  {set_name}: single {accs.mean():.4f}±{accs.std(ddof=1):.4f} -> ensemble {ens_acc:.4f} (F1 {ens_f1:.4f})")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    report = out / f"ensemble_ablation_{args.env}.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"Saved: {report}")


if __name__ == "__main__":
    main()
