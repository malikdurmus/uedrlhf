import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from datasets import load_from_disk

from uedrlhf.config import BASE_DIR
from uedrlhf.networks import LearnabilityResNet

DATASET = BASE_DIR / "data_v3/xland_minigrid/ruleset_difficult/difficulty_feedback_low/compiled_gemini_3_5_flash_len_21973"
# Finished runs of gemini35-direct-xminigrid-r41313-difficult-low, each at its
# peak eval-accuracy epoch (from wandb).
CHECKPOINTS = {
    "5qgdiit7": 21, "7cawwxny": 25, "og8b6nen": 15, "ljtj99id": 25,
    "36purdt7": 14, "f7hzhoq8": 17, "7cghte1q": 18, "i0z8i1s0": 21,
}
N_PAIRS = 500
OUT_DIR = BASE_DIR / "outputs" / "analysis"

TRANSFORMS = {
    "rot90":  lambda x: np.rot90(x, 1, axes=(1, 2)),
    "rot180": lambda x: np.rot90(x, 2, axes=(1, 2)),
    "rot270": lambda x: np.rot90(x, 3, axes=(1, 2)),
    "fliplr": lambda x: x[:, :, ::-1],
    "flipud": lambda x: x[:, ::-1, :],
}


def load_model(run_id, epoch):
    ckpt = BASE_DIR / "outputs" / "checkpoints" / f"cnn_compiled_gemini_3_5_flash_len_21973_{run_id}" / f"epoch_{epoch}"
    model = LearnabilityResNet(rngs=nnx.Rngs(0))
    graphdef, abstract_state = nnx.split(model)
    model = nnx.merge(graphdef, ocp.StandardCheckpointer().restore(ckpt.resolve(), abstract_state))
    model.eval()
    return model


def score(model, imgs):
    out = []
    for s in range(0, len(imgs), 50):
        out.append(np.asarray(model(jnp.array(np.ascontiguousarray(imgs[s:s+50])),
                                    deterministic=True, return_logits=True)))
    return np.concatenate(out)


def main():
    ds = load_from_disk(str(DATASET))
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(len(ds), size=N_PAIRS, replace=False).tolist())
    ds = ds.select(idx)
    ds.set_format(type="numpy", columns=["env_0_image", "env_1_image", "preference"])
    i0 = np.asarray(ds["env_0_image"], dtype=np.float32)
    i1 = np.asarray(ds["env_1_image"], dtype=np.float32)
    prefs = np.asarray(ds["preference"], dtype=np.float32)
    nt = prefs != 0.5

    names = list(TRANSFORMS)
    flip_rates = {n: [] for n in names}
    accs = {n: [] for n in names}
    base_accs = []

    for run_id, epoch in CHECKPOINTS.items():
        try:
            model = load_model(run_id, epoch)
        except Exception as e:
            print(f"skipping {run_id} (epoch {epoch}): {e}")
            continue
        d_base = score(model, i1) - score(model, i0)
        pred_base = d_base > 0
        acc_b = float((pred_base[nt] == (prefs[nt] == 1)).mean())
        base_accs.append(acc_b)
        for n, T in TRANSFORMS.items():
            d_t = score(model, T(i1)) - score(model, T(i0))
            pred_t = d_t > 0
            flip_rates[n].append(float((pred_t != pred_base).mean()))
            accs[n].append(float((pred_t[nt] == (prefs[nt] == 1)).mean()))
        print(f"{run_id} (epoch {epoch}): base acc {acc_b:.4f}, "
              + ", ".join(f"{n} flip {flip_rates[n][-1]:.3f}" for n in names))

    n_models = len(base_accs)
    print(f"\n=== {n_models} models, {N_PAIRS} pairs each ===")
    print(f"{'transform':>10} | {'flip rate':>16} | {'accuracy':>16} | {'acc drop':>9}")
    base_mu = np.mean(base_accs)
    print(f"{'(none)':>10} | {'-':>16} | {base_mu:.3f} ± {np.std(base_accs):.3f} | {'-':>9}")
    for n in names:
        f_mu, f_sd = np.mean(flip_rates[n]), np.std(flip_rates[n])
        a_mu, a_sd = np.mean(accs[n]), np.std(accs[n])
        print(f"{n:>10} | {f_mu:.3f} ± {f_sd:.3f}    | {a_mu:.3f} ± {a_sd:.3f}   | {base_mu - a_mu:+.3f}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(names))

    ax = axes[0]
    mus = [np.mean(flip_rates[n]) for n in names]
    sds = [np.std(flip_rates[n]) for n in names]
    ax.bar(x, mus, yerr=sds, capsize=4, color="#4878b0", edgecolor="white")
    ax.set_xticks(x, names)
    ax.set_ylabel("fraction of pairwise decisions changed")
    ax.set_title(f"Decision instability under symmetry transforms\n"
                 f"(mean ± std over {n_models} seeds, {N_PAIRS} pairs)")

    ax = axes[1]
    mus = [np.mean(accs[n]) for n in names]
    sds = [np.std(accs[n]) for n in names]
    ax.bar(x, mus, yerr=sds, capsize=4, color="#c0504d", edgecolor="white")
    ax.axhline(base_mu, color="gray", linestyle="--", linewidth=1.4,
               label=f"accuracy on standard images ({base_mu:.3f})")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.0, label="chance")
    ax.set_xticks(x, names)
    ax.set_ylim(0.45, max(base_mu + 0.08, 0.85))
    ax.set_ylabel("pairwise accuracy")
    ax.set_title("Accuracy when both images are transformed\n(ground-truth preference unchanged)")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Trained difficulty CNNs are not orientation-invariant "
                 "(XLand-MiniGrid difficult ruleset, low thinking)", y=1.02, fontsize=11)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"rotation_invariance.{ext}", dpi=200, bbox_inches="tight")
    print(f"\nsaved {OUT_DIR}/rotation_invariance.png / .pdf")


if __name__ == "__main__":
    main()
