"""
demo.py  —  Generate two environments, render them side-by-side,
            run the reward model, and display which one the model prefers.

Usage:
    python demo.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
import optax

# ── Path setup (mirrors the rest of the project) ─────────────────────────────
script_path = Path(__file__).resolve()
base_dir    = script_path.parent          # 'uedrlhf'
export_dir  = base_dir / "train" / "model_export"

# ── Config (must match train_pixels.py) ──────────────────────────────────────
IMG_SIZE = 64

# ── Model definition (must match train_pixels.py) ────────────────────────────
class PixelPrefCNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1       = nnx.Conv(3, 32, kernel_size=(5, 5), rngs=rngs)
        self.batch_norm1 = nnx.BatchNorm(32, rngs=rngs)
        self.dropout1    = nnx.Dropout(rate=0.025)
        self.conv2       = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.batch_norm2 = nnx.BatchNorm(64, rngs=rngs)
        self.avg_pool    = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1     = nnx.Linear(16384, 256, rngs=rngs)
        self.dropout2    = nnx.Dropout(rate=0.025)
        self.linear2     = nnx.Linear(256, 1, rngs=rngs)

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        x = self.avg_pool(nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x), rngs=rngs))))
        x = self.avg_pool(nnx.relu(self.batch_norm2(self.conv2(x))))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
        x = self.linear2(x)
        return x.squeeze(-1)


def load_model(export_dir: Path) -> PixelPrefCNN:
    """
    Re-instantiate the model and load weights from the Orbax SavedModel export.
    Orbax saves a TF SavedModel; we load it via tf.saved_model and extract
    the underlying variables back into the Flax NNX model.

    If you exported with orbax, the simplest path for inference in JAX is to
    reload through the TF serving signature and call it as a TF function,
    OR — more conveniently for a JAX demo — re-train the model's parameter
    pytree directly.  Here we use the TF concrete function approach so no
    re-training is needed.
    """
    import tensorflow as tf  # only needed for loading

    saved = tf.saved_model.load(str(export_dir))
    infer = saved.signatures["serving_default"]   # name set in ServingConfig

    # Wrap so we can call with a numpy / jax array
    def predict(img_np: np.ndarray) -> float:
        """img_np: (1, H, W, 3) float32 numpy array → scalar reward score"""
        tf_input = tf.constant(img_np, dtype=tf.float32)
        out = infer(tf_input)                       # dict with one output key
        score = list(out.values())[0].numpy()
        return float(score.squeeze())

    return predict


def render_env(env, state, ax):
    """Render a single JaxNav state onto a matplotlib Axes and return an RGB array."""
    ax.clear()
    env.init_render(ax, state, lidar=False, agent=True, goal=True)
    # Grab the canvas pixels as an RGB array for the model
    ax.figure.canvas.draw()
    w, h = ax.figure.canvas.get_width_height()
    buf  = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
    rgb  = buf.reshape(h, w, 3)
    return rgb


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Resize to IMG_SIZE × IMG_SIZE and normalise to [0, 1]."""
    from PIL import Image
    img = Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0


def main():
    from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

    # ── 1. Generate two fresh environments ───────────────────────────────────
    print("Generating environments …")
    rng = jax.random.PRNGKey(np.random.randint(0, 100_000))   # random seed each run
    rng_a, rng_b = jax.random.split(rng, 2)

    env = JaxNav(num_agents=1)
    _, state_a = env.reset(rng_a)
    _, state_b = env.reset(rng_b)

    # ── 2. Render environments to images ─────────────────────────────────────
    # Use a hidden figure just for rendering; we will display in a nicer layout
    render_fig, render_ax = plt.subplots(1, 1, figsize=(5, 5))
    render_fig.subplots_adjust(0, 0, 1, 1)

    rgb_a = render_env(env, state_a, render_ax)
    rgb_b = render_env(env, state_b, render_ax)
    plt.close(render_fig)

    img_a = preprocess(rgb_a)   # (64, 64, 3)  float32
    img_b = preprocess(rgb_b)

    # ── 3. Load model and score both environments ─────────────────────────────
    print(f"Loading model from {export_dir} …")
    predict = load_model(export_dir)

    score_a = predict(img_a[np.newaxis])   # add batch dim → (1, 64, 64, 3)
    score_b = predict(img_b[np.newaxis])

    # Bradley-Terry preference probability: P(A preferred) = σ(score_A − score_B)
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
    p_a_wins = sigmoid(score_a - score_b)

    preferred   = "A" if score_a > score_b else "B"
    confidence  = max(p_a_wins, 1.0 - p_a_wins) * 100

    print(f"\nScore A : {score_a:+.4f}")
    print(f"Score B : {score_b:+.4f}")
    print(f"Model prefers  : Environment {preferred}  (confidence {confidence:.1f}%)")

    # ── 4. Display ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 6))
    fig.suptitle("Reward Model Demo", fontsize=15, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 1, 5], wspace=0.05)

    ax_a    = fig.add_subplot(gs[0])
    ax_mid  = fig.add_subplot(gs[1])
    ax_b    = fig.add_subplot(gs[2])

    # Environment images (rendered at full resolution by re-drawing onto axes)
    render_fig2, render_ax2 = plt.subplots(1, 1, figsize=(5, 5))
    render_fig2.subplots_adjust(0, 0, 1, 1)

    env.init_render(render_ax2, state_a, lidar=False, agent=True, goal=True)
    # Copy artist objects is awkward; instead just show the numpy arrays
    ax_a.imshow(rgb_a)
    ax_a.set_title(
        f"Environment A\nScore: {score_a:+.4f}",
        fontsize=12,
        color="green" if preferred == "A" else "black",
        fontweight="bold" if preferred == "A" else "normal",
    )
    ax_a.axis("off")

    env.init_render(render_ax2, state_b, lidar=False, agent=True, goal=True)
    ax_b.imshow(rgb_b)
    ax_b.set_title(
        f"Environment B\nScore: {score_b:+.4f}",
        fontsize=12,
        color="green" if preferred == "B" else "black",
        fontweight="bold" if preferred == "B" else "normal",
    )
    ax_b.axis("off")
    plt.close(render_fig2)

    # Middle panel: preference arrow + confidence
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.axis("off")

    arrow_x  = 0.5
    arrow_dx = 0.35 if preferred == "B" else -0.35

    ax_mid.annotate(
        "",
        xy   =(arrow_x + arrow_dx, 0.5),
        xytext=(arrow_x - arrow_dx, 0.5),
        arrowprops=dict(arrowstyle="->", color="green", lw=3),
    )
    ax_mid.text(
        0.5, 0.62,
        f"{confidence:.0f}%",
        ha="center", va="bottom",
        fontsize=11, color="green", fontweight="bold",
        transform=ax_mid.transAxes,
    )
    ax_mid.text(
        0.5, 0.35,
        f"prefers\n{preferred}",
        ha="center", va="top",
        fontsize=10, color="dimgray",
        transform=ax_mid.transAxes,
    )

    # Highlight winner with a coloured border
    winner_ax = ax_a if preferred == "A" else ax_b
    for spine in winner_ax.spines.values():
        spine.set_edgecolor("green")
        spine.set_linewidth(4)
        spine.set_visible(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()