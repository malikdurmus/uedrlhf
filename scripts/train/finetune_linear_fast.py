import os
import time

import numpy as np
import wandb
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from datasets import load_from_disk

from uedrlhf.config import BASE_DIR, FINETUNING
from uedrlhf.networks import LearnabilityResNet
from uedrlhf.fast_data import (batches, flag_from, print_run_banner, random_symmetry,
                               three_way_split, to_device_uint8, to_float)

PRETRAINED_CHECKPOINT_PATH = os.environ.get(
    "PRETRAINED_CHECKPOINT_PATH",
    FINETUNING.get("pretrained_checkpoint_path", "")
)

IMG_SIZE = FINETUNING.get("img_size", 120)
BATCH_SIZE = FINETUNING.get("batch_size", 64)
EPOCHS = FINETUNING.get("epochs", 30)
LEARNING_RATE = FINETUNING.get("learning_rate", 1e-3)
DATASET_PATH = os.environ.get("DATASET_PATH", FINETUNING.get("dataset_path"))
_model_type = DATASET_PATH.rstrip("/").split("/")[-1]
PROJECT_NAME = os.environ.get("PROJECT_NAME", FINETUNING.get("project_name", "uedrlhf"))
RUN_NAME = os.environ.get("RUN_NAME", f"resnet-finetune-linear-fast-{_model_type}")
SEED = int(os.environ.get("SEED", FINETUNING.get("seed", 0)))
AUGMENT_SYMMETRY = flag_from("AUGMENT_SYMMETRY", FINETUNING, "augment_symmetry")


def loss_fn(model: LearnabilityResNet, batch: tuple, deterministic: bool = False):
    img0, img1, preferences = batch
    combined_inputs = jnp.concatenate([img0, img1], axis=0)
    scores = model(combined_inputs, deterministic=deterministic, return_logits=True)
    scores_input0, scores_input1 = jnp.split(scores, 2, axis=0)
    # preference=1 means img1 is harder; score1 - score0 makes higher score = more difficult
    logits = (scores_input1 - scores_input0)
    batch_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=preferences)
    loss = jnp.mean(batch_loss)
    return loss, logits


def _metrics(logits, labels):
    predictions = (logits > 0).astype(jnp.float32)
    accuracy = jnp.mean(predictions == labels)

    tp = jnp.sum((predictions == 1) & (labels == 1))
    fp = jnp.sum((predictions == 1) & (labels == 0))
    fn = jnp.sum((predictions == 0) & (labels == 1))
    denom = 2 * tp + fp + fn
    f1_score = jnp.where(denom > 0, 2 * tp / jnp.maximum(denom, 1), 0.0)
    return accuracy, f1_score


def _train_body(model, optimizer, data):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, data)
    optimizer.update(model, grads)
    acc, f1 = _metrics(logits, data[2])
    return loss, acc, f1


@nnx.jit
def train_step(model: LearnabilityResNet, optimizer: nnx.Optimizer, batch: tuple):
    img0, img1, prefs = batch
    return _train_body(model, optimizer, (to_float(img0), to_float(img1), prefs))


@nnx.jit
def train_step_aug(model: LearnabilityResNet, optimizer: nnx.Optimizer, batch: tuple, aug_key):
    img0, img1, prefs = batch
    k0, k1 = jax.random.split(aug_key)
    img0 = random_symmetry(k0, img0)
    img1 = random_symmetry(k1, img1)
    return _train_body(model, optimizer, (to_float(img0), to_float(img1), prefs))


@nnx.jit
def eval_step(model: LearnabilityResNet, batch: tuple):
    img0, img1, prefs = batch
    loss, logits = loss_fn(model, (to_float(img0), to_float(img1), prefs), deterministic=True)
    acc, f1 = _metrics(logits, prefs)
    return loss, acc, f1


def get_linear_optimizer():
    """Optax multi-transform that freezes everything except the head."""
    def map_params_to_labels(path_tuple, value):
        path_str = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path_tuple)
        if "head" in path_str:
            return "trainable"
        return "frozen"

    tx = optax.multi_transform(
        {'trainable': optax.adam(LEARNING_RATE), 'frozen': optax.set_to_zero()},
        lambda params: jax.tree_util.tree_map_with_path(map_params_to_labels, params)
    )
    return tx


def main():
    if not PRETRAINED_CHECKPOINT_PATH:
        raise ValueError("CRITICAL: You must specify PRETRAINED_CHECKPOINT_PATH (env or config.yaml)!")

    print_run_banner(
        script="finetune_linear_fast.py",
        mode="linear_probe",
        pipeline="gpu_resident",
        project=PROJECT_NAME,
        run_name=RUN_NAME,
        dataset=DATASET_PATH,
        pretrained_checkpoint=PRETRAINED_CHECKPOINT_PATH,
        seed=SEED,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=f"{LEARNING_RATE} (head), 0.0 (cnn)",
        augment_symmetry=AUGMENT_SYMMETRY,
    )
    wandb.init(
        project=PROJECT_NAME,
        name=RUN_NAME,
        group="linear_probe",
        config={
            "mode": "linear_probe",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "learning_rate": f"{LEARNING_RATE} (head), 0.0 (cnn)",
            "seed": SEED,
            "augment_symmetry": AUGMENT_SYMMETRY,
            "dataset_path": DATASET_PATH,
            "pipeline": "gpu_resident",
        },
    )

    np.random.seed(SEED)

    print(f"Loading dataset from {DATASET_PATH}...")
    ds = load_from_disk(DATASET_PATH)
    train_split, val_split, test_split = three_way_split(ds, SEED)

    t0 = time.perf_counter()
    train_data = to_device_uint8(train_split)
    val_data = to_device_uint8(val_split)
    eval_data = to_device_uint8(test_split)
    jax.block_until_ready(train_data)
    n_train, n_val, n_eval = (train_data[0].shape[0], val_data[0].shape[0],
                              eval_data[0].shape[0])
    print(f"Dataset on device: {n_train} train / {n_val} val / {n_eval} test pairs ({time.perf_counter() - t0:.1f}s)")

    rngs = nnx.Rngs(SEED)
    model = LearnabilityResNet(rngs=rngs)

    print(f"Loading pretrained weights from: {PRETRAINED_CHECKPOINT_PATH}")
    graphdef, abstract_state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(os.path.abspath(PRETRAINED_CHECKPOINT_PATH), abstract_state)
    model = nnx.merge(graphdef, restored_state)
    print("Pretrained weights loaded.")

    optimizer = nnx.Optimizer(model, get_linear_optimizer(), wrt=nnx.Param)

    best_eval_loss = float("inf")
    run_id = wandb.run.id if wandb.run is not None else "local"
    checkpoint_dir = BASE_DIR / "outputs" / "checkpoints" / f"{RUN_NAME}_{run_id}"

    root_key = jax.random.PRNGKey(SEED)
    shuffle_key = jax.random.fold_in(root_key, 0)
    aug_key = jax.random.fold_in(root_key, 1)
    val_order = jnp.arange(n_val)
    eval_order = jnp.arange(n_eval)
    print(f"Symmetry augmentation: {'ON' if AUGMENT_SYMMETRY else 'off'}")

    print(f"Starting Linear Finetune on {n_train} train comparisons, {n_eval} eval comparisons...")

    for epoch in range(EPOCHS):
        shuffle_key, subkey = jax.random.split(shuffle_key)
        order = jax.random.permutation(subkey, n_train)

        model.train()
        tl, ta, tf = [], [], []
        for batch in batches(*train_data, order, BATCH_SIZE):
            if AUGMENT_SYMMETRY:
                aug_key, batch_key = jax.random.split(aug_key)
                loss, acc, f1 = train_step_aug(model, optimizer, batch, batch_key)
            else:
                loss, acc, f1 = train_step(model, optimizer, batch)
            tl.append(loss); ta.append(acc); tf.append(f1)

        train_loss_avg = float(np.mean(jax.device_get(tl)))
        train_acc_avg = float(np.mean(jax.device_get(ta)))
        train_f1_avg = float(np.mean(jax.device_get(tf)))
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f}, Train F1: {train_f1_avg:.4f}")

        model.eval()
        vl, va, vf = [], [], []
        for batch in batches(*val_data, val_order, BATCH_SIZE):
            loss, acc, f1 = eval_step(model, batch)
            vl.append(loss); va.append(acc); vf.append(f1)
        el, ea, ef = [], [], []
        for batch in batches(*eval_data, eval_order, BATCH_SIZE):
            loss, acc, f1 = eval_step(model, batch)
            el.append(loss); ea.append(acc); ef.append(f1)

        val_loss_avg = float(np.mean(jax.device_get(vl)))
        val_acc_avg = float(np.mean(jax.device_get(va)))
        val_f1_avg = float(np.mean(jax.device_get(vf)))
        print(f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc_avg:.4f}, Val F1: {val_f1_avg:.4f}")
        eval_loss_avg = float(np.mean(jax.device_get(el)))
        eval_acc_avg = float(np.mean(jax.device_get(ea)))
        eval_f1_avg = float(np.mean(jax.device_get(ef)))
        if eval_loss_avg < best_eval_loss:
            best_eval_loss = eval_loss_avg

        print(f"Epoch {epoch+1}/{EPOCHS} - Eval Loss: {eval_loss_avg:.4f}, Eval Acc: {eval_acc_avg:.4f}, Eval F1: {eval_f1_avg:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss_avg,
            "train/accuracy": train_acc_avg,
            "train/f1": train_f1_avg,
            "val/loss": val_loss_avg,
            "val/accuracy": val_acc_avg,
            "val/f1": val_f1_avg,
            "eval/loss": eval_loss_avg,
            "eval/accuracy": eval_acc_avg,
            "eval/f1": eval_f1_avg,
        })

        _, state = nnx.split(model)
        save_path = checkpoint_dir / f"epoch_{epoch+1}"
        checkpointer.save(save_path.resolve(), state)
        checkpointer.wait_until_finished()
        print(f"Checkpoint saved to {save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
