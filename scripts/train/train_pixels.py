"""Train a CNN reward model on compiled preference datasets.

Usage:
    python scripts/train/train_pixels.py
"""

import time
import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from pathlib import Path
from queue import Queue
import threading
import numpy as np
from functools import partial
import torch
from torch.utils.data import DataLoader
import multiprocessing

import wandb
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from datasets import load_from_disk

from uedrlhf.config import BASE_DIR, TRAINING
from uedrlhf.fast_data import print_run_banner
from uedrlhf.networks import LearnabilityResNet

# Training config from config.yaml
IMG_SIZE = TRAINING.get("img_size", 64) # This is unnecesary No use other then loggind and it does not depend on the compiled dataset so this might have a dangling reference
BATCH_SIZE = TRAINING.get("batch_size", 512)
EPOCHS = TRAINING.get("epochs", 50)
LEARNING_RATE = TRAINING.get("learning_rate", 1e-3)
DATASET_PATH = TRAINING.get("dataset_path", "data/difficulty/compiled_gemini_2_5_flash_20k")
if "DATASET_PATH" in os.environ:
    DATASET_PATH = os.environ["DATASET_PATH"]
    
MODEL_TYPE = DATASET_PATH.rstrip("/").split("/")[-1]
RUN_NAME = f"cnn-{MODEL_TYPE}" if TRAINING.get("run_name") == None else TRAINING.get("run_name")
if "RUN_NAME" in os.environ:
    RUN_NAME = os.environ["RUN_NAME"]
    
PROJECT_NAME = TRAINING.get("project_name" , "uedrlhf")
if "PROJECT_NAME" in os.environ:
    PROJECT_NAME = os.environ["PROJECT_NAME"]
    
MODE = TRAINING.get("mode", "online")

# Controls model init, train/test split and shuffling. Runs of different methods
# with the same SEED form a pair for the Wilcoxon signed-rank test.
# Env var overrides config.yaml (needed for Slurm array sweeps).
SEED = int(os.environ.get("SEED", TRAINING.get("seed", 0)))

# Random grid-symmetry augmentation (rot90/flips) on TRAIN batches only.
# Difficulty is invariant to these transforms; eval stays unaugmented.
# Env var overrides config.yaml.
AUGMENT_SYMMETRY = str(os.environ.get(
    "AUGMENT_SYMMETRY", TRAINING.get("augment_symmetry", False)
)).lower() in ("1", "true", "yes")

print(MODEL_TYPE)
print(BATCH_SIZE)


def loss_fn(model: LearnabilityResNet, data: tuple, deterministic: bool = False):
    img0, img1, preferences = data
    combined_inputs = jnp.concatenate([img0, img1], axis=0)

    scores = model(combined_inputs, deterministic=deterministic, return_logits=True)
    scores_input0, scores_input1 = jnp.split(scores, 2, axis=0)

    logits = (scores_input1 - scores_input0)
    batch_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=preferences)
    loss = jnp.mean(batch_loss)
    return loss, logits


def _metrics(logits, labels):
    predictions = (logits > 0).astype(jnp.float32)  # >0 means img1 won, <=0 means img0 won
    accuracy = jnp.mean(predictions == labels)

    # F1 with positive class "img1 harder"; TIE labels (0.5) drop out of tp/fp/fn
    tp = jnp.sum((predictions == 1) & (labels == 1))
    fp = jnp.sum((predictions == 1) & (labels == 0))
    fn = jnp.sum((predictions == 0) & (labels == 1))
    denom = 2 * tp + fp + fn
    f1_score = jnp.where(denom > 0, 2 * tp / jnp.maximum(denom, 1), 0.0)
    return accuracy, f1_score


@nnx.jit
def train_step(model: LearnabilityResNet, optimizer: nnx.Optimizer, rngs: nnx.Rngs, batch: tuple):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model=model, data=batch)

    optimizer.update(model, grads)

    accuracy, f1_score = _metrics(logits, batch[-1])
    return loss, accuracy, f1_score


@nnx.jit
def eval_step(model: LearnabilityResNet, batch: tuple):
    loss, logits = loss_fn(model, batch, deterministic=True)
    accuracy, f1_score = _metrics(logits, batch[-1])
    return loss, accuracy, f1_score


def apply_random_symmetry(imgs, rng):

    out = np.empty_like(imgs)
    ks = rng.integers(0, 8, size=imgs.shape[0])
    for i, k in enumerate(ks):
        x = np.rot90(imgs[i], k % 4)
        if k >= 4:
            x = x[:, ::-1]
        out[i] = x
    return out


def prefetch_to_gpu(dataloader, queue_size=2, augment_rng=None):
    queue = Queue(maxsize=queue_size)

    def producer():
        for batch_dict in dataloader:
            i0 = np.array(batch_dict['env_0_image'], dtype=np.float32)
            i1 = np.array(batch_dict['env_1_image'], dtype=np.float32)

            # Auto-normalize older datasets that are [0, 255] instead of [0, 1]
            if np.max(i0) > 2.0:
                i0 = i0 / 255.0
                i1 = i1 / 255.0

            if augment_rng is not None:
                i0 = apply_random_symmetry(i0, augment_rng)
                i1 = apply_random_symmetry(i1, augment_rng)

            batch = (
                jnp.array(i0),
                jnp.array(i1),
                jnp.array(batch_dict['preference'], dtype=jnp.float32)
            )
            queue.put(batch)

        queue.put(None)  # signal

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    while True:
        batch = queue.get()
        if batch is None:
            break
        yield batch


def main():
    print_run_banner(
        script="train_pixels.py",
        pipeline="dataloader",
        project=PROJECT_NAME,
        run_name=RUN_NAME,
        dataset=DATASET_PATH,
        seed=SEED,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        augment_symmetry=AUGMENT_SYMMETRY,
    )
    wandb.init(
        project= PROJECT_NAME,
        name=RUN_NAME,
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "adam",
            "dropout": 0.025,
            "seed": SEED,
            "augment_symmetry": AUGMENT_SYMMETRY,
            "dataset_path": DATASET_PATH,
        },
        mode = MODE
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds = load_from_disk(DATASET_PATH)
    ds_splitted = ds.train_test_split(test_size=0.1, seed=SEED)

    columns = ['env_0_image', 'env_1_image', 'preference']
    ds_splitted.set_format(type="numpy", columns=columns)

    train_loader = DataLoader(
        ds_splitted["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=4)

    eval_loader = DataLoader(
        ds_splitted["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=4)

    rngs = nnx.Rngs(SEED)
    model = LearnabilityResNet(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=LEARNING_RATE), wrt=nnx.Param)

    augment_rng = np.random.default_rng(SEED) if AUGMENT_SYMMETRY else None
    print(f"Symmetry augmentation: {'ON' if AUGMENT_SYMMETRY else 'off'}")


    best_eval_loss = float("inf")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = wandb.run.id if wandb.run is not None else timestamp
    checkpoint_dir = BASE_DIR / "outputs" / "checkpoints" / f"cnn_{MODEL_TYPE}_{run_id}"
    checkpointer = ocp.StandardCheckpointer()

    # ----- TRAINING & EVAL LOOP
    print("Starting training...")
    for epoch in range(EPOCHS):

        epoch_train_losses = []
        epoch_train_accs = []
        epoch_train_f1s = []

        # --- TRAIN ---
        model.train()
        for batch_tuple in prefetch_to_gpu(train_loader, augment_rng=augment_rng):
            loss, acc, f1 = train_step(model=model, optimizer=optimizer, rngs=rngs, batch=batch_tuple)
            epoch_train_losses.append(loss.item())
            epoch_train_accs.append(acc.item())
            epoch_train_f1s.append(f1.item())

        train_loss_avg = np.mean(epoch_train_losses)
        train_acc_avg = np.mean(epoch_train_accs)
        train_f1_avg = np.mean(epoch_train_f1s)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f}, Train F1: {train_f1_avg:.4f}")

        epoch_eval_losses = []
        epoch_eval_accs = []
        epoch_eval_f1s = []

        # --- EVAL ---
        model.eval()

        for batch_tuple in prefetch_to_gpu(eval_loader):
            loss, acc, f1 = eval_step(model, batch_tuple)
            epoch_eval_losses.append(loss.item())
            epoch_eval_accs.append(acc.item())
            epoch_eval_f1s.append(f1.item())

        eval_loss_avg = np.mean(epoch_eval_losses)
        eval_acc_avg = np.mean(epoch_eval_accs)
        eval_f1_avg = np.mean(epoch_eval_f1s)
        if eval_loss_avg < best_eval_loss:
            best_eval_loss = eval_loss_avg

        print(f"Epoch {epoch+1}/{EPOCHS} - Eval Loss: {eval_loss_avg:.4f}, Eval Acc: {eval_acc_avg:.4f}, Eval F1: {eval_f1_avg:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss_avg,
            "train/accuracy": train_acc_avg,
            "train/f1": train_f1_avg,
            "eval/loss": eval_loss_avg,
            "eval/accuracy": eval_acc_avg,
            "eval/f1": eval_f1_avg,
            "eval/loss_gap": eval_loss_avg - train_loss_avg,  # overfitting signal
            "best_eval_loss": best_eval_loss,
        })

        # Save checkpoint for this epoch
        state = nnx.state(model)
        save_path = checkpoint_dir / f"epoch_{epoch+1}"
        checkpointer.save(save_path.resolve(), state)
        checkpointer.wait_until_finished()
        print(f"Checkpoint saved to {save_path}")

    wandb.finish()

    print("Training done")


if __name__ == "__main__":
    main()

