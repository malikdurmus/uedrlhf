"""Pretrain the Learnability ResNet on the 1M Solvability Dataset.

Usage:
    uv run scripts/train/pretrain_solvability.py
"""

import time
import os
from pathlib import Path
from queue import Queue
import threading
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from datasets import load_from_disk
from torch.utils.data import DataLoader
import wandb
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

from uedrlhf.config import BASE_DIR
from uedrlhf.networks import LearnabilityResNet

# Configuration
# ==========================================
IMG_SIZE = 120
BATCH_SIZE = 128 
EPOCHS = 30
LEARNING_RATE = 1e-3
DATASET_PATH = str(BASE_DIR / "dataset_1M" / "dataset")
RUN_NAME = "resnet-pretrain-solvability_f1"
PROJECT_NAME = "uedrlhf_TRAINING-solvabilitypretrain"
# ==========================================


@nnx.jit
def loss_fn(model: LearnabilityResNet, batch: tuple):
    images, labels = batch
    logits = model(images)
    batch_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
    loss = jnp.mean(batch_loss)
    return loss, logits


@nnx.jit
def train_step(model: LearnabilityResNet, optimizer: nnx.Optimizer, rngs: nnx.Rngs, batch: tuple):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model=model, batch=batch)

    optimizer.update(model, grads)

    labels = batch[1]
    predictions = (logits > 0).astype(jnp.float32)
    accuracy = jnp.mean(predictions == labels)

    tp = jnp.sum((predictions == 1) & (labels == 1))
    fp = jnp.sum((predictions == 1) & (labels == 0))
    fn = jnp.sum((predictions == 0) & (labels == 1))
    denom = 2 * tp + fp + fn
    f1_score = jnp.where(denom > 0, 2 * tp / jnp.maximum(denom, 1), 0.0)

    return loss, accuracy, f1_score


@nnx.jit
def eval_step(model: LearnabilityResNet, batch: tuple):
    images, labels = batch

    logits = model(images, deterministic=True)
    batch_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
    loss = jnp.mean(batch_loss)

    predictions = (logits > 0).astype(jnp.float32)
    accuracy = jnp.mean(predictions == labels)

    tp = jnp.sum((predictions == 1) & (labels == 1))
    fp = jnp.sum((predictions == 1) & (labels == 0))
    fn = jnp.sum((predictions == 0) & (labels == 1))
    denom = 2 * tp + fp + fn
    f1_score = jnp.where(denom > 0, 2 * tp / jnp.maximum(denom, 1), 0.0)

    return loss, accuracy, f1_score


def prefetch_to_gpu(dataloader, queue_size=2):
    queue = Queue(maxsize=queue_size)

    def producer():
        for batch_dict in dataloader:
            
            imgs = batch_dict['image'].numpy() if hasattr(batch_dict['image'], 'numpy') else np.array(batch_dict['image'])
            lbls = batch_dict['solvable'].numpy() if hasattr(batch_dict['solvable'], 'numpy') else np.array(batch_dict['solvable'])
            
            images_fp32 = imgs.astype(np.float32) / 255.0
            labels_fp32 = lbls.astype(np.float32)
            
            batch = (
                jnp.array(images_fp32),
                jnp.array(labels_fp32)
            )
            queue.put(batch)
        queue.put(None)  # signal end 

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    while True:
        batch = queue.get()
        if batch is None:
            break
        yield batch


def main():
    wandb.init(
        project=PROJECT_NAME,
        name=RUN_NAME,
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "adam",
            "model": "LearnabilityResNet"
        }
    )

    print(f"Loading dataset from {DATASET_PATH}...")
    ds = load_from_disk(DATASET_PATH)
    
    print("Splitting dataset into train/test...")
    ds_splitted = ds.train_test_split(test_size=0.1)

    columns = ['image', 'solvable']
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
        shuffle=True,
        drop_last=False,
        num_workers=4)

    rngs = nnx.Rngs(0)
    model = LearnabilityResNet(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=LEARNING_RATE), wrt=nnx.Param)

    best_eval_loss = float("inf")

    # Creating a unique folder corresponding directly to this W&B run ID
    checkpoint_dir = BASE_DIR / "outputs" / "checkpoints" / wandb.run.id
    checkpointer = ocp.StandardCheckpointer()

    # ----- TRAINING & EVAL LOOP
    print(f"Starting training on {len(ds_splitted['train'])} train examples, {len(ds_splitted['test'])} eval examples...")
    
    for epoch in range(EPOCHS): #EPOCHS

        epoch_train_losses = []
        epoch_train_accs = []
        epoch_train_f1s = []

        # --- TRAIN ---
        model.train()
        for batch_tuple in prefetch_to_gpu(train_loader):
            loss, acc, f1 = train_step(model=model, optimizer=optimizer, rngs=rngs, batch=batch_tuple)
            l_val = loss.item()
            a_val = acc.item()
            f1_val = f1.item()
            
            epoch_train_losses.append(l_val)
            epoch_train_accs.append(a_val)
            epoch_train_f1s.append(f1_val)
            
            wandb.log({
                "batch/train_loss": l_val,
                "batch/train_accuracy": a_val,
                "batch/train_f1": f1_val
            })

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
        
        _, state = nnx.split(model)
        save_path = checkpoint_dir / f"epoch_{epoch+1}"
        checkpointer.save(save_path.resolve(), state)
        checkpointer.wait_until_finished()
        print(f"Checkpoint saved to {save_path}")

    wandb.finish()

    print("Training done.")


if __name__ == "__main__":
    main()
