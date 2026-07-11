"""Train a CNN reward model on compiled preference datasets.

Usage:
    python scripts/train/train_pixels.py
"""

import time
import os
from pathlib import Path
from queue import Queue
import threading
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

import wandb
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from datasets import load_from_disk

from uedrlhf.config import BASE_DIR, TRAINING
from uedrlhf.networks import LearnabilityResNet

# Training config from config.yaml
IMG_SIZE = TRAINING.get("img_size", 64) # This is unnecesary No use other then loggind and it does not depend on the compiled dataset so this might have a dangling reference
BATCH_SIZE = TRAINING.get("batch_size", 512)
EPOCHS = TRAINING.get("epochs", 50)
LEARNING_RATE = TRAINING.get("learning_rate", 1e-3)
DATASET_PATH = TRAINING.get("dataset_path", "data/difficulty/compiled_gemini_2_5_flash_20k")
MODEL_TYPE = DATASET_PATH.split("/", 1)[-1] if "/" in DATASET_PATH else DATASET_PATH
RUN_NAME = f"cnn-{MODEL_TYPE}" if TRAINING.get("run_name") == None else TRAINING.get("run_name")  
PROJECT_NAME = TRAINING.get("project_name" , "uedrlhf")
MODE = TRAINING.get("mode", "online")
print(MODEL_TYPE)
x = 14 


@nnx.jit(static_argnames='deterministic')
def loss_fn(model: LearnabilityResNet, data: tuple, deterministic: bool = False):
    img0, img1, preferences = data
    combined_inputs = jnp.concatenate([img0, img1], axis=0)

    scores = model(combined_inputs, deterministic=deterministic)
    scores_input0, scores_input1 = jnp.split(scores, 2, axis=0)

    logits = (scores_input0 - scores_input1)
    batch_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=preferences)
    loss = jnp.mean(batch_loss)
    return loss, logits


@nnx.jit
def train_step(model: LearnabilityResNet, optimizer: nnx.Optimizer, rngs: nnx.Rngs, batch: tuple):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model=model, data=batch)

    optimizer.update(model, grads)

    labels = batch[-1]
    predictions = (logits > 0).astype(jnp.float32)  # >0 means img1 won, <=0 means img0 won
    accuracy = jnp.mean(predictions == labels)

    return loss, accuracy


@nnx.jit
def eval_step(model: LearnabilityResNet, batch: tuple):
    loss, logits = loss_fn(model, batch, deterministic=True)

    labels = batch[-1]
    predictions = (logits > 0).astype(jnp.float32)
    accuracy = jnp.mean(predictions == labels)

    return loss, accuracy


def prefetch_to_gpu(dataloader, queue_size=2):
    queue = Queue(maxsize=queue_size)

    def producer():
        for batch_dict in dataloader:
            batch = (
                jnp.array(batch_dict['env_0_image']),
                jnp.array(batch_dict['env_1_image']),
                jnp.array(batch_dict['preference'])
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
    
    wandb.init(
        project= PROJECT_NAME,
        name=f"cnn-{MODEL_TYPE}",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "adam",
            "dropout": 0.025,
        },
        mode = MODE
    )

    ds = load_from_disk(DATASET_PATH)
    ds_splitted = ds.train_test_split(test_size=0.1)

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

    rngs = nnx.Rngs(0)
    model = LearnabilityResNet(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=LEARNING_RATE), wrt=nnx.Param)


    best_eval_loss = float("inf")

    # ----- TRAINING & EVAL LOOP
    print("Starting training...")
    for epoch in range(EPOCHS):

        epoch_train_losses = []
        epoch_train_accs = []

        # --- TRAIN ---
        model.train()
        for batch_tuple in prefetch_to_gpu(train_loader):
            loss, acc = train_step(model=model, optimizer=optimizer, rngs=rngs, batch=batch_tuple)
            epoch_train_losses.append(loss.item())
            epoch_train_accs.append(acc.item())

        train_loss_avg = np.mean(epoch_train_losses)
        train_acc_avg = np.mean(epoch_train_accs)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f}")

        epoch_eval_losses = []
        epoch_eval_accs = []

        # --- EVAL ---
        model.eval()

        for batch_tuple in prefetch_to_gpu(eval_loader):
            loss, acc = eval_step(model, batch_tuple)
            epoch_eval_losses.append(loss.item())
            epoch_eval_accs.append(acc.item())

        eval_loss_avg = np.mean(epoch_eval_losses)
        eval_acc_avg = np.mean(epoch_eval_accs)
        if eval_loss_avg < best_eval_loss:
            best_eval_loss = eval_loss_avg

        print(f"Epoch {epoch+1}/{EPOCHS} - Eval Loss: {eval_loss_avg:.4f}, Eval Acc: {eval_acc_avg:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss_avg,
            "train/accuracy": train_acc_avg,
            "eval/loss": eval_loss_avg,
            "eval/accuracy": eval_acc_avg,
            "eval/loss_gap": eval_loss_avg - train_loss_avg,  # overfitting signal
            "best_eval_loss": best_eval_loss,
        })

    wandb.finish()
    _, state = nnx.split(model)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = BASE_DIR / "outputs" / "checkpoints" / timestamp / MODEL_TYPE

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(save_path, state)
    checkpointer.wait_until_finished()

    print("Training done")


if __name__ == "__main__":
    main()

