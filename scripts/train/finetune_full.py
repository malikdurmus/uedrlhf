"""Fine-tune LearnabilityResNet fully.

All components of the CNN are trained, with a reduced learning rate (1e-5).
"""

import time
import os
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from datasets import load_from_disk
from torch.utils.data import DataLoader
import wandb
from queue import Queue
import threading
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

from uedrlhf.config import BASE_DIR, TRAINING
from uedrlhf.networks import LearnabilityResNet

# ==========================================
# IMPORTANT: DEFINE THE CHECKPOINT PATH HERE
PRETRAINED_CHECKPOINT_PATH = "/home/d/durmusy/Desktop/GIT/new/uedrlhf/outputs/checkpoints/lbec1lyl/epoch_14" 
# Example: str(BASE_DIR / "outputs" / "checkpoints" / "123456" / "epoch_30")
# ==========================================

IMG_SIZE = 120
BATCH_SIZE = 64
EPOCHS = 30
DATASET_PATH = "/home/d/durmusy/Desktop/GIT/new/uedrlhf/data_v2/difficulty/compiled_gemini_2_5_flash_len_7700" # TODO change this to config
RUN_NAME = "resnet-finetune-full"
PROJECT_NAME = "uedrlhf"


@nnx.jit
def loss_fn(model: LearnabilityResNet, batch: tuple):
    img0, img1, preferences = batch
    combined_inputs = jnp.concatenate([img0, img1], axis=0)
    scores = model(combined_inputs)
    scores_input0, scores_input1 = jnp.split(scores, 2, axis=0)
    logits = (scores_input0 - scores_input1)
    batch_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=preferences)
    loss = jnp.mean(batch_loss)
    return loss, logits

@nnx.jit
def train_step(model: LearnabilityResNet, optimizer: nnx.Optimizer, batch: tuple):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model=model, batch=batch)
    optimizer.update(model, grads)
    preferences = batch[2]
    predictions = (logits > 0).astype(jnp.float32)
    #acc_mask = (preferences != 0.5)
    #accuracy = jnp.sum((predictions == preferences) * acc_mask) / jnp.maximum(1, jnp.sum(acc_mask))
    accuracy = jnp.mean(predictions == preferences)
    return loss, accuracy

@nnx.jit
def eval_step(model: LearnabilityResNet, batch: tuple):
    loss, logits = loss_fn(model, batch)
    preferences = batch[2]
    predictions = (logits > 0).astype(jnp.float32)
    #acc_mask = (preferences != 0.5)
    accuracy = jnp.mean(predictions == preferences)
    return loss, accuracy

def prefetch_to_gpu(dataloader, queue_size=2):
    queue = Queue(maxsize=queue_size)
    def producer():
        for batch_dict in dataloader:
            i0 = batch_dict['env_0_image'].numpy() if hasattr(batch_dict['env_0_image'], 'numpy') else np.array(batch_dict['env_0_image'])
            i1 = batch_dict['env_1_image'].numpy() if hasattr(batch_dict['env_1_image'], 'numpy') else np.array(batch_dict['env_1_image'])
            pref = batch_dict['preference'].numpy() if hasattr(batch_dict['preference'], 'numpy') else np.array(batch_dict['preference'])

            i0_fp32 = i0.astype(np.float32) / 255.0
            i1_fp32 = i1.astype(np.float32) / 255.0
            p_fp32 = pref.astype(np.float32)

            batch = (jnp.array(i0_fp32), jnp.array(i1_fp32), jnp.array(p_fp32))
            queue.put(batch)
        queue.put(None)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    while True:
        batch = queue.get()
        if batch is None: break
        yield batch

def main():
    if not PRETRAINED_CHECKPOINT_PATH:
        raise ValueError("CRITICAL: You must specify the PRETRAINED_CHECKPOINT_PATH at the top of the script!")

    wandb.init(
        project=PROJECT_NAME,
        name=RUN_NAME,
        config={
            "mode": "full_finetuning",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "learning_rate": "1e-5 (all)",
        }
    )

    print(f"Loading dataset from {DATASET_PATH}...")
    ds = load_from_disk(DATASET_PATH)
    ds_splitted = ds.train_test_split(test_size=0.1)
    ds_splitted.set_format(type="numpy", columns=["env_0_image", "env_1_image", "preference"])

    train_loader = DataLoader(ds_splitted["train"], batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)
    eval_loader = DataLoader(ds_splitted["test"], batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

    rngs = nnx.Rngs(0)
    model = LearnabilityResNet(rngs=rngs)

    # 1. LOAD PRETRAINED WEIGHTS
    print(f"Loading pretrained weights from: {PRETRAINED_CHECKPOINT_PATH}")
    
    # We load the exact abstract state shape to tell Orbax what to expect
    graphdef, abstract_state = nnx.split(model)
    
    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(PRETRAINED_CHECKPOINT_PATH, abstract_state)
    
    # Re-merge to set the model to eval mode (disables dropout)
    model = nnx.merge(graphdef, restored_state)
    
    # Split one final time to return the pure JAX components    

    print("Pretrained weights loaded.")


    print(f"Loading pretrained weights from: {PRETRAINED_CHECKPOINT_PATH}")

    # 2. CREATE STANDARD OPTIMIZER (Micro LR)
    optimizer = nnx.Optimizer(model, optax.adam(1e-5), wrt=nnx.Param)

    best_eval_loss = float("inf")
    checkpoint_dir = BASE_DIR / "outputs" / "checkpoints" / f"finetune_full_checkpoints" / wandb.run.id

    for epoch in range(EPOCHS):
        epoch_train_losses = []
        epoch_train_accs = []
        model.train()
        for batch_idx, batch_tuple in enumerate(prefetch_to_gpu(train_loader)):
            loss, acc = train_step(model=model, optimizer=optimizer, batch=batch_tuple)
            l_val, a_val = loss.item(), acc.item()
            epoch_train_losses.append(l_val)
            epoch_train_accs.append(a_val)
            wandb.log({"batch/train_loss": l_val, "batch/train_accuracy": a_val})

        train_loss_avg = np.mean(epoch_train_losses)
        train_acc_avg = np.mean(epoch_train_accs)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f}")

        epoch_eval_losses = []
        epoch_eval_accs = []
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
        })
        
        _, state = nnx.split(model)
        save_path = checkpoint_dir / f"epoch_{epoch+1}"
        checkpointer.save(save_path.resolve(), state)
        checkpointer.wait_until_finished()
        print(f"Checkpoint saved to {save_path}")

    wandb.finish()

if __name__ == "__main__":
    main()
