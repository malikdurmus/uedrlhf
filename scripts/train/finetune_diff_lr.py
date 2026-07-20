"""Fine-tune LearnabilityResNet with Differential Learning Rates.

- Stem & Stage 1: Frozen (0.0).
- Stage 2 & Stage 3: Slow Learning Rate (1e-5).
- Head: Normal Learning Rate (1e-3).
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

from uedrlhf.config import BASE_DIR, FINETUNING
from uedrlhf.networks import LearnabilityResNet

# ==========================================
PRETRAINED_CHECKPOINT_PATH = os.environ.get(
    "PRETRAINED_CHECKPOINT_PATH", 
    FINETUNING.get("pretrained_checkpoint_path", "")
)
# ==========================================

IMG_SIZE = FINETUNING.get("img_size", 120)
BATCH_SIZE = FINETUNING.get("batch_size", 64)
EPOCHS = FINETUNING.get("epochs", 30)
LEARNING_RATE = FINETUNING.get("learning_rate", 1e-3)
DATASET_PATH = os.environ.get(
    "DATASET_PATH", 
    FINETUNING.get("dataset_path", str(BASE_DIR / "data" / "difficulty" / "compiled_gemini_2_5_flash_dijkstra_15k"))
)

# Extract dataset folder name
_model_type = DATASET_PATH.rstrip("/").split("/")[-1]

PROJECT_NAME = os.environ.get("PROJECT_NAME", FINETUNING.get("project_name", "uedrlhf"))
RUN_NAME = os.environ.get("RUN_NAME", f"resnet-finetune-diff-lr-{_model_type}")


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
    acc_mask = (preferences != 0.5)
    accuracy = jnp.sum((predictions == preferences) * acc_mask) / jnp.maximum(1, jnp.sum(acc_mask))
    return loss, accuracy

@nnx.jit
def eval_step(model: LearnabilityResNet, batch: tuple):
    loss, logits = loss_fn(model, batch)
    preferences = batch[2]
    predictions = (logits > 0).astype(jnp.float32)
    acc_mask = (preferences != 0.5)
    accuracy = jnp.sum((predictions == preferences) * acc_mask) / jnp.maximum(1, jnp.sum(acc_mask))
    return loss, accuracy

def prefetch_to_gpu(dataloader, queue_size=2):
    queue = Queue(maxsize=queue_size)
    def producer():
        for batch_dict in dataloader:
            i0 = batch_dict['env_0_image'].numpy() if hasattr(batch_dict['env_0_image'], 'numpy') else np.array(batch_dict['env_0_image'])
            i1 = batch_dict['env_1_image'].numpy() if hasattr(batch_dict['env_1_image'], 'numpy') else np.array(batch_dict['env_1_image'])
            pref = batch_dict['preference'].numpy() if hasattr(batch_dict['preference'], 'numpy') else np.array(batch_dict['preference'])

            # The dataset images are already 0.0 - 1.0 float arrays
            i0_fp32 = i0.astype(np.float32)
            i1_fp32 = i1.astype(np.float32)
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

def get_diff_lr_optimizer():
    """Optax multi-transform distributing learning rates across CNN blocks."""
    def map_params_to_labels(path_tuple, value):
        path_str = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path_tuple)
        if "head" in path_str:
            return "head"
        elif "stage2" in path_str or "stage3" in path_str:
            return "slow"
        else:
            return "frozen"

    tx = optax.multi_transform(
        {
            'head': optax.adam(LEARNING_RATE), 
            'slow': optax.adam(LEARNING_RATE/100), 
            'frozen': optax.set_to_zero()
        },
        lambda params: jax.tree_util.tree_map_with_path(map_params_to_labels, params)
    )
    return tx


def main():
    if not PRETRAINED_CHECKPOINT_PATH:
        raise ValueError("CRITICAL: You must specify the PRETRAINED_CHECKPOINT_PATH at the top of the script!")

    wandb.init(
        project=PROJECT_NAME,
        name=RUN_NAME,
        config={
            "mode": "differential_learning",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "learning_rate": f"{LEARNING_RATE} (head), {LEARNING_RATE/100} (stages 2-3), 0.0 (stem+stage1)",
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

    print(f"Loading pretrained weights from: {PRETRAINED_CHECKPOINT_PATH}")
    checkpointer = ocp.StandardCheckpointer()
    _, state_def = nnx.split(model)
    restored_state = checkpointer.restore(PRETRAINED_CHECKPOINT_PATH, state_def)
    nnx.update(model, restored_state)
    print("Pretrained weights grafted.")

    # 2. CREATE ISOLATING OPTIMIZER
    optimizer = nnx.Optimizer(model, get_diff_lr_optimizer(), wrt=nnx.Param)

    best_eval_loss = float("inf")
    run_id = wandb.run.id if wandb.run is not None else "local"
    checkpoint_dir = BASE_DIR / "outputs" / "checkpoints" / f"{RUN_NAME}_{run_id}"

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
