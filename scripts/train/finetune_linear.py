import time
import os
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
import wandb
from queue import Queue
import threading
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

from uedrlhf.config import BASE_DIR, FINETUNING
from uedrlhf.fast_data import print_run_banner
from uedrlhf.networks import LearnabilityResNet

# ==========================================
PRETRAINED_CHECKPOINT_PATH = os.environ.get(
    "PRETRAINED_CHECKPOINT_PATH", 
    FINETUNING.get("pretrained_checkpoint_path", "/home/d/durmusy/Desktop/GIT/new/uedrlhf/outputs/checkpoints/lbec1lyl/epoch_14")
)
# ==========================================

IMG_SIZE = FINETUNING.get("img_size", 120)
BATCH_SIZE = FINETUNING.get("batch_size", 64)
EPOCHS = FINETUNING.get("epochs", 30)
LEARNING_RATE = FINETUNING.get("learning_rate", 1e-3)
DATASET_PATH = os.environ.get(
    "DATASET_PATH", 
    FINETUNING.get("dataset_path", "/home/d/durmusy/Desktop/GIT/new/uedrlhf/data_v2/difficulty/compiled_human_preference_pairs_len_690")
)

# Extract dataset folder name
_model_type = DATASET_PATH.rstrip("/").split("/")[-1]

PROJECT_NAME = os.environ.get("PROJECT_NAME", FINETUNING.get("project_name", "uedrlhf"))
RUN_NAME = os.environ.get("RUN_NAME", f"resnet-finetune-linear_hf-{_model_type}")

SEED = int(os.environ.get("SEED", FINETUNING.get("seed", 0)))


def loss_fn(model: LearnabilityResNet, batch: tuple, deterministic: bool = False):
    img0, img1, preferences = batch

    #  pass  images through the cnn simultaneously to maximize gpu util 
    combined_inputs = jnp.concatenate([img0, img1], axis=0)
    scores = model(combined_inputs, deterministic=deterministic, return_logits=True)
    
    scores_input0, scores_input1 = jnp.split(scores, 2, axis=0)
    # preference=1 means img1 is harder; score1 - score0 makes higher score = more difficult
    logits = (scores_input1 - scores_input0)
    
    batch_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=preferences)
    loss = jnp.mean(batch_loss)
    return loss, logits


@nnx.jit
def train_step(model: LearnabilityResNet, optimizer: nnx.Optimizer, batch: tuple):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model=model, batch=batch)

    optimizer.update(model, grads)

    labels = batch[2]
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
    loss, logits = loss_fn(model, batch, deterministic=True)
    labels = batch[2]
    predictions = (logits > 0).astype(jnp.float32)
    accuracy = jnp.mean(predictions == labels)
    #accuracy = jnp.sum((predictions == preferences) * acc_mask) / jnp.maximum(1, jnp.sum(acc_mask))

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
            i0 = batch_dict['env_0_image'].numpy() if hasattr(batch_dict['env_0_image'], 'numpy') else np.array(batch_dict['env_0_image'])
            i1 = batch_dict['env_1_image'].numpy() if hasattr(batch_dict['env_1_image'], 'numpy') else np.array(batch_dict['env_1_image'])
            pref = batch_dict['preference'].numpy() if hasattr(batch_dict['preference'], 'numpy') else np.array(batch_dict['preference'])

            i0_fp32 = i0.astype(np.float32)
            i1_fp32 = i1.astype(np.float32)
            
            # Map preference so 1.0 is Right Image, 0.0 is Left Image
            p_fp32 = pref.astype(np.float32)

            batch = (jnp.array(i0_fp32), jnp.array(i1_fp32), jnp.array(p_fp32))
            queue.put(batch)
        queue.put(None)

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    while True:
        batch = queue.get()
        if batch is None:
            break
        yield batch


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
        raise ValueError("CRITICAL: You must specify the PRETRAINED_CHECKPOINT_PATH at the top of the script!")

    print_run_banner(
        script="finetune_linear.py",
        mode="linear_probe",
        pipeline="dataloader",
        project=PROJECT_NAME,
        run_name=RUN_NAME,
        dataset=DATASET_PATH,
        pretrained_checkpoint=PRETRAINED_CHECKPOINT_PATH,
        seed=SEED,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=f"{LEARNING_RATE} (head), 0.0 (cnn)",
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
        }
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading dataset from {DATASET_PATH}...")
    ds = load_from_disk(DATASET_PATH)

    ds_splitted = ds.train_test_split(test_size=0.1, seed=SEED)
    ds_splitted.set_format(type="numpy", columns=["env_0_image", "env_1_image", "preference"])

    train_loader = DataLoader(ds_splitted["train"], batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)
    eval_loader = DataLoader(ds_splitted["test"], batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)

    rngs = nnx.Rngs(SEED)
    model = LearnabilityResNet(rngs=rngs)

    # 1. LOAD PRETRAINED WEIGHTS
    print(f"Loading pretrained weights from: {PRETRAINED_CHECKPOINT_PATH}")
    
    graphdef, abstract_state = nnx.split(model)
    
    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(PRETRAINED_CHECKPOINT_PATH, abstract_state)
    
    model = nnx.merge(graphdef, restored_state)
    

    print("Pretrained weights loaded.")

    optimizer = nnx.Optimizer(model, get_linear_optimizer(), wrt=nnx.Param)

    best_eval_loss = float("inf")
    run_id = wandb.run.id if wandb.run is not None else "local"
    checkpoint_dir = BASE_DIR / "outputs" / "checkpoints" / f"{RUN_NAME}_{run_id}"

    print(f"Starting Linear Finetune on {len(ds_splitted['train'])} train comparisons, {len(ds_splitted['test'])} eval comparisons...")
    
    for epoch in range(EPOCHS):
        epoch_train_losses = []
        epoch_train_accs = []
        epoch_train_f1s = []
        model.train()
        for batch_idx, batch_tuple in enumerate(prefetch_to_gpu(train_loader)):
            loss, acc, f1 = train_step(model=model, optimizer=optimizer, batch=batch_tuple)

            l_val, a_val, f1_val = loss.item(), acc.item(), f1.item()
            epoch_train_losses.append(l_val)
            epoch_train_accs.append(a_val)
            epoch_train_f1s.append(f1_val)

            wandb.log({
                "batch/train_loss": l_val,
                "batch/train_accuracy": a_val,
                "batch/train_f1": f1_val,
            })

        train_loss_avg = np.mean(epoch_train_losses)
        train_acc_avg = np.mean(epoch_train_accs)
        train_f1_avg = np.mean(epoch_train_f1s)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f}, Train F1: {train_f1_avg:.4f}")

        epoch_eval_losses = []
        epoch_eval_accs = []
        epoch_eval_f1s = []
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
        })
        
        _, state = nnx.split(model)
        save_path = checkpoint_dir / f"epoch_{epoch+1}"
        checkpointer.save(save_path.resolve(), state)
        checkpointer.wait_until_finished()
        print(f"Checkpoint saved to {save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
