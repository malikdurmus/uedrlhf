import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import wandb
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from datasets import load_from_disk

from uedrlhf.config import BASE_DIR, PRETRAINING
from uedrlhf.networks import LearnabilityResNet
from uedrlhf.fast_data import flag_from, print_run_banner, random_symmetry, to_float

IMG_SIZE = 120
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
DATASET_PATH = os.environ.get("DATASET_PATH", str(BASE_DIR / "dataset_1M" / "dataset"))
SEED = int(os.environ.get("SEED", PRETRAINING.get("seed", 0)))
RUN_NAME = os.environ.get("RUN_NAME", f"resnet-pretrain-solvability-fast_seed{SEED}")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "uedrlhf_TRAINING-solvabilitypretrain")
AUGMENT_SYMMETRY = flag_from("AUGMENT_SYMMETRY", PRETRAINING, "augment_symmetry")

DECODE_THREADS = int(os.environ.get("DECODE_THREADS", "6"))
PREFETCH_WINDOW = 12


def loss_fn(model: LearnabilityResNet, batch: tuple, deterministic: bool = False):
    images, labels = batch
    logits = model(images, deterministic=deterministic, return_logits=True)
    batch_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
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


def _train_body(model, optimizer, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    acc, f1 = _metrics(logits, batch[1])
    return loss, acc, f1


@nnx.jit
def train_step(model: LearnabilityResNet, optimizer: nnx.Optimizer, batch: tuple):
    imgs_u8, labels = batch
    return _train_body(model, optimizer, (to_float(imgs_u8), labels))


@nnx.jit
def train_step_aug(model: LearnabilityResNet, optimizer: nnx.Optimizer, batch: tuple, aug_key):
    imgs_u8, labels = batch
    imgs_u8 = random_symmetry(aug_key, imgs_u8)
    return _train_body(model, optimizer, (to_float(imgs_u8), labels))


@nnx.jit
def eval_step(model: LearnabilityResNet, batch: tuple):
    imgs_u8, labels = batch
    loss, logits = loss_fn(model, (to_float(imgs_u8), labels), deterministic=True)
    acc, f1 = _metrics(logits, labels)
    return loss, acc, f1


def decode_rows(d, idx):
    """Decode one batch of rows to (uint8 images, flipped float32 labels)."""
    rows = d[idx]
    imgs = np.asarray(rows["image"])
    labels = 1.0 - np.asarray(rows["solvable"], dtype=np.float32)
    return imgs, labels


def prefetch_batches(d, order):
    """Decode upcoming batches in a thread pool while the GPU consumes earlier ones."""
    batch_idx = [order[s : s + BATCH_SIZE].tolist() for s in range(0, len(order), BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=DECODE_THREADS) as pool:
        pending = deque()
        it = iter(batch_idx)
        for _ in range(PREFETCH_WINDOW):
            try:
                pending.append(pool.submit(decode_rows, d, next(it)))
            except StopIteration:
                break
        while pending:
            imgs, labels = pending.popleft().result()
            try:
                pending.append(pool.submit(decode_rows, d, next(it)))
            except StopIteration:
                pass
            yield jnp.asarray(imgs), jnp.asarray(labels)


def device_batches(images_dev, labels_dev, n):
    for start in range(0, n, BATCH_SIZE):
        yield images_dev[start : start + BATCH_SIZE], labels_dev[start : start + BATCH_SIZE]


def main():
    print_run_banner(
        script="pretrain_solvability_fast.py",
        pipeline="threaded_prefetch",
        project=PROJECT_NAME,
        run_name=RUN_NAME,
        dataset=DATASET_PATH,
        seed=SEED,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        augment_symmetry=AUGMENT_SYMMETRY,
        decode_threads=DECODE_THREADS,
    )
    wandb.init(
        project=PROJECT_NAME,
        name=RUN_NAME,
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "adam",
            "model": "LearnabilityResNet",
            "seed": SEED,
            "augment_symmetry": AUGMENT_SYMMETRY,
            "dataset_path": DATASET_PATH,
            "pipeline": "threaded_prefetch",
        },
    )

    np.random.seed(SEED)

    print(f"Loading dataset from {DATASET_PATH}...")
    ds = load_from_disk(DATASET_PATH)
    ds_splitted = ds.train_test_split(test_size=0.1, seed=SEED)
    train_d = ds_splitted["train"].with_format("numpy", columns=["image", "solvable"])
    n_train = len(train_d)

    t0 = time.perf_counter()
    print("Decoding eval split to device...")
    eval_d = ds_splitted["test"].with_format("numpy", columns=["image", "solvable"])
    n_eval = len(eval_d)
    _imgs = np.empty((n_eval, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _lbls = np.empty(n_eval, dtype=np.float32)
    def fill(s):
        e = min(s + 2048, n_eval)
        _imgs[s:e], _lbls[s:e] = decode_rows(eval_d, list(range(s, e)))
    with ThreadPoolExecutor(max_workers=DECODE_THREADS) as pool:
        list(pool.map(fill, range(0, n_eval, 2048)))
    eval_images_dev, eval_labels_dev = jnp.asarray(_imgs), jnp.asarray(_lbls)
    jax.block_until_ready(eval_images_dev)
    del _imgs, _lbls
    print(f"Eval on device: {n_eval} images ({time.perf_counter() - t0:.1f}s)")

    rngs = nnx.Rngs(SEED)
    model = LearnabilityResNet(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=LEARNING_RATE), wrt=nnx.Param)

    best_eval_loss = float("inf")
    checkpoint_dir = BASE_DIR / "outputs" / "checkpoints" / wandb.run.id
    checkpointer = ocp.StandardCheckpointer()

    root_key = jax.random.PRNGKey(SEED)
    shuffle_key = jax.random.fold_in(root_key, 0)
    aug_key = jax.random.fold_in(root_key, 1)
    print(f"Symmetry augmentation: {'ON' if AUGMENT_SYMMETRY else 'off'}")

    print(f"Starting training on {n_train} train examples, {n_eval} eval examples...")
    for epoch in range(EPOCHS):
        t_epoch = time.perf_counter()
        shuffle_key, subkey = jax.random.split(shuffle_key)
        order = np.asarray(jax.random.permutation(subkey, n_train))

        model.train()
        tl, ta, tf = [], [], []
        for batch in prefetch_batches(train_d, order):
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
        el, ea, ef = [], [], []
        for batch in device_batches(eval_images_dev, eval_labels_dev, n_eval):
            loss, acc, f1 = eval_step(model, batch)
            el.append(loss); ea.append(acc); ef.append(f1)

        eval_loss_avg = float(np.mean(jax.device_get(el)))
        eval_acc_avg = float(np.mean(jax.device_get(ea)))
        eval_f1_avg = float(np.mean(jax.device_get(ef)))
        if eval_loss_avg < best_eval_loss:
            best_eval_loss = eval_loss_avg

        print(f"Epoch {epoch+1}/{EPOCHS} - Eval Loss: {eval_loss_avg:.4f}, Eval Acc: {eval_acc_avg:.4f}, Eval F1: {eval_f1_avg:.4f} "
              f"({time.perf_counter() - t_epoch:.1f}s)")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss_avg,
            "train/accuracy": train_acc_avg,
            "train/f1": train_f1_avg,
            "eval/loss": eval_loss_avg,
            "eval/accuracy": eval_acc_avg,
            "eval/f1": eval_f1_avg,
            "eval/loss_gap": eval_loss_avg - train_loss_avg,
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
