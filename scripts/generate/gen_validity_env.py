"""
Generates valid and invalid JaxNav environments and saves them as a HF dataset.

Dataset columns:
    image       : PIL Image (120x120 RGB render of the environment)
    map_data    : 11x11 int32 array (occupancy grid, 1=wall, 0=free)
    goal_pos    : float32 array shape (num_agents, 2) - goal (x, y) per agent
    agent_pos   : float32 array shape (num_agents, 2)  -agent (x, y) per agent
    solvable    : int32 label (1=solvable, 0=unsolvable)
"""

import argparse
import threading
import queue
import os
import shutil
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from datasets import Dataset, Features, Array2D, Array3D, Value, Image, Sequence
from datasets import concatenate_datasets, load_from_disk

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from uedrlhf.utils import get_jaxnav_rasterizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_envs",       type=int, default=100_000,
                        help="Total number of environments (50%% solvable, 50%% unsolvable)")
    parser.add_argument("-m", "--max_size_queue", type=int, default=10,
                        help="Max batches buffered between producer and consumer")
    parser.add_argument("-b", "--batch_size",     type=int, default=5_000,
                        help="JAX environments generated per vmap call")
    parser.add_argument("--shard_size",           type=int, default=10_000,
                        help="Examples per on-disk shard (controls peak RAM)")
    parser.add_argument("-o", "--out_dir",        type=str, default="data/validity_dataset")
    parser.add_argument("-s", "--seed",           type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HuggingFace dataset features
# ---------------------------------------------------------------------------
FEATURES = Features({
    "image":     Image(),           # stored as PNG-compressed bytes
    "map_data":  Array2D(shape=(11, 11), dtype="int32"),
    "goal_pos":  Array2D(shape=(1, 2),  dtype="float32"),   # shape: (num_agents, 2)
    "agent_pos": Array2D(shape=(1, 2),  dtype="float32"),
    "solvable":  Value("int32"),
})


def save_shard(records: dict, shard_dir: str):
    """Write one shard of accumulated records to disk."""
    # `records` is a dict of lists; each list has `shard_size` entries.
    ds = Dataset.from_dict(records, features=FEATURES)
    ds.save_to_disk(shard_dir)
    print(f"  Saved shard → {shard_dir}  ({len(ds)} examples)")


def main(args):
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    num_solvable   = args.num_envs // 2
    num_unsolvable = args.num_envs - num_solvable

    def n_batches(total):
        return int(np.ceil(total / args.batch_size))

    env = JaxNav(num_agents=1)

    render_fn = get_jaxnav_rasterizer(
        img_height=120, img_width=120,
        map_height=11.0, map_width=11.0,
        cell_size=1.0,
    )
    vmapped_render = jax.vmap(render_fn)

    @partial(jax.jit, static_argnames=["solvability"])
    def generate_batch(keys, solvability="both"):
        obs, states = jax.vmap(env.reset, in_axes=(0, None))(keys, solvability)
        images = vmapped_render(states)                       # (B, H, W, 3) float32
        images_u8 = (images * 255).astype(jnp.uint8)
        return images_u8, states

    # ------------------------------------------------------------------
    # Producer — generates batches and pushes them onto the queue
    # ------------------------------------------------------------------
    q: queue.Queue = queue.Queue(maxsize=args.max_size_queue)

    def producer():
        rng = jax.random.PRNGKey(args.seed)

        def push_batches(total, solvability, label):
            nonlocal rng
            for i in range(n_batches(total)):
                rng, _rng = jax.random.split(rng)
                bs = min(args.batch_size, total - i * args.batch_size)
                keys = jax.random.split(_rng, num=bs)
                print(f"Generating {solvability} batch {i+1}/{n_batches(total)} (size={bs})")
                images_u8, states = generate_batch(keys, solvability=solvability)
                q.put((
                    np.array(images_u8),                          # (bs, 120, 120, 3) uint8
                    np.array(states.map_data, dtype=np.int32),    # (bs, 11, 11)
                    np.array(states.goal,     dtype=np.float32),  # (bs, num_agents, 2)
                    np.array(states.pos,      dtype=np.float32),  # (bs, num_agents, 2)
                    np.full(bs, label, dtype=np.int32),           # (bs,)
                ))

        push_batches(num_solvable,   "solvable",   label=1)
        push_batches(num_unsolvable, "unsolvable", label=0)
        q.put(None)  # sentinel
        print("Producer finished.")

    # ------------------------------------------------------------------
    # Consumer — accumulates records, flushes shards to disk
    # ------------------------------------------------------------------
    def consumer():
        shard_dir_prefix = os.path.join(args.out_dir, "_shards", "shard")
        os.makedirs(os.path.dirname(shard_dir_prefix), exist_ok=True)

        def empty_records():
            return {"image": [], "map_data": [], "goal_pos": [],
                    "agent_pos": [], "solvable": []}

        records    = empty_records()
        shard_idx  = 0
        shard_dirs = []
        total_seen = 0

        def flush(force=False):
            nonlocal records, shard_idx
            if not records["image"]:
                return
            if force or len(records["image"]) >= args.shard_size:
                sd = f"{shard_dir_prefix}_{shard_idx:04d}"
                shard_dirs.append(sd)
                save_shard(records, sd)
                records = empty_records()
                shard_idx += 1

        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break

            images, map_data, goal, pos, labels = item
            bs = len(labels)

            for i in range(bs):
                records["image"].append(images[i])        # numpy array → PIL via Image()
                records["map_data"].append(map_data[i])   # (11, 11)
                records["goal_pos"].append(goal[i])       # (num_agents, 2)
                records["agent_pos"].append(pos[i])       # (num_agents, 2)
                records["solvable"].append(int(labels[i]))

            total_seen += bs
            q.task_done()
            print(f"Consumer: {total_seen}/{args.num_envs} examples buffered")
            flush()

        flush(force=True)  # write any remaining examples

        # ------------------------------------------------------------------
        # Merge shards into a single dataset
        # ------------------------------------------------------------------
        print(f"\nMerging {len(shard_dirs)} shards...")
        shards = [load_from_disk(sd) for sd in shard_dirs]
        final_ds = concatenate_datasets(shards)

        out_path = os.path.join(args.out_dir, "dataset")
        os.makedirs(out_path, exist_ok=True)
        final_ds.save_to_disk(out_path)
        print(f"Final dataset saved → {out_path}  ({len(final_ds)} examples)")

        del shards
        del final_ds
        import gc
        gc.collect()

        # 2. Use python's shutil to delete the directory, ignoring open-file locks if they persist
        try:
            shutil.rmtree(os.path.dirname(shard_dir_prefix), ignore_errors=True)
            print("Shard temp files removed.")
        except Exception as e:
            print(f"Warning: Could not completely remove temp shards: {e}")
    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    consumer_error = []   # shared list to capture exception from consumer thread

    def consumer_safe():
        try:
            consumer()
        except Exception as e:
            consumer_error.append(e)
            raise

    prod_thread = threading.Thread(target=producer, daemon=True)
    cons_thread = threading.Thread(target=consumer_safe, daemon=True)

    prod_thread.start()
    cons_thread.start()

    prod_thread.join()
    cons_thread.join()

    if consumer_error:
        raise RuntimeError(f"Consumer thread failed: {consumer_error[0]}") from consumer_error[0]

    print("All done.")


if __name__ == "__main__":
    _args = parse_args()
    main(_args)