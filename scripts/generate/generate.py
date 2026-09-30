"""Generate environment pairs for difficulty feedback (environment-agnostic).

Works with any environment backend (JaxNav, XLand-MiniGrid, etc.) by
reading the `environment` field from config.yaml.

Creates random environment pairs, saves images + state arrays,
and populates the human_preference_pairs and llm_preference_pairs tables.

Usage:
    uv run scripts/generate/generate.py
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import pickle
import uuid
import time
import jax
import numpy as np
from PIL import Image

from uedrlhf.config import ENVIRONMENT, ENV_CONFIG, FEEDBACK_TYPE, DATA_DIR, IMAGES_DIR, ARRAYS_DIR
from uedrlhf.db import get_connection, create_pair_table, resolve_table_name
from uedrlhf.environments import get_env_backend

NUM_TRAIN_PAIRS = 20000
NUM_TEST_PAIRS = 2000
TOTAL_PAIRS = NUM_TRAIN_PAIRS + NUM_TEST_PAIRS

# How many pairs to generate + render per GPU batch (2 envs per pair).
# 5000 pairs = 10 000 envs ≈ 2 GB on a 9×9 grid with tile_size=32.
BATCH_PAIRS = 2000


def main():
    # Ensure directories exist
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

    # Database setup
    human_table = resolve_table_name("human")
    llm_table = resolve_table_name("llm")

    conn = get_connection()
    cursor = conn.cursor()
    create_pair_table(cursor, human_table)
    create_pair_table(cursor, llm_table)
    conn.commit()

    # Initialize environment backend
    print(f"Environment: {ENVIRONMENT}")
    backend = get_env_backend(ENVIRONMENT, ENV_CONFIG)
    backend.create_env()

    seed = int(time.time())
    print(f"Using PRNG seed: {seed}")
    rng = jax.random.PRNGKey(seed)

    num_batches = (TOTAL_PAIRS + BATCH_PAIRS - 1) // BATCH_PAIRS
    print(f"Total pairs: {TOTAL_PAIRS}  |  Batch size: {BATCH_PAIRS} pairs  |  Batches: {num_batches}")

    global_pair_idx = 0

    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_PAIRS
        batch_end = min(batch_start + BATCH_PAIRS, TOTAL_PAIRS)
        batch_pair_count = batch_end - batch_start
        batch_env_count = batch_pair_count * 2

        print(f"\n=== Batch {batch_idx + 1}/{num_batches}: pairs {batch_start}–{batch_end - 1} ({batch_env_count} envs) ===")

        # Split a fresh key for this batch
        rng, batch_rng = jax.random.split(rng)

        # Generate states for this batch
        states = backend.generate_states(batch_rng, batch_env_count)

        # Render this batch
        print(f"  Rendering {batch_env_count} environments...")
        batch_images = backend.render_batch(states)
        print(f"  Rendered batch shape: {batch_images.shape}")

        # Save pairs from this batch
        for local_pair_idx in range(batch_pair_count):
            split = "train" if global_pair_idx < NUM_TRAIN_PAIRS else "test"

            while True:
                id_1 = f"{split}_{global_pair_idx:05d}_A_{uuid.uuid4().hex[:8]}"
                id_2 = f"{split}_{global_pair_idx:05d}_B_{uuid.uuid4().hex[:8]}"
                if not (IMAGES_DIR / f"{id_1}.png").exists() and not (IMAGES_DIR / f"{id_2}.png").exists():
                    break

            env_idx_1 = local_pair_idx * 2
            env_idx_2 = local_pair_idx * 2 + 1

            for env_id, arr_idx in [(id_1, env_idx_1), (id_2, env_idx_2)]:
                # Extract single state from the batch
                state = jax.tree_util.tree_map(lambda x: x[arr_idx], states)

                # Serialize and save state
                state_dict = backend.state_to_dict(state)
                with open(ARRAYS_DIR / f"{env_id}.pkl", "wb") as f:
                    pickle.dump(state_dict, f)

                # Save pre-rendered image
                img_pil = Image.fromarray(batch_images[arr_idx])
                img_pil.save(IMAGES_DIR / f"{env_id}.png")

            # Insert pair into both tables
            cursor.execute(f"""
                INSERT INTO "{human_table}" (env_1_id, env_2_id, split, status) 
                VALUES (?, ?, ?, ?)
            """, (id_1, id_2, split, "pending"))

            cursor.execute(f"""
                INSERT INTO "{llm_table}" (env_1_id, env_2_id, split, status) 
                VALUES (?, ?, ?, ?)
            """, (id_1, id_2, split, "pending"))

            global_pair_idx += 1

            if global_pair_idx % 500 == 0:
                print(f"  Saved {global_pair_idx} / {TOTAL_PAIRS} pairs...")

        # Commit after each batch and let JAX free GPU memory
        conn.commit()
        del states, batch_images
        print(f"  Batch {batch_idx + 1} committed.")

    conn.close()

    print(f"\n[{ENVIRONMENT} / {FEEDBACK_TYPE}] Done! Saved to {DATA_DIR}/")
    print(f"Total pairs: {TOTAL_PAIRS}")
    print(f"Tables: {human_table}, {llm_table}")


if __name__ == "__main__":
    main()

