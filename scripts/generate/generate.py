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

NUM_TRAIN_PAIRS = 7000
NUM_TEST_PAIRS = 700
TOTAL_PAIRS = NUM_TRAIN_PAIRS + NUM_TEST_PAIRS
TOTAL_ENVS = TOTAL_PAIRS * 2


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

    # Generate all environment states at once
    seed = int(time.time())
    print(f"Using PRNG seed: {seed}")
    rng = jax.random.PRNGKey(seed)

    print(f"Generating {TOTAL_ENVS} environments...")
    states = backend.generate_states(rng, TOTAL_ENVS)

    # Render all images in one batch
    print("Rendering all environments...")
    all_images = backend.render_batch(states)
    print(f"Rendered batch shape: {all_images.shape}")

    # Create pairs and save
    for pair_idx in range(TOTAL_PAIRS):
        split = "train" if pair_idx < NUM_TRAIN_PAIRS else "test"

        while True:
            id_1 = f"{split}_{pair_idx:03d}_A_{uuid.uuid4().hex[:8]}"
            id_2 = f"{split}_{pair_idx:03d}_B_{uuid.uuid4().hex[:8]}"
            if not (IMAGES_DIR / f"{id_1}.png").exists() and not (IMAGES_DIR / f"{id_2}.png").exists():
                break

        idx_1 = pair_idx * 2
        idx_2 = pair_idx * 2 + 1

        for env_id, arr_idx in [(id_1, idx_1), (id_2, idx_2)]:
            # Extract single state from the batch
            state = jax.tree_util.tree_map(lambda x: x[arr_idx], states)

            # Serialize and save state
            state_dict = backend.state_to_dict(state)
            with open(ARRAYS_DIR / f"{env_id}.pkl", "wb") as f:
                pickle.dump(state_dict, f)

            # Save pre-rendered image
            img_pil = Image.fromarray(all_images[arr_idx])
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

        # Progress marker
        if (pair_idx + 1) % 500 == 0:
            print(f"Generated {pair_idx + 1} / {TOTAL_PAIRS} pairs...")
            conn.commit()

    conn.commit()
    conn.close()

    print(f"[{ENVIRONMENT} / {FEEDBACK_TYPE}] Done! Saved to {DATA_DIR}/")
    print(f"Total pairs: {TOTAL_PAIRS}")
    print(f"Tables: {human_table}, {llm_table}")


if __name__ == "__main__":
    main()
