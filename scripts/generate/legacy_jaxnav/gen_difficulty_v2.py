"""Generate environment pairs for difficulty feedback using JAX Rasterizer (V2) this is the difference from v1.

Creates random environment pairs, saves images + state arrays,
and populates the human_preference_pairs and llm_preference_pairs tables.

Usage:
    python scripts/generate/gen_difficulty_v2.py
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

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

from uedrlhf.config import FEEDBACK_TYPE, DATA_DIR, IMAGES_DIR, ARRAYS_DIR, DATA_VERSION
from uedrlhf.db import get_connection, create_pair_table, resolve_table_name
from uedrlhf.jax_renderer import get_jaxnav_rasterizer

# Fast fail to ensure V2 is active
if DATA_VERSION != "v2":
    raise ValueError(f"CRITICAL: Trying to run gen_difficulty_v2.py but data_version is '{DATA_VERSION}'. Set it to 'v2' in config.yaml.")

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

    # Generate environments
    seed = int(time.time())
    print(f"Using PRNG seed: {seed}")
    rng = jax.random.PRNGKey(seed)
    reset_rng = jax.random.split(rng, TOTAL_ENVS)
    env = JaxNav(num_agents=1)
    
    # We vmap reset to generate all states at once
    obsv, env_state = jax.vmap(env.reset)(reset_rng)
    
    # Initialize the faster, pure-JAX rasterizer
    rasterizer = get_jaxnav_rasterizer(img_height=120, img_width=120)
    rasterizer_jit = jax.jit(rasterizer)

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
            state = jax.tree_util.tree_map(lambda x: x[arr_idx], env_state)
            state_np = jax.tree_util.tree_map(lambda x: np.array(x), state)
            with open(ARRAYS_DIR / f"{env_id}.pkl", "wb") as f:
                pickle.dump(state_np, f)

            # Rasterize and Save Image natively
            img_jnp = rasterizer_jit(state)
            # The rasterizer returns [0.0, 1.0] floats, scale to 0-255 uint8
            img_np = np.asarray(img_jnp)
            img_uint8 = (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8)
            img_pil.save(IMAGES_DIR / f"{env_id}.png")

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

    print(f"[{FEEDBACK_TYPE}] Done! Saved to {DATA_DIR}/")
    print(f"Total pairs: {TOTAL_PAIRS}")
    print(f"Tables: {human_table}, {llm_table}")


if __name__ == "__main__":
    main()
