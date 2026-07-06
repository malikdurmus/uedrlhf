"""Generate environment pairs for difficulty feedback.

Creates random environment pairs, saves images + state arrays,
and populates the human_preference_pairs and llm_preference_pairs tables.

Usage:
    python scripts/generate/gen_difficulty.py
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import pickle
import uuid
import time
import jax
import numpy as np
import matplotlib.pyplot as plt

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

from uedrlhf.config import FEEDBACK_TYPE, DATA_DIR, IMAGES_DIR, ARRAYS_DIR
from uedrlhf.db import get_connection, create_pair_table, resolve_table_name

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
    obsv, env_state = jax.vmap(env.reset)(reset_rng)
    rng, _rng = jax.random.split(rng)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

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

            ax.clear()
            env.init_render(ax, state, lidar=False, agent=True, goal=True)
            plt.savefig(IMAGES_DIR / f"{env_id}.png")

        cursor.execute(f"""
            INSERT INTO "{human_table}" (env_1_id, env_2_id, split, status) 
            VALUES (?, ?, ?, ?)
        """, (id_1, id_2, split, "pending"))

        cursor.execute(f"""
            INSERT INTO "{llm_table}" (env_1_id, env_2_id, split, status) 
            VALUES (?, ?, ?, ?)
        """, (id_1, id_2, split, "pending"))

    conn.commit()
    conn.close()
    plt.close(fig)

    print(f"[{FEEDBACK_TYPE}] Done! Saved to {DATA_DIR}/")
    print(f"Total pairs: {TOTAL_PAIRS}")
    print(f"Tables: {human_table}, {llm_table}")


if __name__ == "__main__":
    main()
