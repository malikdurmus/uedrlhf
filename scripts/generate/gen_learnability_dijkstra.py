"""Generate environment pairs for learnability feedback using Dijkstra distance binning.

1. Generate environments and compute Dijkstra shortest path distances
2. Bin by distance percentiles (4 equal bins)
3. Create pairs within each bin
4. Save images + state arrays and populate DB tables

Usage:
    python scripts/generate/gen_learnability_dijkstra.py
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

from uedrlhf.config import DATA_DIR, IMAGES_DIR, ARRAYS_DIR
from uedrlhf.db import get_connection, create_pair_table, resolve_table_name

NUM_TRAIN_PAIRS = 7000
NUM_TEST_PAIRS = 700
NUM_PAIRS_PER_BIN = (NUM_TRAIN_PAIRS + NUM_TEST_PAIRS) // 4  # spread evenly across bins
NUM_SAMPLES = 20000  # environments to generate for binning


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

    # Database setup
    human_table = resolve_table_name("human")
    llm_table = resolve_table_name("llm")

    conn = get_connection()
    cursor = conn.cursor()
    create_pair_table(cursor, human_table, include_bin=True)
    create_pair_table(cursor, llm_table, include_bin=True)
    conn.commit()

    # Environment setup
    seed = int(time.time())
    print(f"Using PRNG seed: {seed}")
    rng = jax.random.PRNGKey(seed)
    env = JaxNav(num_agents=1)

    # --- Step 1: Generate environments and compute Dijkstra distances ---
    print(f"Generating {NUM_SAMPLES} environments for binning...")
    batch_size = 500
    keys = jax.random.split(rng, NUM_SAMPLES)
    rng, _ = jax.random.split(rng)

    all_map_data, all_cases, all_valid, all_d = [], [], [], []

    for i in range(0, NUM_SAMPLES, batch_size):
        if i % 2000 == 0:
            print(f"  {i}/{NUM_SAMPLES}")
        batch_keys = keys[i:i + batch_size]
        map_data_batch, case_batch = jax.vmap(env._map_obj.sample_test_case)(batch_keys)
        starts = case_batch[:, 0, 0, :2]
        goals = case_batch[:, 0, 1, :2]
        valid_batch, d_batch = jax.vmap(env._map_obj.dikstra_path)(map_data_batch, starts, goals)

        all_map_data.append(np.array(map_data_batch))
        all_cases.append(np.array(case_batch))
        all_valid.append(np.array(valid_batch))
        all_d.append(np.array(d_batch))

    all_map_data = np.concatenate(all_map_data)
    all_cases = np.concatenate(all_cases)
    all_valid = np.concatenate(all_valid)
    all_d = np.concatenate(all_d)

    # --- Step 2: Filter unsolvable ---
    solvable_mask = all_valid
    map_data_sol = all_map_data[solvable_mask]
    cases_sol = all_cases[solvable_mask]
    d_sol = all_d[solvable_mask]

    print(f"Solvable: {solvable_mask.sum()} / {NUM_SAMPLES} ({100 * solvable_mask.mean():.1f}%)")

    # --- Step 3: Compute bin boundaries from percentiles ---
    boundaries = np.percentile(d_sol, [0, 25, 50, 75, 100])
    bin_assignments = np.digitize(d_sol, boundaries[1:-1])  # 0, 1, 2, 3

    print(f"Bin boundaries: {boundaries}")
    for bin_idx in range(4):
        count = (bin_assignments == bin_idx).sum()
        low, high = boundaries[bin_idx], boundaries[bin_idx + 1]
        print(f"  Bin {bin_idx} [{low:.1f}, {high:.1f}): {count} environments")

    # --- Step 4: Create pairs and save ---
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    pair_idx = 0
    total_pairs = 0

    for bin_idx in range(4):
        bin_env_ids = np.where(bin_assignments == bin_idx)[0]

        if len(bin_env_ids) < 2:
            print(f"Warning: bin {bin_idx} has fewer than 2 environments, skipping")
            continue

        sampled_pairs = set()
        attempts = 0

        while len(sampled_pairs) < NUM_PAIRS_PER_BIN and attempts < NUM_PAIRS_PER_BIN * 10:
            rng, key = jax.random.split(rng)
            idx_a, idx_b = jax.random.choice(key, len(bin_env_ids), (2,), replace=False)
            pair = (min(int(bin_env_ids[idx_a]), int(bin_env_ids[idx_b])),
                    max(int(bin_env_ids[idx_a]), int(bin_env_ids[idx_b])))
            sampled_pairs.add(pair)
            attempts += 1

        print(f"Bin {bin_idx}: {len(sampled_pairs)} pairs")

        for arr_idx_1, arr_idx_2 in sampled_pairs:
            split = "train" if pair_idx < NUM_TRAIN_PAIRS else "test"

            id_1 = f"{split}_{pair_idx:04d}_A_{uuid.uuid4().hex[:8]}"
            id_2 = f"{split}_{pair_idx:04d}_B_{uuid.uuid4().hex[:8]}"

            for env_id, arr_idx in [(id_1, arr_idx_1), (id_2, arr_idx_2)]:
                # Reconstruct env state via reset with the original key
                reset_rng_single = keys[arr_idx]
                _, env_state_single = env.reset(reset_rng_single)

                state_np = jax.tree_util.tree_map(lambda x: np.array(x), env_state_single)
                with open(ARRAYS_DIR / f"{env_id}.pkl", "wb") as f:
                    pickle.dump(state_np, f)

                ax.clear()
                env.init_render(ax, env_state_single, lidar=False, agent=True, goal=True)
                plt.savefig(IMAGES_DIR / f"{env_id}.png")

            cursor.execute(f"""
                INSERT INTO "{human_table}" (env_1_id, env_2_id, split, bin, status)
                VALUES (?, ?, ?, ?, ?)
            """, (id_1, id_2, split, bin_idx, "pending"))

            cursor.execute(f"""
                INSERT INTO "{llm_table}" (env_1_id, env_2_id, split, bin, status)
                VALUES (?, ?, ?, ?, ?)
            """, (id_1, id_2, split, bin_idx, "pending"))

            pair_idx += 1
            total_pairs += 1

    conn.commit()
    conn.close()
    plt.close(fig)

    print(f"Done! Saved to {DATA_DIR}/")
    print(f"Total pairs: {total_pairs}")
    print(f"Tables: {human_table}, {llm_table}")


if __name__ == "__main__":
    main()
