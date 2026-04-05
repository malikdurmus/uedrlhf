import os
import sqlite3
import pickle
import uuid
import yaml
from pathlib import Path
import jax
import numpy as np
import matplotlib.pyplot as plt

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

# load config
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent
config_path = base_dir / "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

FEEDBACK_TYPE = config["feedback_type"]           # "difficulty_feedback" or "learnability_feedback"

# Map feedback_type to subfolder name
SUBFOLDER_MAP = {
    "difficulty_feedback": "difficulty",
    "learnability_feedback": "learnability",
}

if FEEDBACK_TYPE not in SUBFOLDER_MAP:
    raise ValueError(
        f"Unknown feedback_type '{FEEDBACK_TYPE}' in config.yaml. "
        f"Must be one of: {list(SUBFOLDER_MAP.keys())}"
    )

subfolder = SUBFOLDER_MAP[FEEDBACK_TYPE]

# env generation config TODO: make it configurable
NUM_TRAIN_PAIRS = 10000
NUM_TEST_PAIRS = 1000
TOTAL_PAIRS = NUM_TRAIN_PAIRS + NUM_TEST_PAIRS
TOTAL_ENVS = TOTAL_PAIRS * 2  # 2 envs per pair

# paths
data_dir = base_dir / "data" / subfolder
img_dir = data_dir / "images"
arr_dir = data_dir / "arrays"
db_path = data_dir / "feedback.db"

# check directories
img_dir.mkdir(parents=True, exist_ok=True)
arr_dir.mkdir(parents=True, exist_ok=True)

#db 

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS human_preference_pairs (
        pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
        env_1_id TEXT,
        env_2_id TEXT,
        split TEXT,
        preference REAL,
        status TEXT, x_pct REAL , y_pct REAL , time_ms REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS llm_preference_pairs (
        pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
        env_1_id TEXT,
        env_2_id TEXT,
        split TEXT,
        preference REAL,
        status TEXT, x_pct REAL , y_pct REAL , time_ms REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')


conn.commit()



import time
seed = int(time.time())
print(f"Using PRNG seed: {seed}")
rng = jax.random.PRNGKey(seed)  # Dynamic seed for unique envs across runs
reset_rng = jax.random.split(rng, TOTAL_ENVS)
env = JaxNav(num_agents=1)
obsv, env_state = jax.vmap(env.reset)(reset_rng)
rng , _rng = jax.random.split(rng)


fig, ax = plt.subplots(1, 1, figsize=(5, 5))


# generate envs image arrays

for pair_idx in range(TOTAL_PAIRS):
    # split the data
    split = "train" if pair_idx < NUM_TRAIN_PAIRS else "test"
    
    # generate uuid
    id_1 = f"{split}_{pair_idx:03d}_A_{uuid.uuid4().hex[:8]}"
    id_2 = f"{split}_{pair_idx:03d}_B_{uuid.uuid4().hex[:8]}"
    
    
    idx_1 = pair_idx * 2
    idx_2 = pair_idx * 2 + 1
    
    for env_id, arr_idx in [(id_1, idx_1), (id_2, idx_2)]:
        # extract state from batched array
        state = jax.tree_util.tree_map(lambda x: x[arr_idx], env_state)
        
        #  save arrays in numpy (jax creates problems with different versions )
        state_np = jax.tree_util.tree_map(lambda x: np.array(x), state)
        with open(arr_dir / f"{env_id}.pkl", "wb") as f:
            pickle.dump(state_np, f)
    
        # save image
        ax.clear() 
        env.init_render(ax, state, lidar=False, agent=True, goal=True)
        plt.savefig(img_dir / f"{env_id}.png")

    # save to preference db
    cursor.execute("""
        INSERT INTO human_preference_pairs (env_1_id, env_2_id, split, status) 
        VALUES (?, ?, ?, ?)
    """, (id_1, id_2, split, "pending"))

    cursor.execute("""
        INSERT INTO llm_preference_pairs (env_1_id, env_2_id, split, status) 
        VALUES (?, ?, ?, ?)
    """, (id_1, id_2, split, "pending"))


# clean up
conn.commit()
conn.close()
plt.close(fig)

print(f"[{FEEDBACK_TYPE}] Done! Saved to {data_dir}/")
print(f"Total pairs: {TOTAL_PAIRS}")
print("Two different tables are saved: human_feedback_pairs and llm_feedback_pairs")
