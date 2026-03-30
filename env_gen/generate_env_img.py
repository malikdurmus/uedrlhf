import os
import sqlite3
import pickle
import uuid
from pathlib import Path
import jax
import numpy as np
import matplotlib.pyplot as plt

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

# config # To do: use Config.yaml 
NUM_TRAIN_PAIRS = 50
NUM_TEST_PAIRS = 20
TOTAL_PAIRS = NUM_TRAIN_PAIRS + NUM_TEST_PAIRS
TOTAL_ENVS = TOTAL_PAIRS * 2  # 2 envs per pair

# path
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent  
img_dir = base_dir / "data" / "images"
arr_dir = base_dir / "data" / "arrays"
db_path = base_dir / "data" / "feedback.db"


# check directories
img_dir.mkdir(parents=True, exist_ok=True)
arr_dir.mkdir(parents=True, exist_ok=True)

#db 

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS preference_pairs (
        pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
        env_1_id TEXT,
        env_2_id TEXT,
        split TEXT,
        preference REAL,
        status TEXT, x_pct REAL , y_pct REAL , time_ms REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()



rng = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
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
    id_1 = f"{split}_{pair_idx:03d}_A_{uuid.uuid4().hex[:4]}"
    id_2 = f"{split}_{pair_idx:03d}_B_{uuid.uuid4().hex[:4]}"
    
    
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
    
    # save to prefrence db
    cursor.execute("""
        INSERT INTO preference_pairs (env_1_id, env_2_id, split, status) 
        VALUES (?, ?, ?, ?)
    """, (id_1, id_2, split, "pending"))
    
# clean up
conn.commit()
conn.close()
plt.close(fig)

print(f"Done! Saved to {base_dir}/data/")



