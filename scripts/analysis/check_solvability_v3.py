import sqlite3
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
import matplotlib.pyplot as plt
import argparse
from uedrlhf.config import BASE_DIR

def get_env_ids_from_db(db_path: Path):
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall() if t[0] != 'sqlite_sequence']
    env_ids = set()
    for table in tables:
        try:
            cursor.execute(f"SELECT env_1_id, env_2_id FROM '{table}'")
            for row in cursor.fetchall():
                if row[0]: env_ids.add(row[0])
                if row[1]: env_ids.add(row[1])
        except sqlite3.OperationalError:
            pass
    conn.close()
    return env_ids

def check_solvability(env_ids, arrays_dir: Path):
    if not env_ids:
        return 0, 0, 0
    env = JaxNav(num_agents=1)
    
    @jax.jit
    def batch_is_solvable(map_data, start, goal):
        valid, _ = jax.vmap(env._map_obj.dikstra_path)(map_data, start, goal)
        return valid

    solvable_count = 0
    missing_count = 0
    total = len(env_ids)
    
    batch_size = 250
    env_ids_list = list(env_ids)
    
    for i in range(0, total, batch_size):
        batch_ids = env_ids_list[i:i+batch_size]
        map_datas = []
        starts = []
        goals = []
        
        for env_id in batch_ids:
            arr_path = arrays_dir / f"{env_id}.pkl"
            if not arr_path.exists():
                missing_count += 1
                continue
                
            with open(arr_path, "rb") as f:
                state = pickle.load(f)
                
            if isinstance(state, dict):
                map_datas.append(state["map_data"])
                starts.append(state["pos"][0])
                goals.append(state["goal"][0])
            else:
                map_datas.append(state.map_data)
                starts.append(state.pos[0])
                goals.append(state.goal[0])
            
        if not map_datas:
            continue
            
        map_datas_np = jnp.array(map_datas)
        starts_np = jnp.array(starts)
        goals_np = jnp.array(goals)
        
        valid_mask = batch_is_solvable(map_datas_np, starts_np, goals_np)
        solvable_count += int(jnp.sum(valid_mask))
        print(f"  Processed {min(i+batch_size, total)}/{total}...")
            
    unsolvable_count = total - solvable_count - missing_count
    return solvable_count, unsolvable_count, missing_count

def main():
    base_dir = BASE_DIR / "data_v3/jaxnav"
    datasets = [d for d in base_dir.iterdir() if d.is_dir()]
    
    results = {}
    
    for dataset_dir in datasets:
        db_path = dataset_dir / "database_v2.db"
        arrays_dir = dataset_dir / "arrays"
        
        if db_path.exists() and arrays_dir.exists():
            print(f"\nChecking dataset: {dataset_dir.name}")
            env_ids = get_env_ids_from_db(db_path)
            if not env_ids:
                print(f"  No environments found in {db_path.name}")
                continue
                
            solvable, unsolvable, missing = check_solvability(env_ids, arrays_dir)
            total = solvable + unsolvable + missing
            
            print(f"  Solvable: {solvable} ({(solvable/(total-missing))*100 if total-missing > 0 else 0:.1f}%)")
            print(f"  Unsolvable: {unsolvable} ({(unsolvable/(total-missing))*100 if total-missing > 0 else 0:.1f}%)")
            print(f"  Missing: {missing}")
            
            results[dataset_dir.name] = {
                'solvable': solvable,
                'unsolvable': unsolvable,
                'missing': missing,
                'total': total
            }
            
    # Create plot
    if results:
        labels = list(results.keys())
        solvable_counts = [results[l]['solvable'] for l in labels]
        unsolvable_counts = [results[l]['unsolvable'] for l in labels]
        
        x = range(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar([pos - width/2 for pos in x], solvable_counts, width, label='Solvable', color='green')
        rects2 = ax.bar([pos + width/2 for pos in x], unsolvable_counts, width, label='Unsolvable', color='red')
        
        ax.set_ylabel('Number of Levels')
        ax.set_title('Distribution of Solvable vs Unsolvable Levels in data_v3/jaxnav')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        
        fig.tight_layout()
        out_path = BASE_DIR / "data_v3/solvability_distribution.png"
        plt.savefig(out_path)
        print(f"\nPlot saved to {out_path}")
    else:
        print("\nNo valid datasets found to plot.")

if __name__ == "__main__":
    main()
