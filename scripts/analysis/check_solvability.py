import sqlite3
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
import argparse
import sys

def check_db_solvability(db_path: Path, arrays_dir: Path):
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
        
    print(f"\n--- Checking {db_path.name} ---")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
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
    
    print(f"Found {len(env_ids)} unique environments across all tables.")
    if not env_ids:
        return
    
    env = JaxNav(num_agents=1)
    
    # We vmap dijkstra_path to avoid the shape issue (it might expect batched inputs)
    # Actually, we can batch process all envs for speed
    
    @jax.jit
    def batch_is_solvable(map_data, start, goal):
        valid, _ = jax.vmap(env._map_obj.dikstra_path)(map_data, start, goal)
        return valid

    solvable_count = 0
    missing_count = 0
    total = len(env_ids)
    
    # Process in batches
    batch_size = 1000
    env_ids_list = list(env_ids)
    
    for i in range(0, total, batch_size):
        batch_ids = env_ids_list[i:i+batch_size]
        map_datas = []
        starts = []
        goals = []
        
        valid_batch_ids = []
        
        for env_id in batch_ids:
            arr_path = arrays_dir / f"{env_id}.pkl"
            if not arr_path.exists():
                missing_count += 1
                continue
                
            with open(arr_path, "rb") as f:
                state = pickle.load(f)
                
            map_datas.append(state.map_data)
            starts.append(state.pos[0])
            goals.append(state.goal[0])
            valid_batch_ids.append(env_id)
            
        if not map_datas:
            continue
            
        map_datas_np = jnp.array(map_datas)
        starts_np = jnp.array(starts)
        goals_np = jnp.array(goals)
        
        valid_mask = batch_is_solvable(map_datas_np, starts_np, goals_np)
        solvable_count += int(jnp.sum(valid_mask))
        
        print(f"Processed min({i+batch_size}, {total})/{total} environments...")
            
    unsolvable_count = total - solvable_count - missing_count
    print(f"\nResults for {db_path.name}:")
    print(f"  Total envs:  {total}")
    print(f"  Solvable:    {solvable_count} ({(solvable_count/(total-missing_count))*100 if total-missing_count > 0 else 0:.1f}%)")
    print(f"  Unsolvable:  {unsolvable_count} ({(unsolvable_count/(total-missing_count))*100 if total-missing_count > 0 else 0:.1f}%)")
    print(f"  Missing:     {missing_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check environment solvability for a database.")
    parser.add_argument("--db", type=str, help="Path to database", required=False)
    parser.add_argument("--arrays", type=str, help="Path to arrays directory", required=False)
    args = parser.parse_args()
    
    if args.db and args.arrays:
        check_db_solvability(Path(args.db), Path(args.arrays))
    else:
        base = Path(__file__).parent.parent.parent
        # Check both databases by default
        check_db_solvability(
            base / "data" / "difficulty" / "feedback.db",
            base / "data" / "difficulty" / "arrays"
        )
        check_db_solvability(
            base / "data_v2" / "jaxnav" / "difficulty" / "database_v2.db",
            base / "data_v2" / "jaxnav" / "difficulty" / "arrays"
        )
