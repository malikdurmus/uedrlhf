import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

dataset_path = "data/difficulty/compiled_gemini_2_5_flash_clean"
print(f"Loading dataset from {dataset_path}")
dataset = load_from_disk(dataset_path)
print(f"Dataset size: {len(dataset)}")

env = JaxNav(num_agents=1)
dikstra_jit = jax.jit(env._map_obj.dikstra_path)

def analyze_image(img):
    # img is (64, 64, 3) float [0, 1]
    # 1. Recover the 13x13 map_data
    # world_x goes from 0 to 11.0
    # Map index 0 in map_data is x=0 in world
    map_data = np.ones((13, 13), dtype=bool) # default wall
    
    # Grid cell coordinates i, j in [0, 10] map to [1, 11] in map_data
    # px = (i + 0.5) * 64 / 11.0 - 0.5
    px_indices = [int(round((i + 0.5) * 64 / 11.0 - 0.5)) for i in range(11)]
    py_indices = [int(round((j + 0.5) * 64 / 11.0 - 0.5)) for j in range(11)]
    
    # j is y-axis. world_y = (64 - 0.5 - py) * (11/64)
    # wait, in my old rasterizer:
    # world_y = (img_height - 0.5 - py_grid) * px_h
    # So py_grid = img_height - 0.5 - world_y / px_h
    # For world_y = j + 0.5:
    # py_grid = 64 - 0.5 - (j + 0.5) * 64 / 11.0
    py_indices = [int(round(64 - 0.5 - (j + 0.5) * 64 / 11.0)) for j in range(11)]
    
    for i, px in enumerate(px_indices):
        for j, py in enumerate(py_indices):
            # Sample pixel
            val = img[py, px]
            # Wall if all channels < 0.1
            is_wall = np.all(val < 0.1)
            map_data[j+1, i+1] = is_wall
            
    # 2. Find Agent (Red)
    # R > 0.5, G < 0.3, B < 0.3
    agent_mask = (img[:, :, 0] > 0.5) & (img[:, :, 1] < 0.3) & (img[:, :, 2] < 0.3)
    if not np.any(agent_mask):
        return None # Failed to find agent
    py_a, px_a = np.where(agent_mask)
    px_a_mean = np.mean(px_a)
    py_a_mean = np.mean(py_a)
    
    world_x_a = (px_a_mean + 0.5) * (11.0 / 64.0)
    world_y_a = (64.0 - 0.5 - py_a_mean) * (11.0 / 64.0)
    # Shift by 1.0 because map_data has a 1-cell border
    agent_pos = jnp.array([world_x_a + 1.0, world_y_a + 1.0])
    
    # 3. Find Goal (Green)
    # G > 0.3, R < 0.3, B < 0.3
    goal_mask = (img[:, :, 1] > 0.3) & (img[:, :, 0] < 0.3) & (img[:, :, 2] < 0.3)
    if not np.any(goal_mask):
        return None # Failed to find goal
    py_g, px_g = np.where(goal_mask)
    px_g_mean = np.mean(px_g)
    py_g_mean = np.mean(py_g)
    
    world_x_g = (px_g_mean + 0.5) * (11.0 / 64.0)
    world_y_g = (64.0 - 0.5 - py_g_mean) * (11.0 / 64.0)
    goal_pos = jnp.array([world_x_g + 1.0, world_y_g + 1.0])
    
    return map_data, agent_pos, goal_pos

unsolvable_count = 0
failed_count = 0

print("Scanning dataset to check solvability...")
# Check first 500 images as a representative sample or we can do all 
for idx in tqdm(range(min(1000, len(dataset)))):
    img = np.array(dataset[idx]["env_0_image"])
    res = analyze_image(img)
    if res is None:
        failed_count += 1
        continue
    
    map_data, agent_pos, goal_pos = res
    valid, _ = dikstra_jit(jnp.array(map_data), agent_pos, goal_pos)
    
    if not valid:
        unsolvable_count += 1

print(f"\nResults over {min(1000, len(dataset))} env_0 images:")
print(f"Failed to parse: {failed_count}")
print(f"Unsolvable: {unsolvable_count}")
