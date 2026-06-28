import pickle
import jax.numpy as jnp

# Load your specific file
with open('data_v2/difficulty/arrays/train_340_B_55f32276.pkl', 'rb') as f:
    data = pickle.load(f)

# 1. Access attributes using DOT notation, not brackets
pos = data.pos[0]   # [x, y]
goal = data.goal[0] # [x, y]

# 2. Calculate the math
true_dist = jnp.linalg.norm(pos - goal)

print(f"Agent Position: {pos}")
print(f"Goal Position:  {goal}")
print(f"True Distance:  {true_dist:.4f}")

# 3. Decision Logic
# In JaxNav, if the goal is visible, the relative observation 
# matches the distance. If blocked, it returns the distance to the wall.
print("\n--- VISIBILITY CHECK ---")
print("If you run one step and obs['agent_0'][:2] magnitude is")
print(f"LESS than {true_dist:.4f}, the agent is BLOCKED.")
print(f"EQUAL to {true_dist:.4f}, the agent SEES the goal.")

# Check the column where the agent and goal are (approx index 8)
column_index = 8
# Check Y-range from 4 (agent) to 9 (goal)
path_cells = data.map_data[4:10, column_index]

is_blocked = (path_cells == 1).any()
print(f"Walls in path: {path_cells}")
print(f"Is Goal Visible? {'NO' if is_blocked else 'YES'}")

import numpy as np

# Extract Agent 0's coordinates
p_start = np.array(data.pos[0]) # [x, y]
p_end = np.array(data.goal[0])  # [x, y]

# 1. Generate points along the vector from start to end
num_points = 100
t = np.linspace(0, 1, num_points).reshape(-1, 1) # Shape (100, 1)
path_points = p_start + t * (p_end - p_start)    # Shape (100, 2)

# 2. Extract X and Y for indexing
path_x = path_points[:, 0]
path_y = path_points[:, 1]

# 3. Convert to grid indices (assuming 1.0 unit = 1 cell)
grid_x = np.floor(path_x).astype(int)
grid_y = np.floor(path_y).astype(int)

# 4. Clip to map boundaries to avoid IndexErrors
grid_x = np.clip(grid_x, 0, data.map_data.shape[1] - 1)
grid_y = np.clip(grid_y, 0, data.map_data.shape[0] - 1)

# 5. Check if any grid cell along that line is a wall (1)
path_values = data.map_data[grid_y, grid_x]
hit_indices = np.where(path_values == 1)[0]

if len(hit_indices) > 0:
    idx = hit_indices[0]
    print(f"--- RESULT: BLOCKED ---")
    print(f"Hit wall at Grid: [x={grid_x[idx]}, y={grid_y[idx]}]")
    print(f"Hit at World Coords: ({path_x[idx]:.2f}, {path_y[idx]:.2f})")
else:
    print("--- RESULT: VISIBLE ---")
    print("No walls found on the direct line to the goal.")
