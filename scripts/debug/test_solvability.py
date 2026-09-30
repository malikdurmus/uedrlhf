import pickle
import jax.numpy as jnp
from pathlib import Path
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

arr_dir = Path("data_v2/jaxnav/difficulty/arrays")
first_pkl = next(arr_dir.glob("*.pkl"))

with open(first_pkl, "rb") as f:
    state = pickle.load(f)

print(type(state))
print(hasattr(state, "map_data"))
print(state.map_data.shape)
print(state.pos.shape)
print(state.goal.shape)

env = JaxNav(num_agents=1)
valid, _ = env._map_obj.dikstra_path(jnp.array(state.map_data), jnp.array(state.pos[0]), jnp.array(state.goal[0]))
print("Solvable:", valid)
