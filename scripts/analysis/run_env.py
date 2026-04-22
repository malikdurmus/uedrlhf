"""Quick one-off script to generate and render a single JaxNav environment.

Usage:
    python scripts/analysis/run_env.py
"""

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
import jax
import matplotlib.pyplot as plt

jax.config.update("jax_disable_jit", True)

import time

seed = int(time.time())
print(f"Using PRNG seed: {seed}")

MAP_PARAMS = {
    "map_size": (11, 11),
    "min_fill": 0.0,
    "max_fill": 0.3,
    "valid_path_check": True
}

env = JaxNav(num_agents=1, map_params=MAP_PARAMS)

rng = jax.random.PRNGKey(seed)
reset_rng, rng = jax.random.split(rng)

obsv, env_state = env.reset(rng)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

env.init_render(ax, env_state, lidar=False, agent=True, goal=True)
plt.show()
