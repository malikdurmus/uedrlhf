import pathlib
import time
import jax
from jaxmarl.environments.jaxnav import JaxNav
import matplotlib.pyplot as plt


base = pathlib.Path(__file__).resolve()
base = base.parent
img_dir = base / "someimages"

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
MAP_PARAMS = {
    "map_size": (11, 11),
    "min_fill": 0.0,
    "max_fill": 0.6,
    "valid_path_check" : True
}



env = JaxNav(num_agents=1,map_params= MAP_PARAMS) 

seed = int(time.time())
print(f"Using PRNG seed: {seed}")
rng = jax.random.PRNGKey(seed)  # Dynamic seed for unique envs across runs
reset_rng = jax.random.split(rng, 50)
obsv, env_state = jax.vmap(env.reset)(reset_rng)
rng , _rng = jax.random.split(rng)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))


for i in range(50):
    # save image
    state = jax.tree_util.tree_map(lambda x: x[i], env_state)

    ax.clear() 
    env.init_render(ax, state, lidar=False, agent=True, goal=True)
    plt.savefig(img_dir / f"{i}.png")
