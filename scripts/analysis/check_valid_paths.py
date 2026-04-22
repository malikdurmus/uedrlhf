"""Generate and visualize environments to verify valid path checking.

Usage:
    python scripts/analysis/check_valid_paths.py
"""

import time
import jax
import matplotlib.pyplot as plt
from jaxmarl.environments.jaxnav import JaxNav

from uedrlhf.config import BASE_DIR


def main():
    output_dir = BASE_DIR / "outputs" / "plots" / "valid_path_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    MAP_PARAMS = {
        "map_size": (11, 11),
        "min_fill": 0.0,
        "max_fill": 0.6,
        "valid_path_check": True
    }

    env = JaxNav(num_agents=1, map_params=MAP_PARAMS)

    seed = int(time.time())
    print(f"Using PRNG seed: {seed}")
    rng = jax.random.PRNGKey(seed)
    reset_rng = jax.random.split(rng, 50)
    obsv, env_state = jax.vmap(env.reset)(reset_rng)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for i in range(50):
        state = jax.tree_util.tree_map(lambda x: x[i], env_state)
        ax.clear()
        env.init_render(ax, state, lidar=False, agent=True, goal=True)
        plt.savefig(output_dir / f"{i}.png")

    plt.close(fig)
    print(f"Saved 50 images to {output_dir}")


if __name__ == "__main__":
    main()
