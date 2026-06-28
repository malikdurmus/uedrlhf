"""
Generate 100 XLand-MiniGrid environments and save rendered images
using the JAX-jittable renderer.
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import xminigrid
from uedrlhf.jax_renderer import jax_render, build_tile_caches

# ---- Config ----
NUM_ENVS = 100
TILE_SIZE = 32
OUTPUT_DIR = "/home/d/durmusy/Desktop/GIT/new/uedrlhf/testing_renderer"
ENV_NAME = "XLand-MiniGrid-R4-9x9"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Build tile caches once ----
print("Building tile caches...")
tile_cache, agent_tile_cache = build_tile_caches(tile_size=TILE_SIZE)

# ---- Create environment ----
env, env_params = xminigrid.make(ENV_NAME)
print(f"Environment: {ENV_NAME}")
print(f"Grid size: {env_params.height}x{env_params.width}")

# ---- Generate 100 environments with different random seeds ----
key = jax.random.key(42)
reset_keys = jax.random.split(key, NUM_ENVS)

# Vectorized reset: create all 100 environments at once
timesteps = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_keys)

# ---- Render all 100 environments in one vectorized call ----
print(f"Rendering {NUM_ENVS} environments...")
render_fn = jax.jit(jax.vmap(
    lambda grid, agent: jax_render(
        grid, agent, tile_cache, agent_tile_cache,
        view_size=env_params.view_size, tile_size=TILE_SIZE,
    )
))

imgs = render_fn(timesteps.state.grid, timesteps.state.agent)
imgs_np = np.asarray(imgs)

print(f"Rendered batch shape: {imgs_np.shape}")  # (100, H*ts, W*ts, 3)

# ---- Save each image ----
for i in range(NUM_ENVS):
    img = Image.fromarray(imgs_np[i])
    path = os.path.join(OUTPUT_DIR, f"env_{i:03d}.png")
    img.save(path)

print(f"Saved {NUM_ENVS} images to {OUTPUT_DIR}/")
