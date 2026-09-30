"""
Dynamically generate initial states for all 44 registered XLand-MiniGrid 
environments and render them using the highly optimized JAX-jittable renderer.
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import xminigrid
from uedrlhf.jax_renderer import jax_render, build_tile_caches

# ---- Config ----
NUM_ENVS_PER_TYPE = 100  # Generates 100 variations per environment layout
TILE_SIZE = 32
BASE_OUTPUT_DIR = "env_baselines"

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ---- Build tile caches once ----
print("Building global JAX tile caches...")
tile_cache, agent_tile_cache = build_tile_caches(tile_size=TILE_SIZE)

# ---- Extract all 44 target environments dynamically ----
env_names = xminigrid.registered_environments()

# Base JAX PRNG Key split-point
base_key = jax.random.PRNGKey(42)

for idx, env_name in enumerate(env_names):
    print(f"\n[{idx+1}/{len(env_names)}] Processing: {env_name}")
    
    try:
        # Create environment and unpack its specific structural specifications
        env, env_params = xminigrid.make(env_name)
        
        # Prepare directory for this specific variant type
        clean_env_folder = env_name.replace("-", "_").lower()
        env_output_dir = os.path.join(BASE_OUTPUT_DIR, clean_env_folder)
        os.makedirs(env_output_dir, exist_ok=True)
        
        # Split base keys to isolate this layout's generation seeds
        base_key, split_key = jax.random.split(base_key)
        reset_keys = jax.random.split(split_key, NUM_ENVS_PER_TYPE)
        
        # 1. Vectorized Reset: Generate all 100 instances in parallel on GPU
        timesteps = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_keys)
        
        # 2. Vectorized Render Function compilation definition
        render_fn = jax.jit(jax.vmap(
            lambda grid, agent: jax_render(
                grid, agent, tile_cache, agent_tile_cache,
                view_size=env_params.view_size, tile_size=TILE_SIZE,
            )
        ))
        
        # Run parallelized render pass
        imgs = render_fn(timesteps.state.grid, timesteps.state.agent)
        imgs_np = np.asarray(imgs)
        
        # 3. Fast-Save the generated batch array directly onto storage disks
        for i in range(NUM_ENVS_PER_TYPE):
            img = Image.fromarray(imgs_np[i])
            path = os.path.join(env_output_dir, f"seed_{i:03d}.png")
            img.save(path)
            
        print(f"   Successfully saved {NUM_ENVS_PER_TYPE} images to {env_output_dir}/")
        
    except Exception as e:
        print(f"❌ Critical pipeline compilation error on {env_name}: {e}")

print(f"\n🎉 All environments processed! Outputs saved under: {BASE_OUTPUT_DIR}/")
