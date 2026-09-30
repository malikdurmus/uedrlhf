import xminigrid
from xminigrid.environment import Environment, EnvParams
from xminigrid.wrappers import DirectionObservationWrapper, GymAutoResetWrapper
import jax

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import timeit
import imageio
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm

from uedrlhf.jax_renderer import jax_render, build_tile_caches
from uedrlhf.config import BASE_DIR


def save_img(img, output_path, dpi=32):
    plt.figure(dpi=dpi)
    plt.axis('off')
    plt.imshow(img)
    # Save the file with tight bounds and no white padding
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close() # Closes the figure to free up memory


def show_img(img, dpi=32):
    plt.figure(dpi=dpi)
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(img)


# ---- Setup environment ----
key = jax.random.key(0)
key, reset_key = jax.random.split(key)

# to list available environments: xminigrid.registered_environments()
env, env_params = xminigrid.make("MiniGrid-Empty-8x8")
print("Observation shape:", env.observation_shape(env_params))
print("Num actions:", env.num_actions(env_params))

# fully jit-compatible step and reset methods
timestep = jax.jit(env.reset)(env_params, reset_key)
timestep = jax.jit(env.step)(env_params, timestep, action=0)

print("TimeStep shapes:", jtu.tree_map(jnp.shape, timestep))

# ---- Original (numpy) render ----
img_original = env.render(env_params, timestep)
save_img(img_original, str(BASE_DIR / "imggg1.png"), dpi=64)

# ---- JAX jittable render ----
TILE_SIZE = 32
tile_cache, agent_tile_cache = build_tile_caches(tile_size=TILE_SIZE)

# Run the jittable renderer
img_jax = jax_render(
    grid=timestep.state.grid,
    agent=timestep.state.agent,
    tile_cache=tile_cache,
    agent_tile_cache=agent_tile_cache,
    view_size=env_params.view_size,
    tile_size=TILE_SIZE,
)
img_jax_np = np.asarray(img_jax)

save_img(img_jax_np, str(BASE_DIR / "imggg1_jax.png"), dpi=64)

# ---- Compare the two renders ----
print("\n=== Renderer Comparison ===")
print(f"Original shape: {img_original.shape}, dtype: {img_original.dtype}")
print(f"JAX shape:      {img_jax_np.shape}, dtype: {img_jax_np.dtype}")

diff = np.abs(img_original.astype(np.int16) - img_jax_np.astype(np.int16))
max_diff = diff.max()
mean_diff = diff.mean()
num_mismatch = (diff > 0).sum()
total_pixels = diff.size

print(f"Max pixel difference:  {max_diff}")
print(f"Mean pixel difference: {mean_diff:.4f}")
print(f"Mismatched values:     {num_mismatch} / {total_pixels} ({100*num_mismatch/total_pixels:.2f}%)")

if max_diff <= 1:
    print("✅ PASS: Renders match within ±1 tolerance (float→uint8 rounding).")
elif max_diff == 0:
    print("✅ PASS: Renders match exactly!")
else:
    print(f"❌ FAIL: Max difference is {max_diff}, which exceeds ±1 tolerance.")

# Save side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img_original)
axes[0].set_title("Original (numpy)")
axes[0].axis("off")
axes[1].imshow(img_jax_np)
axes[1].set_title("JAX jittable")
axes[1].axis("off")
axes[2].imshow(diff.astype(np.uint8) * 50)  # amplify differences for visibility
axes[2].set_title(f"Difference (×50), max={max_diff}")
axes[2].axis("off")
plt.tight_layout()
plt.savefig(str(BASE_DIR / "render_comparison.png"), dpi=150)
plt.close()
print("Saved comparison image to render_comparison.png")

# ---- Verify jit compilation works ----
print("\n=== JIT Compilation Test ===")
jitted_render = jax.jit(
    jax_render,
    static_argnames=("view_size", "tile_size"),
)

# First call triggers compilation
img_jit = jitted_render(
    grid=timestep.state.grid,
    agent=timestep.state.agent,
    tile_cache=tile_cache,
    agent_tile_cache=agent_tile_cache,
    view_size=env_params.view_size,
    tile_size=TILE_SIZE,
)
print(f"JIT render shape: {img_jit.shape}, dtype: {img_jit.dtype}")
assert jnp.array_equal(img_jax, img_jit), "JIT output differs from eager output!"
print("✅ JIT compilation and execution successful, output matches eager mode.")

# ---- Speed comparison ----
print("\n=== Speed Comparison ===")
n_repeats = 50

# Time the original renderer
t_original = timeit.timeit(
    lambda: env.render(env_params, timestep),
    number=n_repeats,
)
print(f"Original renderer: {1000*t_original/n_repeats:.2f} ms/render")

# Time the jitted renderer (already compiled)
_ = jitted_render(
    grid=timestep.state.grid,
    agent=timestep.state.agent,
    tile_cache=tile_cache,
    agent_tile_cache=agent_tile_cache,
    view_size=env_params.view_size,
    tile_size=TILE_SIZE,
).block_until_ready()

t_jax = timeit.timeit(
    lambda: jitted_render(
        grid=timestep.state.grid,
        agent=timestep.state.agent,
        tile_cache=tile_cache,
        agent_tile_cache=agent_tile_cache,
        view_size=env_params.view_size,
        tile_size=TILE_SIZE,
    ).block_until_ready(),
    number=n_repeats,
)
print(f"JAX jitted renderer: {1000*t_jax/n_repeats:.2f} ms/render")
print(f"Speedup: {t_original/t_jax:.1f}x")
