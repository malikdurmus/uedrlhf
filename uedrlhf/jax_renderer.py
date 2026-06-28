"""
JAX-jittable renderer for XLand-MiniGrid environments.

Uses a pre-computed tile cache (built once with the original numpy renderer)
and pure JAX operations for fully jittable full-grid rendering.

Usage:
    from uedrlhf.jax_renderer import jax_render, build_tile_caches

    # Build caches once (uses numpy, not jittable)
    tile_cache, agent_tile_cache = build_tile_caches(tile_size=32)

    # Render inside jit
    img = jax.jit(jax_render)(
        grid=timestep.state.grid,
        agent=timestep.state.agent,
        tile_cache=tile_cache,
        agent_tile_cache=agent_tile_cache,
        view_size=7,
        tile_size=32,
    )
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from xminigrid.core.constants import NUM_COLORS, NUM_TILES
from xminigrid.rendering.rgb_render import render_tile
from xminigrid.types import AgentState


# ---------------------------------------------------------------------------
# Cache construction (runs once at init time, uses the original numpy renderer)
# ---------------------------------------------------------------------------

def build_tile_caches(tile_size: int = 32) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pre-render every (tile, color) and (tile, color, direction) combination.

    Returns:
        tile_cache: jnp.ndarray of shape (NUM_TILES * NUM_COLORS, tile_size, tile_size, 3)
            Indexed by flat_idx = tile_id * NUM_COLORS + color_id.
        agent_tile_cache: jnp.ndarray of shape (NUM_TILES * NUM_COLORS, 4, tile_size, tile_size, 3)
            Same flat_idx, second axis is agent direction (0-3).
    """
    n = NUM_TILES * NUM_COLORS
    tile_cache = np.zeros((n, tile_size, tile_size, 3), dtype=np.uint8)
    agent_tile_cache = np.zeros((n, 4, tile_size, tile_size, 3), dtype=np.uint8)

    for tile_id in range(NUM_TILES):
        for color_id in range(NUM_COLORS):
            flat_idx = tile_id * NUM_COLORS + color_id
            tile_tuple = (tile_id, color_id)

            # Base tile (no agent)
            tile_cache[flat_idx] = render_tile(
                tile=tile_tuple,
                agent_direction=None,
                highlight=False,
                tile_size=tile_size,
            )

            # Tile with agent in each of the 4 directions
            for direction in range(4):
                agent_tile_cache[flat_idx, direction] = render_tile(
                    tile=tile_tuple,
                    agent_direction=direction,
                    highlight=False,
                    tile_size=tile_size,
                )

    return jnp.asarray(tile_cache), jnp.asarray(agent_tile_cache)


# ---------------------------------------------------------------------------
# Jittable highlight mask
# ---------------------------------------------------------------------------

def _highlight_mask_jax(
    grid_h: int,
    grid_w: int,
    agent_position: jax.Array,
    agent_direction: jax.Array,
    view_size: int,
) -> jax.Array:
    """Compute the FOV highlight mask in pure JAX (jittable).

    Returns a bool array of shape (grid_h, grid_w).
    """
    padded_h = grid_h + 2 * view_size
    padded_w = grid_w + 2 * view_size

    # Agent position in the padded coordinate system
    agent_y = agent_position[0] + view_size
    agent_x = agent_position[1] + view_size

    # Compute the top-left corner of the view window depending on direction
    # Direction 0 (Up):    y = agent_y - view_size + 1, x = agent_x - view_size // 2
    # Direction 1 (Right): y = agent_y - view_size // 2, x = agent_x
    # Direction 2 (Down):  y = agent_y,                  x = agent_x - view_size // 2
    # Direction 3 (Left):  y = agent_y - view_size // 2, x = agent_x - view_size + 1
    half = view_size // 2

    def dir0(_):
        return agent_y - view_size + 1, agent_x - half

    def dir1(_):
        return agent_y - half, agent_x

    def dir2(_):
        return agent_y, agent_x - half

    def dir3(_):
        return agent_y - half, agent_x - view_size + 1

    vy, vx = jax.lax.switch(agent_direction, [dir0, dir1, dir2, dir3], None)

    # Build mask on the padded grid
    ys = jnp.arange(padded_h)
    xs = jnp.arange(padded_w)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")

    mask = (yy >= vy) & (yy < vy + view_size) & (xx >= vx) & (xx < vx + view_size)

    # Crop to original grid size
    mask = mask[view_size : view_size + grid_h, view_size : view_size + grid_w]
    return mask


# ---------------------------------------------------------------------------
# Main jittable render function
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("view_size", "tile_size"))
def jax_render(
    grid: jax.Array,
    agent: AgentState,
    tile_cache: jax.Array,
    agent_tile_cache: jax.Array,
    view_size: int = 7,
    tile_size: int = 32,
) -> jax.Array:
    """Render the full grid as an RGB image, fully jittable.

    Args:
        grid: jax.Array of shape (H, W, 2) — the environment grid.
            grid[:,:,0] = tile type, grid[:,:,1] = color.
        agent: AgentState with .position (2,) and .direction scalar.
        tile_cache: Pre-built cache from build_tile_caches().
        agent_tile_cache: Pre-built cache from build_tile_caches().
        view_size: Agent view size for FOV highlight (default 7).
        tile_size: Pixel size per tile (must match cache, default 32).

    Returns:
        img: jax.Array of shape (H*tile_size, W*tile_size, 3), dtype uint8.
    """
    grid_h, grid_w = grid.shape[0], grid.shape[1]

    # 1. Compute flat tile indices: tile_id * NUM_COLORS + color_id
    flat_idxs = grid[:, :, 0].astype(jnp.int32) * NUM_COLORS + grid[:, :, 1].astype(jnp.int32)

    # 2. Look up all tiles from cache → (H, W, tile_size, tile_size, 3)
    rendered = jnp.take(tile_cache, flat_idxs, axis=0)

    # 3. Overlay the agent tile at agent.position
    agent_y = agent.position[0]
    agent_x = agent.position[1]
    agent_flat_idx = flat_idxs[agent_y, agent_x]
    agent_tile = agent_tile_cache[agent_flat_idx, agent.direction.astype(jnp.int32)]
    rendered = rendered.at[agent_y, agent_x].set(agent_tile)

    # 4. Apply FOV highlight
    highlight_mask = _highlight_mask_jax(
        grid_h, grid_w, agent.position, agent.direction.astype(jnp.int32), view_size
    )
    # Expand mask to match tile dimensions: (H, W) → (H, W, 1, 1, 1)
    mask_expanded = highlight_mask[:, :, None, None, None]

    # Blend: img + alpha * (white - img), matching the original highlight_img
    alpha = 0.2
    blended = rendered.astype(jnp.float32) + alpha * (255.0 - rendered.astype(jnp.float32))
    blended = jnp.clip(blended, 0, 255).astype(jnp.uint8)

    rendered = jnp.where(mask_expanded, blended, rendered)

    # 5. Reshape (H, W, ts, ts, 3) → (H*ts, W*ts, 3)
    img = rendered.transpose((0, 2, 1, 3, 4))  # (H, ts, W, ts, 3)
    img = img.reshape(grid_h * tile_size, grid_w * tile_size, 3)

    return img
