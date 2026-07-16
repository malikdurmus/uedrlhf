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

@partial(jax.jit, static_argnames=("view_size", "tile_size", "sight_highlight"))
def jax_render(
    grid: jax.Array,
    agent: AgentState,
    tile_cache: jax.Array,
    agent_tile_cache: jax.Array,
    view_size: int = 7,
    tile_size: int = 32,
    sight_highlight: bool = True,
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
        sight_highlight: Whether to draw the agent's FOV highlight (default True).

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

    # 4. Apply FOV highlight (only if enabled)
    if sight_highlight:
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


def get_jaxnav_rasterizer(
    img_height=120,
    img_width=120,
    map_height=11.0,
    map_width=11.0,
    cell_size=1.0,
):
    # Calculate world units per pixel
    px_w = map_width / img_width
    px_h = map_height / img_height
    
    # Precompute coordinate grids (centered on pixels)
    px_idx = jnp.arange(img_width)
    py_idx = jnp.arange(img_height)
    px_grid, py_grid = jnp.meshgrid(px_idx, py_idx)
    
    # Correct coordinate mapping: 
    # px=0 should be left edge (0.0), px=img_width should be right (map_width)
    world_x = (px_grid + 0.5) * px_w
    world_y = (img_height - 0.5 - py_grid) * px_h
    world_xy = jnp.stack([world_x, world_y], axis=-1)

    def sdf_box(p, half_extents):
        """
        p: (H, W, 2)
        half_extents: (2,)
        """
        # Standard AABB SDF
        d = jnp.abs(p) - half_extents
        external = jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)
        internal = jnp.minimum(jnp.maximum(d[..., 0], d[..., 1]), 0.0)
        return external + internal

    def dist_to_segment(p, a, b):
        """
        p: (H, W, 2)
        a: (2,)
        b: (2,)
        """
        pa = p - a
        ba = b - a
        # We use jnp.sum(... axis=-1) instead of jnp.dot to handle the (H, W, 2) shape
        num = jnp.sum(pa * ba, axis=-1)
        den = jnp.sum(ba * ba)
        
        h = jnp.clip(num / jnp.maximum(den, 1e-7), 0.0, 1.0)
        
        # h is (H, W). We add a trailing dimension [..., None] to make it (H, W, 1)
        # This allows it to broadcast correctly with ba which is (2,)
        return jnp.linalg.norm(pa - ba * h[..., None], axis=-1)

    def render_aa(dist, thickness, blur=1.0):
        """
        Anti-aliased rendering helper.
        blur: width in pixels to smooth the edge.
        """
        # thickness is in world units. We convert blur to world units.
        edge_width = blur * px_w 
        return jnp.clip(0.5 - (dist - thickness) / edge_width, 0.0, 1.0)

    def render_state(env_state):
        # 1. Background / Walls
        # Standardize grid indices
        grid_col = jnp.clip(jnp.floor(world_x / cell_size).astype(jnp.int32), 0, env_state.map_data.shape[1] - 1)
        grid_row = jnp.clip(jnp.floor(world_y / cell_size).astype(jnp.int32), 0, env_state.map_data.shape[0] - 1)
        
        is_wall = env_state.map_data[grid_row, grid_col]
        # Map: 1 (wall) -> Black (0,0,0), 0 (free) -> White (1,1,1)
        wall_color = jnp.array([0.0, 0.0, 0.0])  # Black
        free_color = jnp.array([1.0, 1.0, 1.0])  # White    

        img = jnp.where(is_wall[..., None], wall_color, free_color)

        def render_agent(img_carry, agent_idx):
            pos = env_state.pos[agent_idx]
            theta = env_state.theta[agent_idx]
            goal = env_state.goal[agent_idx]
            done = env_state.done[agent_idx]
            
            # --- Layer: Goal Line (Transparent Black) ---
            # Increase thickness slightly for low-res (at least 1 pixel)
            line_thickness = jnp.maximum(0.02, px_w * 0.5) 
            d_line = dist_to_segment(world_xy, pos, goal)
            alpha_line = render_aa(d_line, line_thickness, blur=1.5) * 0.3 # 0.3 alpha
            img_curr = img_carry * (1.0 - alpha_line[..., None])

            # --- Layer: Goal Marker (Green +) ---
            p_goal = world_xy - goal
            m_size, m_thick = 0.25, jnp.maximum(0.06, px_w * 0.8)
            sdf_g = jnp.minimum(sdf_box(p_goal, jnp.array([m_size, m_thick])), 
                                sdf_box(p_goal, jnp.array([m_thick, m_size])))
            alpha_goal = render_aa(sdf_g, 0.0, blur=1.0)
            goal_color = jnp.array([0.0, 0.6, 0.0])
            img_curr = img_curr * (1.0 - alpha_goal[..., None]) + goal_color * alpha_goal[..., None]

            # --- Layer: Agent Body (Red/Black Box) ---
            p_agent = world_xy - pos
            # Rotate points opposite to theta to align with AABB SDF
            cos_t, sin_t = jnp.cos(-theta), jnp.sin(-theta)
            rot_inv = jnp.array([[cos_t, -sin_t], [sin_t, cos_t]])
            p_agent_rot = jnp.matmul(p_agent, rot_inv.T)
            
            sdf_a = sdf_box(p_agent_rot, jnp.array([0.25, 0.25]))
            alpha_agent = render_aa(sdf_a, 0.0, blur=1.0)
            agent_color = jnp.where(done, jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]))
            img_curr = img_curr * (1.0 - alpha_agent[..., None]) + agent_color * alpha_agent[..., None]

            # --- Layer: Direction Line (Black) ---
            # Middle line logic (usually from center to front edge)
            dir_end = pos + jnp.array([jnp.cos(theta), jnp.sin(theta)]) * 0.25
            d_dir = dist_to_segment(world_xy, pos, dir_end)
            alpha_dir = render_aa(d_dir, jnp.maximum(0.03, px_w * 0.6), blur=1.0)
            img_curr = img_curr * (1.0 - alpha_dir[..., None])

            return img_curr, None

        num_agents = env_state.pos.shape[0]
        img_final, _ = jax.lax.scan(render_agent, img, jnp.arange(num_agents))
        return img_final

    return render_state