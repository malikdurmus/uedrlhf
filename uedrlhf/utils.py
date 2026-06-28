"""Shared utility functions for the uedrlhf project."""

import base64
import pickle
import numpy as np
import PIL.Image
from pathlib import Path
from pydantic import BaseModel, Field
import jax
import jax.numpy as jnp


RESIZE = (64, 64)


def choice_to_preference(choice: str) -> float:
    """Map LLM choice string to preference score.

    Returns:
        0.0 — Image A (left) preferred
        1.0 — Image B (right) preferred
        0.5 — Tie
    """
    c = choice.strip().upper()
    if c == "A":
        return 0.0
    elif c == "B":
        return 1.0
    return 0.5


def load_image_as_nparray(file_path: Path, size: tuple = RESIZE) -> np.ndarray:
    """Load an image, resize it, and return as a float32 numpy array."""
    img = PIL.Image.open(file_path).resize(size).convert("RGB")
    return np.array(img).astype(np.float32)


def load_array(file_path: Path):
    """Load a pickled array from disk."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def state_to_dict(state_obj) -> dict:
    """Convert a JaxNav State PyTree into a plain dictionary of NumPy arrays."""
    return {
        "pos": np.array(state_obj.pos),
        "theta": np.array(state_obj.theta),
        "vel": np.array(state_obj.vel),
        "done": np.array(state_obj.done),
        "term": np.array(state_obj.term),
        "goal_reached": np.array(state_obj.goal_reached),
        "move_term": np.array(state_obj.move_term),
        "step": np.array(state_obj.step),
        "ep_done": np.array(state_obj.ep_done),
        "goal": np.array(state_obj.goal),
        "map_data": np.array(state_obj.map_data),
        "rew_lambda": np.array(state_obj.rew_lambda),
    }


def img_to_base64(path: Path) -> str:
    """Encode an image file to a base64 string."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


class LLMFeedbackResponse(BaseModel):
    """Pydantic model for structured Gemini JSON output."""
    choice: str = Field(description="A, B, or TIE")
    x_pct: float = Field(description="x% position (0-100) in the chosen image of the most influential feature")
    y_pct: float = Field(description="y% position (0-100) in the chosen image of the most influential feature")
    reasoning: str = Field(description="Brief one-sentence explanation of the decision")






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