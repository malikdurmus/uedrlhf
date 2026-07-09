"""XLand-MiniGrid environment backend.

Wraps the XLand-MiniGrid environment with the JAX-jittable tile-cache
renderer from jax_renderer.py for fast GPU-accelerated rendering.
"""

import jax
import jax.numpy as jnp
import numpy as np

from uedrlhf.environments.base import EnvBackend


class XLandMiniGridBackend(EnvBackend):

    def __init__(self, config: dict):
        self.env_id = config.get("env_id", "XLand-MiniGrid-R4-9x9")
        self.tile_size = config.get("tile_size", 32)
        self.view_size = config.get("view_size", 7)

        self._env = None
        self._env_params = None
        self._tile_cache = None
        self._agent_tile_cache = None

    def _ensure_caches(self):
        """Build tile caches lazily (runs once, uses numpy internally)."""
        if self._tile_cache is not None:
            return
        from uedrlhf.jax_renderer import build_tile_caches
        self._tile_cache, self._agent_tile_cache = build_tile_caches(
            tile_size=self.tile_size
        )

    def create_env(self):
        import xminigrid
        self._env, self._env_params = xminigrid.make(self.env_id)
        return self._env, self._env_params

    def generate_states(self, rng, num_envs):
        if self._env is None:
            self.create_env()
        keys = jax.random.split(rng, num_envs)
        timesteps = jax.vmap(self._env.reset, in_axes=(None, 0))(
            self._env_params, keys
        )
        # Return the timesteps — the state lives inside timestep.state
        return timesteps

    def render_state(self, timestep) -> np.ndarray:
        self._ensure_caches()
        from uedrlhf.jax_renderer import jax_render

        img = jax_render(
            grid=timestep.state.grid,
            agent=timestep.state.agent,
            tile_cache=self._tile_cache,
            agent_tile_cache=self._agent_tile_cache,
            view_size=self.view_size,
            tile_size=self.tile_size,
        )
        return np.asarray(img)

    def render_batch(self, timesteps) -> np.ndarray:
        self._ensure_caches()
        from uedrlhf.jax_renderer import jax_render

        tile_cache = self._tile_cache
        agent_tile_cache = self._agent_tile_cache
        view_size = self.view_size
        tile_size = self.tile_size

        render_fn = jax.jit(jax.vmap(
            lambda grid, agent: jax_render(
                grid, agent, tile_cache, agent_tile_cache,
                view_size=view_size, tile_size=tile_size,
            )
        ))
        imgs = render_fn(timesteps.state.grid, timesteps.state.agent)
        return np.asarray(imgs)

    def state_to_dict(self, timestep) -> dict:
        return {
            "grid": np.array(timestep.state.grid),
            "agent_position": np.array(timestep.state.agent.position),
            "agent_direction": np.array(timestep.state.agent.direction),
        }

    def get_prompts(self) -> dict[str, str]:
        from uedrlhf.environments.xland_prompts import get_xland_prompt
        # Build the dict dynamically — only include feedback types that have a prompt
        prompts = {}
        for ftype in ("difficulty_feedback", "learnability_feedback"):
            try:
                prompts[ftype] = get_xland_prompt(self.env_id, ftype)
            except KeyError:
                pass
        return prompts

