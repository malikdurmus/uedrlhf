"""JaxNav environment backend.

Wraps the JaxNav multi-agent navigation environment with its built-in
pure-JAX rasterizer for GPU-accelerated rendering.
"""

import jax
import jax.numpy as jnp
import numpy as np

from uedrlhf.environments.base import EnvBackend
from uedrlhf.config import JSON_INSTRUCTION


class JaxNavBackend(EnvBackend):

    def __init__(self, config: dict):
        self.num_agents = config.get("num_agents", 1)
        self.map_height = config.get("map_height", 11.0)
        self.map_width = config.get("map_width", 11.0)
        self.cell_size = config.get("cell_size", 1.0)
        self.img_size = config.get("img_size", 120)

        self._env = None
        self._rasterizer = None

    def create_env(self):
        from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "num_agents": self.num_agents,
            "map_height": self.map_height,
            "map_width": self.map_width,
            "cell_size": self.cell_size,
        })
        self._env = JaxNav(num_agents=cfg.num_agents, **cfg)
        return self._env, None

    def _get_rasterizer(self):
        """Build the pure-JAX rasterizer (once)."""
        if self._rasterizer is not None:
            return self._rasterizer

        from jaxmarl.environments.jaxnav.jaxnav_viz import rasterize_jaxnav

        cell_size = self.cell_size
        img_size = self.img_size
        h = int(self.map_height / cell_size) + 2
        w = int(self.map_width / cell_size) + 2

        @jax.jit
        def _rasterize(state):
            return rasterize_jaxnav(state, h, w, img_size)

        self._rasterizer = _rasterize
        return self._rasterizer

    def generate_states(self, rng, num_envs):
        if self._env is None:
            self.create_env()
        keys = jax.random.split(rng, num_envs)
        reset_fn = jax.vmap(self._env.reset_env)
        states = reset_fn(keys)
        return states

    def render_state(self, state) -> np.ndarray:
        rasterize = self._get_rasterizer()
        img = rasterize(state)
        return np.asarray(img)

    def render_batch(self, states) -> np.ndarray:
        rasterize = self._get_rasterizer()
        render_batch_fn = jax.jit(jax.vmap(rasterize))
        imgs = render_batch_fn(states)
        return np.asarray(imgs)

    def state_to_dict(self, state) -> dict:
        return {
            "map_data": np.array(state.map_data),
            "pos": np.array(state.pos),
            "goal": np.array(state.goal),
        }

    def get_prompts(self) -> dict[str, str]:
        return {
            "difficulty_feedback": (
                "You are evaluating two 2D grid-world navigation environments for a circular robot agent. "
                "Each image shows a top-down view with:\n"
                "- Black cells = obstacles (walls)\n"
                "- White cells = free space\n"
                "- Green circle = agent start position\n"
                "- Red circle = goal position\n\n"
                "Your task: Which environment is HARDER for the agent to navigate?\n\n"
                "Consider factors like:\n"
                "- Number and placement of obstacles\n"
                "- Narrowness of corridors the agent must pass through\n"
                "- Length and complexity of the path from start to goal\n"
                "- Whether the path requires sharp turns or backtracking\n\n"
                "Choose:\n"
                "  A  — if Image A is harder\n"
                "  B  — if Image B is harder\n"
                "  TIE — if they are equally difficult"
                + JSON_INSTRUCTION
            ),
            "learnability_feedback": (
                "You are evaluating two 2D navigation environments for training an RL agent.\n\n"
                "Your task: Which environment is MORE LEARNABLE — i.e. which is better as a "
                "training environment that helps the agent improve its policy?\n\n"
                "A good learning environment:\n"
                "- Is neither trivially easy nor impossibly hard\n"
                "- Provides meaningful challenge that forces the agent to learn\n"
                "- Has clear, reachable goals with non-trivial paths\n\n"
                "Choose:\n"
                "  A  — if Image A is more learnable\n"
                "  B  — if Image B is more learnable\n"
                "  TIE — if they are equally learnable"
                + JSON_INSTRUCTION
            ),
        }
