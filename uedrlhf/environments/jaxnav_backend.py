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
        self.only_solvable = config.get("only_solvable", False)

        self._env = None
        self._rasterizer = None

    def create_env(self):
        from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

        self._env = JaxNav(num_agents=self.num_agents)
        return self._env, None

    def _get_rasterizer(self):
        """Build the pure-JAX rasterizer (once)."""
        if self._rasterizer is not None:
            return self._rasterizer

        from uedrlhf.jax_renderer import get_jaxnav_rasterizer

        self._rasterizer = get_jaxnav_rasterizer(
            img_height=self.img_size,
            img_width=self.img_size,
            map_height=self.map_height,
            map_width=self.map_width,
            cell_size=self.cell_size
        )
        return self._rasterizer

    def generate_states(self, rng, num_envs):
        if self._env is None:
            self.create_env()
        keys = jax.random.split(rng, num_envs)
        
        solvability = "solvable" if self.only_solvable else "both"
        
        def reset_state_only(key):
            obs, state = self._env.reset(key, solvability=solvability)
            return state
            
        reset_fn = jax.vmap(reset_state_only)
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
