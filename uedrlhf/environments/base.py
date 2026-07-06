"""Abstract base class for environment backends.

Every environment (JaxNav, XLand-MiniGrid, etc.) must implement this
interface so that generation, rendering, and feedback scripts can work
with any environment without knowing its internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax


class EnvBackend(ABC):
    """Contract that every environment backend must fulfill."""

    @abstractmethod
    def create_env(self):
        """Create and return the environment instance.

        Returns:
            A tuple of (env, env_params) where env_params may be None
            for environments that don't use separate params (e.g. JaxNav).
        """
        ...

    @abstractmethod
    def generate_states(self, rng: jax.Array, num_envs: int):
        """Generate `num_envs` random environment states.

        Args:
            rng: JAX PRNG key.
            num_envs: Number of environments to generate.

        Returns:
            A batched pytree of states with leading dimension `num_envs`.
        """
        ...

    @abstractmethod
    def render_state(self, state) -> np.ndarray:
        """Render a single environment state to an RGB image.

        Args:
            state: A single (unbatched) environment state.

        Returns:
            np.ndarray of shape (H, W, 3) with dtype uint8.
        """
        ...

    @abstractmethod
    def render_batch(self, states) -> np.ndarray:
        """Render a batch of environment states to RGB images.

        Args:
            states: A batched pytree of states.

        Returns:
            np.ndarray of shape (B, H, W, 3) with dtype uint8.
        """
        ...

    @abstractmethod
    def state_to_dict(self, state) -> dict:
        """Serialize a single environment state to a picklable dict.

        Args:
            state: A single (unbatched) environment state.

        Returns:
            A dict of numpy arrays suitable for pickle.dump().
        """
        ...

    @abstractmethod
    def get_prompts(self) -> dict[str, str]:
        """Return LLM prompts for this environment.

        Returns:
            A dict with keys 'difficulty_feedback' and/or
            'learnability_feedback', each mapping to the full prompt string
            describing what the LLM should evaluate.
        """
        ...
