"""Shared utility functions for the uedrlhf project."""

import base64
import pickle
import numpy as np
import PIL.Image
from pathlib import Path
from pydantic import BaseModel, Field

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
