"""Shared utility functions (environment-agnostic).

Environment-specific logic (rasterization, state serialization) lives
in the respective backend modules under uedrlhf/environments/.
"""

import pickle
import base64
from pathlib import Path
from pydantic import BaseModel
from PIL import Image
import numpy as np

from uedrlhf.config import RESIZE


# === Pydantic model for structured LLM feedback response ===

class LLMFeedbackResponse(BaseModel):
    """Expected JSON response from an LLM when comparing two environments."""
    choice: str
    x_pct: float
    y_pct: float
    reasoning: str


# === Helper functions ===

def img_to_base64(path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def choice_to_preference(choice: str) -> float:
    """Convert a textual choice ('A', 'B', 'TIE') to a numeric preference.

    Returns:
        0.0 if 'A' (env_1 preferred),
        1.0 if 'B' (env_2 preferred),
        0.5 if 'TIE'.
    """
    c = choice.strip().upper()
    if c == "A":
        return 0.0
    elif c == "B":
        return 1.0
    elif c == "TIE":
        return 0.5
    else:
        raise ValueError(f"Unexpected choice value: '{choice}'. Expected 'A', 'B', or 'TIE'.")


def load_image_as_nparray(file_path: Path, size: tuple = RESIZE) -> np.ndarray:
    """Load an image, resize it, and return as a float32 numpy array [0, 1]."""
    img = Image.open(file_path).resize(size).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def load_array(file_path: Path):
    """Load a pickled array from disk."""
    with open(file_path, "rb") as f:
        return pickle.load(f)
