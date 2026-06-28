"""Central configuration loader for the uedrlhf project.

All scripts import from here instead of loading YAML themselves.
One config.yaml → one source of truth.
"""

import yaml
from pathlib import Path

# --- Project root ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Load config ---
_config_path = BASE_DIR / "config.yaml"
with open(_config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

# --- Core settings ---
FEEDBACK_TYPE = CONFIG["feedback_type"]
LEARNABILITY_MODE = CONFIG.get("learnability_mode", "dijkstra")
FEEDBACK_SOURCE = CONFIG.get("feedback_source", "human")

# --- Constants ---
SUBFOLDER_MAP = {
    "difficulty_feedback": "difficulty",
    "learnability_feedback": "learnability",
}
VALID_LEARNABILITY_MODES = {"dijkstra", "param"}

# --- Validation ---
if FEEDBACK_TYPE not in SUBFOLDER_MAP:
    raise ValueError(
        f"Unknown feedback_type '{FEEDBACK_TYPE}' in config.yaml. "
        f"Must be one of: {list(SUBFOLDER_MAP.keys())}"
    )

if FEEDBACK_TYPE == "learnability_feedback":
    if LEARNABILITY_MODE not in VALID_LEARNABILITY_MODES:
        raise ValueError(
            f"Unknown learnability_mode '{LEARNABILITY_MODE}' in config.yaml. "
            f"Must be one of: {VALID_LEARNABILITY_MODES}"
        )

# --- Derived paths ---
DATA_VERSION = CONFIG.get("data_version", "v1")
SUBFOLDER = SUBFOLDER_MAP[FEEDBACK_TYPE]

if DATA_VERSION == "v2":
    DATA_DIR = BASE_DIR / "data_v2" / SUBFOLDER
    DB_PATH = DATA_DIR / "database_v2.db"
else:
    DATA_DIR = BASE_DIR / "data" / SUBFOLDER
    DB_PATH = DATA_DIR / "feedback.db"

IMAGES_DIR = DATA_DIR / "images"
ARRAYS_DIR = DATA_DIR / "arrays"

# --- Training config ---
TRAINING = CONFIG.get("training", {})

# --- Vaidity config ---
VALIDITY = CONFIG.get("validity" , {})

# --- UI text for human feedback app ---
FEEDBACK_UI = {
    "difficulty_feedback": {
        "title": "Difficulty Feedback",
        "subtitle": "Click the map that represents a harder environment.",
    },
    "learnability_feedback": {
        "title": "Learnability Feedback",
        "subtitle": "Click the map that represents a more learnable environment.",
    },
}

# --- LLM prompts ---
JSON_INSTRUCTION = (
    "\n\nRespond in JSON with exactly these fields:\n"
    '{"choice": "A" or "B" or "TIE", '
    '"x_pct": <0-100 float, x% position in the CHOSEN image of the most influential feature>, '
    '"y_pct": <0-100 float, y% position in the CHOSEN image of the most influential feature>, '
    '"reasoning": "<brief one-sentence explanation>"}'
)

PROMPTS = {
    "difficulty_feedback": (
        "You are evaluating two 2D navigation environments for a robot. "
        "Each image shows a top-down view with obstacles (walls), an agent (robot), and a goal position.\n\n"
        "Your task: Which environment is HARDER for the robot to navigate from its starting position to the goal?\n\n"
        "Consider factors like:\n"
        "- Number and placement of obstacles\n"
        "- How narrow or blocked the passages are\n"
        "- Distance and complexity of the path to the goal\n"
        "- Dead ends or traps\n\n"
        "Choose:\n"
        "  A  — if Image A (left) is harder\n"
        "  B  — if Image B (right) is harder\n"
        "  TIE — if they are equally difficult"
        + JSON_INSTRUCTION
    ),
    "learnability_feedback": (
        "You are evaluating two 2D navigation environments for a robot learning to navigate. "
        "Each image shows a top-down view with obstacles (walls), an agent (robot), and a goal position.\n\n"
        "Your task: Which environment is MORE LEARNABLE — meaning better suited for a robot to learn useful navigation skills?\n\n"
        "Consider factors like:\n"
        "- Does the environment provide a good learning curriculum (not too easy, not impossibly hard)?\n"
        "- Are there meaningful obstacles that teach the agent to navigate around things?\n"
        "- Is the goal reachable with a reasonable effort?\n"
        "- Does the layout encourage generalisable navigation behaviour?\n\n"
        "Choose:\n"
        "  A  — if Image A (left) is more learnable\n"
        "  B  — if Image B (right) is more learnable\n"
        "  TIE — if they are equally learnable"
        + JSON_INSTRUCTION
    ),
}
