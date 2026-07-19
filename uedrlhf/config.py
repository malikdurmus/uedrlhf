"""Centralized configuration loader.

Reads config.yaml once at import time and exposes all settings as
module-level constants. Every script imports from here rather than
parsing YAML itself.
"""

from pathlib import Path
import yaml


# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
CONFIG_PATH = BASE_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)


# === Top-level settings ===
ENVIRONMENT = CONFIG.get("environment", "jaxnav")
LLM_PROVIDER = CONFIG.get("llm_provider", "gemini")
DATA_VERSION = CONFIG.get("data_version", "v2")
FEEDBACK_TYPE = CONFIG.get("feedback_type", "difficulty_feedback")
LEARNABILITY_MODE = CONFIG.get("learnability_mode", "dijkstra")
FEEDBACK_SOURCE = CONFIG.get("feedback_source", "human")

# === Validation ===
VALID_FEEDBACK_TYPES = {
    "difficulty_feedback",
    "learnability_feedback",
}
VALID_LEARNABILITY_MODES = {"dijkstra", "param"}
VALID_ENVIRONMENTS = {"jaxnav", "xland_minigrid"}
VALID_LLM_PROVIDERS = {"gemini", "openwebui"}

if ENVIRONMENT not in VALID_ENVIRONMENTS:
    raise ValueError(
        f"Invalid environment '{ENVIRONMENT}'. "
        f"Valid options: {VALID_ENVIRONMENTS}"
    )
if LLM_PROVIDER not in VALID_LLM_PROVIDERS:
    raise ValueError(
        f"Invalid llm_provider '{LLM_PROVIDER}'. "
        f"Valid options: {VALID_LLM_PROVIDERS}"
    )
if FEEDBACK_TYPE not in VALID_FEEDBACK_TYPES:
    raise ValueError(
        f"Invalid feedback_type '{FEEDBACK_TYPE}'. "
        f"Valid options: {VALID_FEEDBACK_TYPES}"
    )

# === Per-environment config ===
ENV_CONFIG = CONFIG.get(ENVIRONMENT, {})

# === Per-provider config ===
LLM_CONFIG = CONFIG.get(LLM_PROVIDER, {})

# === Data paths (scoped by environment, env_id, and feedback type) ===
ENV_ID = ENV_CONFIG.get("env_id", ENVIRONMENT)
RESIZE = tuple(ENV_CONFIG.get("resize", [64, 64]))
ONLY_SOLVABLE = ENV_CONFIG.get("only_solvable", False)

# Auto-tag: isolate solvable-only data into its own directory
_feedback_dir = f"{FEEDBACK_TYPE}_solvable" if ONLY_SOLVABLE else FEEDBACK_TYPE

if DATA_VERSION == "v1":
    DATA_DIR = BASE_DIR / "data" / _feedback_dir
else:
    # Avoid duplicate folders like jaxnav/jaxnav/
    if ENV_ID == ENVIRONMENT:
        DATA_DIR = BASE_DIR / f"data_{DATA_VERSION}" / ENVIRONMENT / _feedback_dir
    else:
        DATA_DIR = BASE_DIR / f"data_{DATA_VERSION}" / ENVIRONMENT / ENV_ID / _feedback_dir

IMAGES_DIR = DATA_DIR / "images"
ARRAYS_DIR = DATA_DIR / "arrays"
DB_PATH = DATA_DIR / "database_v2.db"

# === Training config ===
TRAINING = CONFIG.get("training", {})

# === JSON instruction suffix (appended to all LLM prompts) ===
JSON_INSTRUCTION = """

Respond ONLY with the following JSON — no other text:

{
  "choice": "<A | B | TIE>",
  "x_pct": <float 0-100 representing the exact horizontal focus point in the image>,
  "y_pct": <float 0-100 representing the exact vertical focus point in the image>,
  "reasoning": "<one sentence>"
}
"""
