"""
LLM-based feedback collection for environment preference pairs.

Usage:
    python env_gen/llm_feedback.py --model gemini-2.5-flash
    python env_gen/llm_feedback.py --model gemini-2.5-flash --api-key YOUR_KEY
    python env_gen/llm_feedback.py --model gemma-3-27b-it

The script creates a dedicated table per model in the same feedback.db,
so you can compare results across models and against human feedback.

Set your API key via the GEMINI_API_KEY environment variable or --api-key flag.
"""

import os
import re
import sys
import json
import time
import yaml
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timezone

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# load config
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent

# load .env from project root
load_dotenv(base_dir / ".env")

config_path = base_dir / "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

FEEDBACK_TYPE = config["feedback_type"]

SUBFOLDER_MAP = {
    "difficulty_feedback": "difficulty",
    "learnability_feedback": "learnability",
}

if FEEDBACK_TYPE not in SUBFOLDER_MAP:
    raise ValueError(
        f"Unknown feedback_type '{FEEDBACK_TYPE}' in config.yaml. "
        f"Must be one of: {list(SUBFOLDER_MAP.keys())}"
    )

subfolder = SUBFOLDER_MAP[FEEDBACK_TYPE]

# paths
data_dir = base_dir / "data" / subfolder
img_dir = data_dir / "images"
db_path = data_dir / "feedback.db"


# TODO: Better Prompts 
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

# Pydantic model for structured Gemini JSON output
class LLMFeedbackResponse(BaseModel):
    choice: str = Field(description="A, B, or TIE")
    x_pct: float = Field(description="x% position (0-100) in the chosen image of the most influential feature")
    y_pct: float = Field(description="y% position (0-100) in the chosen image of the most influential feature")
    reasoning: str = Field(description="Brief one-sentence explanation of the decision")


def sanitize_table_name(model_name: str) -> str:
    """Convert a model name like 'gemini-2.5-flash' into a valid SQLite table name."""
    name = model_name.lower()
    name = re.sub(r"[^a-z0-9]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def create_model_table(cursor: sqlite3.Cursor, table_name: str):
    """Create a model-specific preference table if it doesn't exist."""
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            pair_id INTEGER PRIMARY KEY,
            env_1_id TEXT NOT NULL,
            env_2_id TEXT NOT NULL,
            split TEXT NOT NULL,
            preference REAL,
            status TEXT DEFAULT 'pending',
            x_pct REAL,
            y_pct REAL,
            time_ms REAL,
            raw_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)


def populate_from_llm_pairs(cursor: sqlite3.Cursor, table_name: str):
    """Copy pairs from llm_preference_pairs into the model table (skip existing)."""
    cursor.execute(f"""
        INSERT OR IGNORE INTO "{table_name}" (pair_id, env_1_id, env_2_id, split, status)
        SELECT pair_id, env_1_id, env_2_id, split, 'pending'
        FROM llm_preference_pairs
    """)


def choice_to_preference(choice: str) -> float:
    """Map choice string to preference score.
    
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
    else:
        return 0.5


def get_llm_preference(
    client: genai.Client,
    model_name: str,
    img_path_1: Path,
    img_path_2: Path,
    prompt: str,
) -> dict:
    """Send two images to the LLM and return parsed JSON response + time_ms."""
    img1_bytes = img_path_1.read_bytes()
    img2_bytes = img_path_2.read_bytes()

    t0 = time.perf_counter()

    response = client.models.generate_content(
        model=model_name,
        contents=[
            "Image A (Left):",
            types.Part.from_bytes(data=img1_bytes, mime_type="image/png"),
            "Image B (Right):",
            types.Part.from_bytes(data=img2_bytes, mime_type="image/png"),
            prompt,
        ],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": LLMFeedbackResponse.model_json_schema(),
        },
    )

    time_ms = (time.perf_counter() - t0) * 1000
    raw = response.text

    parsed = LLMFeedbackResponse.model_validate_json(raw)
    preference = choice_to_preference(parsed.choice)
    x_pct = parsed.x_pct
    y_pct = parsed.y_pct

    return {
        "preference": preference,
        "x_pct": round(x_pct, 2),
        "y_pct": round(y_pct, 2),
        "time_ms": round(time_ms, 1),
        "raw": raw,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect LLM preference feedback on environment pairs."
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls for rate limiting (default: 1.0)",
    )
    args = parser.parse_args()

    # api key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY env var or pass --api-key")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    table_name = sanitize_table_name(args.model)
    prompt = PROMPTS[FEEDBACK_TYPE]

    print(f"Feedback type : {FEEDBACK_TYPE}")
    print(f"Model         : {args.model}")
    print(f"Table         : {table_name}")
    print(f"Database      : {db_path}")
    print()

    # database setup
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    create_model_table(cursor, table_name)
    populate_from_llm_pairs(cursor, table_name)
    conn.commit()

    # get pending pairs
    cursor.execute(
        f'SELECT pair_id, env_1_id, env_2_id FROM "{table_name}" WHERE status = ?',
        ("pending",),
    )
    pending = cursor.fetchall()

    if not pending:
        print("No pending pairs found. All pairs have already been processed.")
        conn.close()
        return

    print(f"Processing {len(pending)} pending pairs...\n")

    completed = 0
    errors = 0

    for i, (pair_id, env_1_id, env_2_id) in enumerate(pending):
        img_1 = img_dir / f"{env_1_id}.png"
        img_2 = img_dir / f"{env_2_id}.png"

        if not img_1.exists() or not img_2.exists():
            print(f"  [{i+1}/{len(pending)}] Pair {pair_id}: SKIP — missing image(s)")
            errors += 1
            continue

        try:
            result = get_llm_preference(
                client, args.model, img_1, img_2, prompt
            )

            cursor.execute(
                f"""
                UPDATE "{table_name}"
                SET preference = ?, x_pct = ?, y_pct = ?, time_ms = ?,
                    status = 'completed', raw_response = ?, timestamp = ?
                WHERE pair_id = ?
                """,
                (
                    result["preference"],
                    result["x_pct"],
                    result["y_pct"],
                    result["time_ms"],
                    result["raw"],
                    datetime.now(timezone.utc).isoformat(),
                    pair_id,
                ),
            )
            conn.commit()

            label = {0.0: "A (left)", 1.0: "B (right)", 0.5: "TIE"}[result["preference"]]
            print(
                f"  [{i+1}/{len(pending)}] Pair {pair_id}: {label}  "
                f"({result['x_pct']:.1f}%, {result['y_pct']:.1f}%)  "
                f"[{result['time_ms']:.0f}ms]"
            )
            completed += 1

        except Exception as e:
            print(f"  [{i+1}/{len(pending)}] Pair {pair_id}: ERROR — {e}")
            errors += 1

        # rate limiting
        if i < len(pending) - 1:
            time.sleep(args.delay)

    conn.close()

    print(f"\nDone! Completed: {completed}, Errors: {errors}")
    print(f"Results saved to table '{table_name}' in {db_path}")


if __name__ == "__main__":
    main()
