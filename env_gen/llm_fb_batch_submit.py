# llm_feedback_batch_submit.py
"""
Phase 1: Build JSONL, upload to File API, submit batch job.
Run this, note the job name printed at the end, then wait for completion.
"""

import os
import re
import sys
import json
import base64
import yaml
import sqlite3
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from google import genai
from google.genai import types



# ---------- config
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent
load_dotenv(base_dir / ".env")

config_path = base_dir / "llm_feedback_config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# ---------- config
FEEDBACK_TYPE     = config["feedback_type"]
LEARNABILITY_MODE = config.get("learnability_mode", "dijkstra")

SUBFOLDER_MAP = {
    "difficulty_feedback": "difficulty",
    "learnability_feedback": "learnability",
}
VALID_LEARNABILITY_MODES = {"dijkstra", "param"}

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


subfolder = SUBFOLDER_MAP[FEEDBACK_TYPE]

data_dir = base_dir / "data" / subfolder
img_dir = data_dir / "images"
db_path = data_dir / "feedback.db"

# ---------- prompts (same as before)
JSON_INSTRUCTION = (
    "\n\nRespond in JSON with exactly these fields:\n"
    '{"choice": "A" or "B" or "TIE", '
    '"x_pct": <0-100 float>, '
    '"y_pct": <0-100 float>, '
    '"reasoning": "<brief one-sentence explanation>"}'
)

PROMPTS = {
    "difficulty_feedback": (
        "You are evaluating two 2D navigation environments for a robot. "
        "Each image shows a top-down view with obstacles (walls), an agent (robot), and a goal position.\n\n"
        "Your task: Which environment is HARDER for the robot to navigate from its starting position to the goal?\n\n"
        "Consider: obstacle placement, narrow passages, path complexity, dead ends.\n\n"
        "Choose A if Image A is harder, B if Image B is harder, TIE if equal."
        + JSON_INSTRUCTION
    ),
    "learnability_feedback": (
        "You are evaluating two 2D navigation environments for a robot learning to navigate. "
        "Each image shows a top-down view with obstacles, an agent, and a goal.\n\n"
        "Your task: Which environment is MORE LEARNABLE?\n\n"
        "Consider: curriculum difficulty, meaningful obstacles, reachable goal, generalisable behaviour.\n\n"
        "Choose A if Image A is more learnable, B if Image B is more learnable, TIE if equal."
        + JSON_INSTRUCTION
    ),
}

def sanitize_table_name(model_name: str) -> str:
    name = model_name.lower()
    name = re.sub(r"[^a-z0-9]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def img_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def build_request(img_a_path: Path, img_b_path: Path, prompt: str) -> dict:
    """Build a single GenerateContentRequest with two images."""
    return {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "Image A (Left):"},
                {"inline_data": {"mime_type": "image/png", "data": img_to_base64(img_a_path)}},
                {"text": "Image B (Right):"},
                {"inline_data": {"mime_type": "image/png", "data": img_to_base64(img_b_path)}},
                {"text": prompt},
            ]
        }],
        "generation_config": {
            "response_mime_type": "application/json",
        }
    }

def populate_from_llm_pairs(cursor: sqlite3.Cursor, table_name: str):
    """Copy pairs from the appropriate source table into the model table (skip existing)."""
    if FEEDBACK_TYPE == "learnability_feedback":
        source_table = f"llm_preference_pairs_{LEARNABILITY_MODE}"
        cursor.execute(f"""
            INSERT OR IGNORE INTO "{table_name}" (pair_id, env_1_id, env_2_id, split, bin, status)
            SELECT pair_id, env_1_id, env_2_id, split, bin, 'pending'
            FROM "{source_table}"
        """)
    else:
        cursor.execute(f"""
            INSERT OR IGNORE INTO "{table_name}" (pair_id, env_1_id, env_2_id, split, status)
            SELECT pair_id, env_1_id, env_2_id, split, 'pending'
            FROM llm_preference_pairs
        """)

def create_model_table(cursor: sqlite3.Cursor, table_name: str):
    """Create a model-specific preference table if it doesn't exist."""
    bin_col = "bin INTEGER," if FEEDBACK_TYPE == "learnability_feedback" else ""
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            pair_id      INTEGER PRIMARY KEY,
            env_1_id     TEXT NOT NULL,
            env_2_id     TEXT NOT NULL,
            split        TEXT NOT NULL,
            {bin_col}
            preference   REAL,
            status       TEXT DEFAULT 'pending',
            x_pct        REAL,
            y_pct        REAL,
            time_ms      REAL,
            raw_response TEXT,
            timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY env var or pass --api-key")
        sys.exit(1)

    client = genai.Client(api_key=api_key)


    if FEEDBACK_TYPE == "learnability_feedback":
        table_name = sanitize_table_name(args.model) + f"_{LEARNABILITY_MODE}"
    else:
        table_name = sanitize_table_name(args.model)
    
    prompt = PROMPTS[FEEDBACK_TYPE]

    # get pending pairs from DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    create_model_table(cursor , table_name)
    populate_from_llm_pairs(cursor, table_name)
    conn.commit()

    cursor.execute(
        f'SELECT pair_id, env_1_id, env_2_id FROM "{table_name}" WHERE status = ? LIMIT 10000',
        ("pending",),
    )
    pending = cursor.fetchall()
    conn.close()

    if not pending:
        print("No pending pairs found.")
        return

    print(f"Building JSONL for {len(pending)} pairs...")

    # build JSONL — one line per pair
    # we store swap info in the key so we can recover it during collection
    jsonl_path = data_dir / f"batch_input_{table_name}.jsonl"
    skipped = 0

    with open(jsonl_path, "w") as f:
        for pair_id, env_1_id, env_2_id in pending:
            img_1 = img_dir / f"{env_1_id}.png"
            img_2 = img_dir / f"{env_2_id}.png"

            if not img_1.exists() or not img_2.exists():
                skipped += 1
                continue

            # random swap to counteract positional bias
            swapped = np.random.random() > 0.5
            img_a, img_b = (img_2, img_1) if swapped else (img_1, img_2)

            # key encodes pair_id and swap flag so collect script can decode it
            key = f"pair_{pair_id}_swapped_{int(swapped)}"

            request = build_request(img_a, img_b, prompt)
            line = {"key": key, "request": request}
            f.write(json.dumps(line) + "\n")

    print(f"JSONL written: {jsonl_path}  (skipped {skipped} missing pairs)")

    # upload JSONL to File API
    print("Uploading to File API...")
    uploaded_file = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(
            display_name=f"batch_{table_name}",
            mime_type="jsonl"
        )
    )
    print(f"Uploaded: {uploaded_file.name}")

    # submit batch job
    print("Submitting batch job...")
    batch_job = client.batches.create(
        model=args.model,
        src=uploaded_file.name,
        config={"display_name": f"llm_feedback_{table_name}"},
    )

    print(f"\nBatch job submitted: {batch_job.name}")
    print(f"State: {batch_job.state.name}")
    print(f"\nSave this job name for collection:")
    print(f"  env_gen/python llm_fb_batch_collect.py --job {batch_job.name} --model {args.model}")

if __name__ == "__main__":
    main()