"""
Standalone script to salvage the hallucinated JSON output from the old difficulty prompt.
It parses a local JSONL file, infers the pairwise choice from the independent difficulty ratings,
and safely writes the salvaged preferences to the database.

Usage:
    uv run scripts/feedback/salvage_difficulty_jsonl.py
"""

import os
import json
import argparse
from datetime import datetime, timezone
import glob

from uedrlhf.config import LLM_CONFIG
from uedrlhf.db import get_connection, resolve_table_name
from uedrlhf.utils import choice_to_preference

def parse_key(key: str) -> tuple[int, bool]:
    parts = key.split("_")
    pair_id = int(parts[1])
    swapped = bool(int(parts[3]))
    return pair_id, swapped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to the downloaded JSONL file. If not provided, it searches the root directory.")
    parser.add_argument("--model", default=LLM_CONFIG.get("model", "gemini-3.5-flash"))
    args = parser.parse_args()

    file_path = args.file
    if not file_path:
        # Search for any JSONL files in the root folder with the batch_outputs naming convention
        files = glob.glob("batch_outputs_difficulty_feedback_prediction*.jsonl")
        if not files:
            files = glob.glob("*predictions (1).jsonl")
            if not files:
                print("Could not find any JSONL file in the current directory.")
                print("Please explicitly specify it: uv run scripts/feedback/salvage_difficulty_jsonl.py --file YOUR_FILE.jsonl")
                return
        file_path = files[0]
        
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    table_name = resolve_table_name(args.model)
    print(f"Reading from: {file_path}")
    print(f"Updating table: {table_name}")

    conn = get_connection()
    cursor = conn.cursor()
    completed = 0
    errors = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
                
            key = parsed.get("key", "")
            try:
                pair_id, swapped = parse_key(key)
            except Exception:
                # Malformed key or not a prediction line
                continue
                
            if parsed.get("error"):
                errors += 1
                continue
                
            if not parsed.get("response"):
                # Job was cancelled, this pair was skipped by Google Cloud
                continue

            try:
                text = parsed["response"]["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                # Clean text if wrapped in markdown
                if text.startswith("```json"):
                    text = text[7:]
                elif text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                
                result_json = json.loads(text)
                choice = None
                
                # Format 1: Model returned a list of evaluations (e.g. [{"difficulty_rating": 2}, {"difficulty_rating": 4}])
                if isinstance(result_json, list) and len(result_json) >= 2:
                    rating_a = float(result_json[0].get("difficulty_rating", 0))
                    rating_b = float(result_json[1].get("difficulty_rating", 0))
                    if rating_a > rating_b: choice = "A"
                    elif rating_b > rating_a: choice = "B"
                    else: choice = "TIE"
                        
                # Format 2: Model returned a dict mapping images to evaluations (e.g. {"image_a": {"difficulty_rating": 2}, ...})
                elif isinstance(result_json, dict) and "choice" not in result_json:
                    val_a = result_json.get("image_a", result_json.get("A", result_json.get("Image A", {})))
                    val_b = result_json.get("image_b", result_json.get("B", result_json.get("Image B", {})))
                    if isinstance(val_a, dict) and isinstance(val_b, dict):
                        rating_a = float(val_a.get("difficulty_rating", 0))
                        rating_b = float(val_b.get("difficulty_rating", 0))
                        if rating_a > rating_b: choice = "A"
                        elif rating_b > rating_a: choice = "B"
                        else: choice = "TIE"
                            
                # Format 3: Expected default format ({"choice": "A"})
                elif isinstance(result_json, dict) and "choice" in result_json:
                    choice = result_json["choice"]
                    
                if choice is None:
                    raise ValueError("Could not parse choice from hallucinated JSON format")
                    
                raw_pref = choice_to_preference(choice)
                if swapped and raw_pref != 0.5:
                    raw_pref = 1.0 - raw_pref
                    
                x_pct = 0.0
                y_pct = 0.0
                if isinstance(result_json, dict):
                    x_pct = round(float(result_json.get("x_pct", 0)), 2)
                    y_pct = round(float(result_json.get("y_pct", 0)), 2)
                    
                cursor.execute(
                    f'UPDATE "{table_name}" SET preference=?, x_pct=?, y_pct=?, '
                    f'status=\'completed\', raw_response=?, timestamp=? WHERE pair_id=?',
                    (raw_pref, x_pct, y_pct, text, datetime.now(timezone.utc).isoformat(), pair_id),
                )
                conn.commit()
                completed += 1
            except Exception as e:
                # Uncomment the line below if you want to see exactly which ones failed to parse
                # print(f"  Pair {pair_id}: parse error — {e}")
                errors += 1

    conn.close()
    print(f"\nSalvage Complete! Successfully recovered: {completed} pairs. Unparseable errors: {errors}")

if __name__ == "__main__":
    main()
