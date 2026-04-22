"""LLM-based feedback collection for environment preference pairs (single-request mode).

Usage:
    python scripts/feedback/llm_feedback.py --model gemini-2.5-flash
    python scripts/feedback/llm_feedback.py --model gemini-2.5-flash --api-key YOUR_KEY
    python scripts/feedback/llm_feedback.py --model gemma-3-27b-it

The script creates a dedicated table per model in the same feedback.db,
so you can compare results across models and against human feedback.

Set your API key via the GEMINI_API_KEY environment variable or --api-key flag.
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

from google import genai
from google.genai import types

from uedrlhf.config import BASE_DIR, FEEDBACK_TYPE, IMAGES_DIR, DB_PATH, PROMPTS
from uedrlhf.db import get_connection, resolve_table_name, create_model_table, populate_from_llm_pairs
from uedrlhf.utils import choice_to_preference, LLMFeedbackResponse

load_dotenv(BASE_DIR / ".env")


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

    return {
        "preference": preference,
        "x_pct": round(parsed.x_pct, 2),
        "y_pct": round(parsed.y_pct, 2),
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

    # API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY env var or pass --api-key")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    table_name = resolve_table_name(args.model)
    prompt = PROMPTS[FEEDBACK_TYPE]

    print(f"Feedback type : {FEEDBACK_TYPE}")
    print(f"Model         : {args.model}")
    print(f"Table         : {table_name}")
    print(f"Database      : {DB_PATH}")
    print()

    # Database setup
    conn = get_connection()
    cursor = conn.cursor()

    create_model_table(cursor, table_name)
    populate_from_llm_pairs(cursor, table_name)
    conn.commit()

    # Get pending pairs
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
        img_1 = IMAGES_DIR / f"{env_1_id}.png"
        img_2 = IMAGES_DIR / f"{env_2_id}.png"

        if not img_1.exists() or not img_2.exists():
            print(f"  [{i+1}/{len(pending)}] Pair {pair_id}: SKIP — missing image(s)")
            errors += 1
            continue

        try:
            # Gemini might have some positional bias, so switch the labels half the time
            swapped = np.random.random() > 0.5
            if swapped:
                img_a, img_b = img_2, img_1
            else:
                img_a, img_b = img_1, img_2

            result = get_llm_preference(
                client, args.model, img_a, img_b, prompt
            )
            raw_pref = result["preference"]
            if swapped and raw_pref != 0.5:
                raw_pref = 1 - raw_pref

            cursor.execute(
                f"""
                UPDATE "{table_name}"
                SET preference = ?, x_pct = ?, y_pct = ?, time_ms = ?,
                    status = 'completed', raw_response = ?, timestamp = ?
                WHERE pair_id = ?
                """,
                (
                    raw_pref,
                    result["x_pct"],
                    result["y_pct"],
                    result["time_ms"],
                    result["raw"],
                    datetime.now(timezone.utc).isoformat(),
                    pair_id,
                ),
            )
            conn.commit()

            label = {0.0: "A (left)", 1.0: "B (right)", 0.5: "TIE"}[raw_pref]
            print(
                f"  [{i+1}/{len(pending)}] Pair {pair_id}: {label}  "
                f"({result['x_pct']:.1f}%, {result['y_pct']:.1f}%)  "
                f"[{result['time_ms']:.0f}ms]"
            )
            completed += 1

        except Exception as e:
            print(f"  [{i+1}/{len(pending)}] Pair {pair_id}: ERROR — {e}")
            errors += 1

        # Rate limiting
        if i < len(pending) - 1:
            pass  # time.sleep(args.delay)

    conn.close()

    print(f"\nDone! Completed: {completed}, Errors: {errors}")
    print(f"Results saved to table '{table_name}' in {DB_PATH}")


if __name__ == "__main__":
    main()
