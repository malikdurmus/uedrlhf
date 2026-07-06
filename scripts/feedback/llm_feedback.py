"""LLM-based feedback collection for environment preference pairs (single-request mode).

Works with any LLM provider (Gemini, OpenAI, etc.) and any environment backend
by reading from config.yaml.

Usage:
    uv run scripts/feedback/llm_feedback.py
    uv run scripts/feedback/llm_feedback.py --model gemini-2.5-flash
    uv run scripts/feedback/llm_feedback.py --model gpt-4o

The script creates a dedicated table per model in the same feedback.db,
so you can compare results across models and against human feedback.
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from uedrlhf.config import (
    BASE_DIR, ENVIRONMENT, ENV_CONFIG,
    LLM_PROVIDER, LLM_CONFIG, FEEDBACK_TYPE, IMAGES_DIR, DB_PATH,
)
from uedrlhf.db import get_connection, resolve_table_name, create_model_table, populate_from_llm_pairs
from uedrlhf.utils import choice_to_preference
from uedrlhf.environments import get_env_backend
from uedrlhf.llm_providers import get_llm_provider


def main():
    parser = argparse.ArgumentParser(
        description="Collect LLM preference feedback on environment pairs."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name from config (e.g. gemini-2.5-flash, gpt-4o)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls for rate limiting (default: 1.0)",
    )
    args = parser.parse_args()

    # Use model from args or fall back to config
    provider_config = dict(LLM_CONFIG)
    if args.model:
        provider_config["model"] = args.model

    # Initialize provider and get prompt
    provider = get_llm_provider(LLM_PROVIDER, provider_config)
    backend = get_env_backend(ENVIRONMENT, ENV_CONFIG)
    prompts = backend.get_prompts()
    prompt = prompts[FEEDBACK_TYPE]

    model_name = provider_config.get("model", LLM_PROVIDER)
    table_name = resolve_table_name(model_name)

    print(f"Environment   : {ENVIRONMENT}")
    print(f"LLM Provider  : {LLM_PROVIDER}")
    print(f"Model         : {model_name}")
    print(f"Feedback type : {FEEDBACK_TYPE}")
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
            # Random swap to counteract positional bias
            swapped = np.random.random() > 0.5
            if swapped:
                img_a, img_b = img_2, img_1
            else:
                img_a, img_b = img_1, img_2

            result = provider.get_preference(img_a, img_b, prompt)
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
            time.sleep(args.delay)

    conn.close()

    print(f"\nDone! Completed: {completed}, Errors: {errors}")
    print(f"Results saved to table '{table_name}' in {DB_PATH}")


if __name__ == "__main__":
    main()
