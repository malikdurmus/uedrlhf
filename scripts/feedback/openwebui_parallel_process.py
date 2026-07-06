"""Parallel feedback processing for Open WebUI.

Since Open WebUI does not have an asynchronous Batch API like OpenAI,
this script uses a local thread pool to send multiple concurrent requests
to the local server to process the dataset quickly.

Usage:
    uv run scripts/feedback/openwebui_parallel_process.py
    uv run scripts/feedback/openwebui_parallel_process.py --workers 4
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures
from threading import Lock

from uedrlhf.config import (
    ENVIRONMENT, ENV_CONFIG,
    FEEDBACK_TYPE, IMAGES_DIR, DB_PATH, CONFIG
)
from uedrlhf.db import get_connection, resolve_table_name, create_model_table, populate_from_llm_pairs
from uedrlhf.environments import get_env_backend
from uedrlhf.llm_providers import get_llm_provider


OPENWEBUI_CONFIG = CONFIG.get("openwebui", {})


def process_pair(pair, provider, prompt, table_name, db_lock):
    pair_id, env_1_id, env_2_id = pair

    img_1 = IMAGES_DIR / f"{env_1_id}.png"
    img_2 = IMAGES_DIR / f"{env_2_id}.png"

    if not img_1.exists() or not img_2.exists():
        return pair_id, "error", "Missing image(s)"

    try:
        # Random swap to counteract positional bias
        swapped = np.random.random() > 0.5
        if swapped:
            img_a, img_b = img_2, img_1
        else:
            img_a, img_b = img_1, img_2

        result = provider.get_preference(img_a, img_b, prompt)
        raw_pref = result["preference"]
        
        # Undo positional swap
        if swapped and raw_pref != 0.5:
            raw_pref = 1.0 - raw_pref

        # Write to DB thread-safely
        with db_lock:
            conn = get_connection()
            cursor = conn.cursor()
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
            conn.close()

        label = {0.0: "A (left)", 1.0: "B (right)", 0.5: "TIE"}[raw_pref]
        return pair_id, "success", f"{label} [{result['time_ms']:.0f}ms]"

    except Exception as e:
        return pair_id, "error", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=OPENWEBUI_CONFIG.get("model", "llama3.1:8b"))
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent requests")
    args = parser.parse_args()

    # Get prompt from environment backend
    backend = get_env_backend(ENVIRONMENT, ENV_CONFIG)
    prompt = backend.get_prompts()[FEEDBACK_TYPE]

    # Initialize provider
    provider_config = dict(OPENWEBUI_CONFIG)
    provider_config["model"] = args.model
    try:
        provider = get_llm_provider("openwebui", provider_config)
    except Exception as e:
        print(f"Error initializing Open WebUI provider: {e}")
        sys.exit(1)

    table_name = resolve_table_name(args.model)

    # Get pending pairs from DB
    conn = get_connection()
    cursor = conn.cursor()

    create_model_table(cursor, table_name)
    populate_from_llm_pairs(cursor, table_name)
    conn.commit()

    cursor.execute(
        f'SELECT pair_id, env_1_id, env_2_id FROM "{table_name}" WHERE status = ?',
        ("pending",),
    )
    pending = cursor.fetchall()
    conn.close()

    if not pending:
        print("No pending pairs found. All pairs have already been processed.")
        return

    print(f"Environment   : {ENVIRONMENT}")
    print(f"LLM Provider  : openwebui")
    print(f"Server URL    : {provider.base_url}")
    print(f"Model         : {args.model}")
    print(f"Workers       : {args.workers}")
    print(f"Table         : {table_name}")
    print(f"Processing {len(pending)} pending pairs...\n")

    db_lock = Lock()
    completed = 0
    errors = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_pair, pair, provider, prompt, table_name, db_lock): pair 
            for pair in pending
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            pair_id, status, msg = future.result()
            
            if status == "success":
                print(f"  [{i+1}/{len(pending)}] Pair {pair_id}: {msg}")
                completed += 1
            else:
                print(f"  [{i+1}/{len(pending)}] Pair {pair_id}: ERROR — {msg}")
                errors += 1

    print(f"\nDone! Completed: {completed}, Errors: {errors}")
    print(f"Results saved to table '{table_name}' in {DB_PATH}")


if __name__ == "__main__":
    main()
