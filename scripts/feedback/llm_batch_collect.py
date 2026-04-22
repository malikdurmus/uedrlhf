"""Phase 2: Poll batch job status, download results, write to DB.

Usage:
    python scripts/feedback/llm_batch_collect.py --job batches/YOUR_JOB_ID --model gemini-2.5-flash
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from dotenv import load_dotenv

from google import genai

from uedrlhf.config import BASE_DIR, DB_PATH
from uedrlhf.db import get_connection, resolve_table_name
from uedrlhf.utils import choice_to_preference

load_dotenv(BASE_DIR / ".env")

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def parse_key(key: str) -> tuple[int, bool]:
    """Extract pair_id and swapped flag from key like 'pair_42_swapped_1'."""
    parts = key.split("_")
    pair_id = int(parts[1])
    swapped = bool(int(parts[3]))
    return pair_id, swapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Batch job name, e.g. batches/abc123")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between status polls")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY env var or pass --api-key")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    table_name = resolve_table_name(args.model)

    # ---------- poll until done
    print(f"Polling job: {args.job}")
    while True:
        batch_job = client.batches.get(name=args.job)
        state = batch_job.state.name
        print(f"[{datetime.now().strftime('%H:%M:%S')}] State: {state}")

        if state in COMPLETED_STATES:
            break

        time.sleep(args.poll_interval)

    if state != "JOB_STATE_SUCCEEDED":
        print(f"Job did not succeed: {state}")
        if hasattr(batch_job, "error") and batch_job.error:
            print(f"Error: {batch_job.error}")
        sys.exit(1)

    # ---------- download result JSONL
    print("Downloading results...")
    result_file_name = batch_job.dest.file_name
    file_bytes = client.files.download(file=result_file_name)
    result_lines = file_bytes.decode("utf-8").splitlines()
    print(f"Got {len(result_lines)} result lines, writing to DB...")

    # ---------- write to DB
    conn = get_connection()
    cursor = conn.cursor()

    completed = 0
    errors = 0

    for line in result_lines:
        if not line.strip():
            continue

        parsed = json.loads(line)
        key = parsed.get("key", "")

        try:
            pair_id, swapped = parse_key(key)
        except Exception:
            print(f"Could not parse key: {key}")
            errors += 1
            continue

        # Check for API-level error on this request
        if parsed.get("error"):
            print(f"  Pair {pair_id}: API error — {parsed['error']}")
            errors += 1
            continue

        try:
            text = parsed["response"]["candidates"][0]["content"]["parts"][0]["text"]
            result_json = json.loads(text)

            raw_pref = choice_to_preference(result_json["choice"])

            # Undo the positional swap
            if swapped and raw_pref != 0.5:
                raw_pref = 1.0 - raw_pref

            x_pct = round(float(result_json.get("x_pct", 0)), 2)
            y_pct = round(float(result_json.get("y_pct", 0)), 2)

            cursor.execute(
                f"""
                UPDATE "{table_name}"
                SET preference = ?, x_pct = ?, y_pct = ?, time_ms = NULL,
                    status = 'completed', raw_response = ?, timestamp = ?
                WHERE pair_id = ?
                """,
                (
                    raw_pref,
                    x_pct,
                    y_pct,
                    text,
                    datetime.now(timezone.utc).isoformat(),
                    pair_id,
                ),
            )
            conn.commit()

            label = {0.0: "A (left)", 1.0: "B (right)", 0.5: "TIE"}[raw_pref]
            reasoning = result_json.get("reasoning", "")
            print(f"  Pair {pair_id}: {label}  | {reasoning[:60]}")
            completed += 1

        except Exception as e:
            print(f"  Pair {pair_id}: parse error — {e} | raw: {line[:120]}")
            errors += 1

    conn.close()
    print(f"\nDone! Completed: {completed}, Errors: {errors}")
    print(f"Results saved to table '{table_name}' in {DB_PATH}")


if __name__ == "__main__":
    main()
