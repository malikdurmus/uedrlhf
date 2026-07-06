"""Phase 2 (Gemini): Poll batch job status, download results, write to DB.

Usage:
    uv run scripts/feedback/llm_batch_collect.py --job batches/YOUR_JOB_ID --model gemini-2.5-flash
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

from google import genai

from uedrlhf.config import BASE_DIR, LLM_CONFIG, DB_PATH
from uedrlhf.db import get_connection, resolve_table_name
from uedrlhf.utils import choice_to_preference

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def parse_key(key: str) -> tuple[int, bool]:
    parts = key.split("_")
    pair_id = int(parts[1])
    swapped = bool(int(parts[3]))
    return pair_id, swapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Batch job name")
    parser.add_argument("--model", default=LLM_CONFIG.get("model", "gemini-2.5-flash"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    api_key_env = LLM_CONFIG.get("api_key_env", "GEMINI_API_KEY")
    api_key = args.api_key or os.environ.get(api_key_env)
    if not api_key:
        print(f"Error: Set {api_key_env} env var or pass --api-key")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    table_name = resolve_table_name(args.model)

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
        sys.exit(1)

    print("Downloading results...")
    result_file_name = batch_job.dest.file_name
    file_bytes = client.files.download(file=result_file_name)
    result_lines = file_bytes.decode("utf-8").splitlines()
    print(f"Got {len(result_lines)} result lines")

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
            errors += 1
            continue
        if parsed.get("error"):
            errors += 1
            continue
        try:
            text = parsed["response"]["candidates"][0]["content"]["parts"][0]["text"]
            result_json = json.loads(text)
            raw_pref = choice_to_preference(result_json["choice"])
            if swapped and raw_pref != 0.5:
                raw_pref = 1.0 - raw_pref
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
            print(f"  Pair {pair_id}: parse error — {e}")
            errors += 1

    conn.close()
    print(f"\nDone! Completed: {completed}, Errors: {errors}")


if __name__ == "__main__":
    main()
