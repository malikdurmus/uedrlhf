"""Phase 2 (Gemini): Poll batch job status, download results, write to DB.

Usage:
    uv run scripts/feedback/llm_batch_collect.py --job batches/YOUR_JOB_ID --model gemini-3.5-flash
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

#  to fix invalid_grant on cluster due to clock problem
original_time = time.time
time.time = lambda: original_time() - 300

import google.auth
from google.oauth2 import service_account
from google.cloud import storage

from google import genai

from uedrlhf.config import BASE_DIR, LLM_CONFIG, DB_PATH, DATA_DIR, GOOGLE_CREDENTIALS_FILE, GCS_BUCKET
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
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    sa_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", GOOGLE_CREDENTIALS_FILE)
    if os.path.exists(sa_file):
        credentials = service_account.Credentials.from_service_account_file(
            sa_file, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project_id = credentials.project_id
    else:
        credentials, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    location = "global"
    if "locations/" in args.job:
        parts = args.job.split("/")
        try:
            location = parts[parts.index("locations") + 1]
        except (ValueError, IndexError):
            pass

    client = genai.Client(vertexai=True, credentials=credentials, project=project_id, location=location)
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
        print("Still processing the output")

    print("Downloading results from GCS...")
    
    # Initialize GCS client using credentials
    storage_client = storage.Client(credentials=credentials, project=project_id)
    bucket_name = GCS_BUCKET
    bucket = storage_client.bucket(bucket_name)

   
    blobs = list(bucket.list_blobs(prefix=f"batch_outputs/{DATA_DIR.name}/"))
    jsonl_blobs = [b for b in blobs if b.name.endswith(".jsonl")]

    # Fallback for jobs submitted before the isolation fix
    if not jsonl_blobs:
        print(f"No results in gs://{bucket_name}/batch_outputs/{DATA_DIR.name}/ ... falling back to root batch_outputs/")
        blobs = list(bucket.list_blobs(prefix="batch_outputs/"))
        jsonl_blobs = [b for b in blobs if b.name.endswith(".jsonl")]

    if not jsonl_blobs:
        print(f"No result JSONL files found in gs://{bucket_name}/batch_outputs/ at all!")
        sys.exit(1)
    # pick the latest
    latest_blob = max(jsonl_blobs, key=lambda b: b.time_created)
    print(f"Reading from: gs://{bucket_name}/{latest_blob.name}")
    
    file_bytes = latest_blob.download_as_bytes()
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
