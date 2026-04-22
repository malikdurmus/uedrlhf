"""Phase 1: Build JSONL, upload to File API, submit batch job.

Run this, note the job name printed at the end, then wait for completion.

Usage:
    python scripts/feedback/llm_batch_submit.py --model gemini-2.5-flash
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

from google import genai
from google.genai import types

from uedrlhf.config import BASE_DIR, FEEDBACK_TYPE, DATA_DIR, IMAGES_DIR, DB_PATH, PROMPTS
from uedrlhf.db import get_connection, resolve_table_name, create_model_table, populate_from_llm_pairs
from uedrlhf.utils import img_to_base64

load_dotenv(BASE_DIR / ".env")


def build_request(img_a_path, img_b_path, prompt):
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
    table_name = resolve_table_name(args.model)
    prompt = PROMPTS[FEEDBACK_TYPE]

    # Get pending pairs from DB
    conn = get_connection()
    cursor = conn.cursor()

    create_model_table(cursor, table_name)
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

    # Build JSONL — one line per pair
    # We store swap info in the key so we can recover it during collection
    jsonl_path = DATA_DIR / f"batch_input_{table_name}.jsonl"
    skipped = 0

    with open(jsonl_path, "w") as f:
        for pair_id, env_1_id, env_2_id in pending:
            img_1 = IMAGES_DIR / f"{env_1_id}.png"
            img_2 = IMAGES_DIR / f"{env_2_id}.png"

            if not img_1.exists() or not img_2.exists():
                skipped += 1
                continue

            # Random swap to counteract positional bias
            swapped = np.random.random() > 0.5
            img_a, img_b = (img_2, img_1) if swapped else (img_1, img_2)

            # Key encodes pair_id and swap flag so collect script can decode it
            key = f"pair_{pair_id}_swapped_{int(swapped)}"

            request = build_request(img_a, img_b, prompt)
            line = {"key": key, "request": request}
            f.write(json.dumps(line) + "\n")

    print(f"JSONL written: {jsonl_path}  (skipped {skipped} missing pairs)")

    # Upload JSONL to File API
    print("Uploading to File API...")
    uploaded_file = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(
            display_name=f"batch_{table_name}",
            mime_type="jsonl"
        )
    )
    print(f"Uploaded: {uploaded_file.name}")

    # Submit batch job
    print("Submitting batch job...")
    batch_job = client.batches.create(
        model=args.model,
        src=uploaded_file.name,
        config={"display_name": f"llm_feedback_{table_name}"},
    )

    print(f"\nBatch job submitted: {batch_job.name}")
    print(f"State: {batch_job.state.name}")
    print(f"\nSave this job name for collection:")
    print(f"  python scripts/feedback/llm_batch_collect.py --job {batch_job.name} --model {args.model}")


if __name__ == "__main__":
    main()
