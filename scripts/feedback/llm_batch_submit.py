"""Phase 1 (Gemini): Build JSONL, upload to File API, submit batch job.

Usage:
    uv run scripts/feedback/llm_batch_submit.py
    uv run scripts/feedback/llm_batch_submit.py --model gemini-2.5-flash
"""

import os
import sys
import json
import argparse
import numpy as np
import time

#  fix invalid_grant on cluster due to clock issue 
original_time = time.time
time.time = lambda: original_time() - 300

import google.auth
from google.oauth2 import service_account
from google import genai
from google.genai import types

from uedrlhf.config import (
    BASE_DIR, ENVIRONMENT, ENV_CONFIG,
    LLM_CONFIG, FEEDBACK_TYPE, DATA_DIR, IMAGES_DIR, DB_PATH, GOOGLE_CREDENTIALS_FILE, GCS_BUCKET,
)
from uedrlhf.db import get_connection, resolve_table_name, create_model_table, populate_from_llm_pairs
from uedrlhf.environments import get_env_backend
from uedrlhf.llm_providers import get_llm_provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=LLM_CONFIG.get("model", "gemini-3.5-flash"))
    args = parser.parse_args()

    sa_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", GOOGLE_CREDENTIALS_FILE)
    if os.path.exists(sa_file):
        credentials = service_account.Credentials.from_service_account_file(
            sa_file, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project_id = credentials.project_id
    else:
        credentials, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    client = genai.Client(vertexai=True, credentials=credentials, project=project_id, location="global")

    backend = get_env_backend(ENVIRONMENT, ENV_CONFIG)
    prompt = backend.get_prompts()[FEEDBACK_TYPE]

    provider_config = dict(LLM_CONFIG)
    provider_config["model"] = args.model
    provider = get_llm_provider("gemini", provider_config)

    table_name = resolve_table_name(args.model)

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
        print("No pending pairs found.")
        return

    print(f"Building JSONL for {len(pending)} pairs...")

    jsonl_path = DATA_DIR / f"batch_input_{table_name}.jsonl"
    skipped = 0

    with open(jsonl_path, "w") as f:
        for pair_id, env_1_id, env_2_id in pending:
            img_1 = IMAGES_DIR / f"{env_1_id}.png"
            img_2 = IMAGES_DIR / f"{env_2_id}.png"
            if not img_1.exists() or not img_2.exists():
                skipped += 1
                continue
            swapped = np.random.random() > 0.5
            img_a, img_b = (img_2, img_1) if swapped else (img_1, img_2)
            key = f"pair_{pair_id}_swapped_{int(swapped)}"
            line = provider.build_batch_request(img_a, img_b, prompt, key)
            f.write(json.dumps(line) + "\n")

    print(f"JSONL written: {jsonl_path}  (skipped {skipped})")

    bucket_name = GCS_BUCKET
    blob_name = f"batch_inputs/{DATA_DIR.name}_batch_input_{table_name}.jsonl"

    print("Uploading to GCS...")
    from google.cloud import storage
    storage_client = storage.Client(credentials=credentials, project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(jsonl_path))
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"Uploaded: {gcs_uri}")

    print("Submitting batch job...")
    batch_job = client.batches.create(
        model=args.model,
        src=gcs_uri,
        config={
            "display_name": f"llm_feedback_{table_name}_{DATA_DIR.name}",
            "dest": f"gs://{bucket_name}/batch_outputs/{DATA_DIR.name}/"
        },
    )

    print(f"\nBatch job submitted: {batch_job.name}")
    print(f"State: {batch_job.state.name}")
    print(f"\nFor collection run:")
    print(f"  uv run scripts/feedback/llm_batch_collect.py --job {batch_job.name} --model {args.model}")


if __name__ == "__main__":
    main()
