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

from google import genai
from google.genai import types

from uedrlhf.config import (
    BASE_DIR, ENVIRONMENT, ENV_CONFIG,
    LLM_CONFIG, FEEDBACK_TYPE, DATA_DIR, IMAGES_DIR, DB_PATH,
)
from uedrlhf.db import get_connection, resolve_table_name, create_model_table, populate_from_llm_pairs
from uedrlhf.environments import get_env_backend
from uedrlhf.llm_providers import get_llm_provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=LLM_CONFIG.get("model", "gemini-2.5-flash"))
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key_env = LLM_CONFIG.get("api_key_env", "GEMINI_API_KEY")
    api_key = args.api_key or os.environ.get(api_key_env)
    if not api_key:
        print(f"Error: Set {api_key_env} env var or pass --api-key")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

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
        f'SELECT pair_id, env_1_id, env_2_id FROM "{table_name}" WHERE status = ? LIMIT 10000',
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

    print("Uploading to File API...")
    uploaded_file = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(display_name=f"batch_{table_name}", mime_type="jsonl")
    )
    print(f"Uploaded: {uploaded_file.name}")

    print("Submitting batch job...")
    batch_job = client.batches.create(
        model=args.model,
        src=uploaded_file.name,
        config={"display_name": f"llm_feedback_{table_name}"},
    )

    print(f"\nBatch job submitted: {batch_job.name}")
    print(f"State: {batch_job.state.name}")
    print(f"\nFor collection run:")
    print(f"  uv run scripts/feedback/llm_batch_collect.py --job {batch_job.name} --model {args.model}")


if __name__ == "__main__":
    main()
