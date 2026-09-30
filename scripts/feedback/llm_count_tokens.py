"""Count tokens for a sample pair before submitting a batch job.

Usage:
    uv run scripts/feedback/llm_count_tokens.py
"""

import os
import argparse
import google.auth
from google.oauth2 import service_account
from google import genai
from google.genai import types

from uedrlhf.config import (
    ENVIRONMENT, ENV_CONFIG, LLM_CONFIG, FEEDBACK_TYPE, 
    IMAGES_DIR, GOOGLE_CREDENTIALS_FILE
)
from uedrlhf.db import get_connection, resolve_table_name
from uedrlhf.environments import get_env_backend

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=LLM_CONFIG.get("model", "gemini-3.5-flash"))
    args = parser.parse_args()

    # Authenticate
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
    
    table_name = resolve_table_name(args.model)

    # Get pending pairs
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            f'SELECT pair_id, env_1_id, env_2_id FROM "{table_name}" WHERE status = ?',
            ("pending",),
        )
        pending = cursor.fetchall()
    except Exception as e:
        print(f"Error querying pending pairs (table might not exist yet): {e}")
        pending = []
    finally:
        conn.close()

    if not pending:
        print("No pending pairs found in the database to estimate.")
        return

    sample_pair_id, sample_env_1, sample_env_2 = pending[0]
    sample_img_a = IMAGES_DIR / f"{sample_env_1}.png"
    sample_img_b = IMAGES_DIR / f"{sample_env_2}.png"

    if not sample_img_a.exists() or not sample_img_b.exists():
        print(f"Could not find images for sample pair: {sample_img_a} or {sample_img_b}")
        return

    print("\n--- Pre-Job Token Estimation (Safety Check) ---")
    print(f"Sampling Pair ID: {sample_pair_id}")
    print(f"Model: {args.model}")
    print(f"Image 1: {sample_img_a.name}")
    print(f"Image 2: {sample_img_b.name}")
    print("Sending sample to Vertex AI for token counting...")

    try:
        count_response = client.models.count_tokens(
            model=args.model,
            contents=[
                "Image A (Left):", 
                types.Part.from_bytes(data=sample_img_a.read_bytes(), mime_type="image/png"), 
                "Image B (Right):", 
                types.Part.from_bytes(data=sample_img_b.read_bytes(), mime_type="image/png"), 
                prompt
            ]
        )
        
        print("\n=== RESULTS ===")
        print(f"Total Input Tokens per pair: {count_response.total_tokens}")
        print(f"Estimated total input tokens for {len(pending)} pending pairs: {count_response.total_tokens * len(pending)}")
        
        # Simple cost calculation based on $1.50 per million input tokens for 3.5-flash
        if "3.5-flash" in args.model:
            estimated_cost = (count_response.total_tokens * len(pending)) / 1_000_000 * 1.50
            print(f"Estimated input cost (at $1.50 / 1M tokens): ${estimated_cost:.2f}")
            print("\nNote: This is ONLY the input token cost. Output 'thinking' tokens will cost extra.")
            print("Since you've enabled thinkingLevel: LOW, output costs will be minimal.")
            
    except Exception as e:
        print(f"\nToken estimation failed: {e}")
    print("-----------------------------------------------\n")


if __name__ == "__main__":
    main()
