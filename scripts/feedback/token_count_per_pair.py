"""Send a single sample request to Gemini to get the real input and output token counts and calculate total estimated costs.

Usage:
    uv run scripts/feedback/token_count_per_pair.py
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
        print(f"Error querying pending pairs: {e}")
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

    print(f"\n--- Real Token & Cost Estimation (Single Pair) ---")
    print(f"Model: {args.model}")
    print(f"Sending 1 real request to Vertex AI to measure true input/output tokens...\n")

    try:
        generation_config_args = {"response_mime_type": "application/json"}
        thinking_level = LLM_CONFIG.get("thinking_level", None)
        if thinking_level:
            generation_config_args["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level.upper())
            
        response = client.models.generate_content(
            model=args.model,
            contents=[
                "Image A (Left):", 
                types.Part.from_bytes(data=sample_img_a.read_bytes(), mime_type="image/png"), 
                "Image B (Right):", 
                types.Part.from_bytes(data=sample_img_b.read_bytes(), mime_type="image/png"), 
                prompt
            ],
            config=types.GenerateContentConfig(**generation_config_args)
        )
        
        usage = response.usage_metadata
        in_tokens = usage.prompt_token_count
        out_tokens = usage.candidates_token_count
        total_tokens = usage.total_token_count
        
        print(f"=== ACTUAL USAGE FOR 1 PAIR ===")
        print(f"Input Tokens:  {in_tokens}")
        print(f"Output Tokens: {out_tokens} (includes thinking tokens)")
        print(f"Total Tokens:  {total_tokens}\n")
        
        if "3.5-flash" in args.model:
            # Batch pricing is exactly 50% of the online pricing, but the standard Vertex AI 
            # 3.5 Flash batch pricing is $1.50/1M input and $9.00/1M output.
            in_cost_per_m = 1.50
            out_cost_per_m = 9.00
            
            single_in_cost = (in_tokens / 1_000_000) * in_cost_per_m
            single_out_cost = (out_tokens / 1_000_000) * out_cost_per_m
            single_total_cost = single_in_cost + single_out_cost
            
            print(f"=== EXTRAPOLATED COST FOR {len(pending)} PAIRS ===")
            total_in_cost = single_in_cost * len(pending)
            total_out_cost = single_out_cost * len(pending)
            
            print(f"Estimated Total Input Cost:  ${total_in_cost:.2f}")
            print(f"Estimated Total Output Cost: ${total_out_cost:.2f}")
            print(f"--------------------------------------------------")
            print(f"ESTIMATED TOTAL BILL:        ${total_in_cost + total_out_cost:.2f}")
            print(f"--------------------------------------------------\n")
            print(f"Note: This uses the exact config (thinking_level={thinking_level or 'FULL'}).")
            print(f"The final cost may vary slightly depending on how many output")
            print(f"tokens the model generates for each specific pair.")
            
    except Exception as e:
        print(f"\nReal request failed: {e}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
