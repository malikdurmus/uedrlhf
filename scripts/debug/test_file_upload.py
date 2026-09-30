import os
import sys
from google import genai
from google.genai import types

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    with open("dummy.jsonl", "w") as f:
        f.write('{"request": {"contents": [{"parts": [{"text": "Hello"}]}]}}\n')
    
    uploaded_file = client.files.upload(file="dummy.jsonl", config=types.UploadFileConfig(mime_type="jsonl"))
    print(f"Uploaded file state: {uploaded_file.state}")
    
    try:
        batch_job = client.batches.create(
            model="gemini-2.5-flash",
            src=uploaded_file.name,
        )
        print("Success without waiting!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
