import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# 1. Configuration - Use the clean base domain name
BASE_URL = "https://llm.mobile.ifi.lmu.de/"

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API_KEY = os.environ.get("OPENWEBUI_API_KEY")
if not API_KEY:
    sys.exit("OPENWEBUI_API_KEY is not set. Copy .env.example to .env and fill it in.")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Payload mapped to Open WebUI's exact specifications
data = {
    "model": "qwen3.6:27b",
    "messages": [
        {"role": "user", "content": "what is Xland-Minigrid"}
    ],
    "stream": False
}

print("Attempting to connect to LMU server...")

# Open WebUI maps its OpenAI-compatible completions to /api/chat/completions
response = requests.post(
    f"{BASE_URL}/api/chat/completions", 
    headers=headers, 
    json=data
)

print(f"Server Response Status Code: {response.status_code}")

if response.status_code == 200:
    json_data = response.json()
    # Extract the string response out of the standard formatting tree
    answer = json_data["choices"][0]["message"]["content"]
    print("\n🟢 Success! Response from LLM:")
    print(answer)
else:
    print("\n❌ Request failed.")
    print("Error Payload from Server:", response.text)
