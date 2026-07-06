import requests

# 1. Configuration - Use the clean base domain name
BASE_URL = "https://llm.mobile.ifi.lmu.de/"
API_KEY = "sk-9d322b65643244009b1d89326213edac"  # Paste your key inside these quotes

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
