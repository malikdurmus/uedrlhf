"""Open WebUI LLM provider.

Uses the `requests` library to connect to an OpenAI-compatible Open WebUI 
local server (e.g. https://llm.mobile.ifi.lmu.de).
"""

import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import requests

from uedrlhf.llm_providers.base import LLMProvider
from uedrlhf.utils import img_to_base64, choice_to_preference, LLMFeedbackResponse
from uedrlhf.config import BASE_DIR

load_dotenv(BASE_DIR / ".env")


class OpenWebUIProvider(LLMProvider):

    def __init__(self, config: dict):
        self.model_name = config.get("model", "llama3.1:8b")
        api_key_env = config.get("api_key_env", "OPENWEBUI_API_KEY")
        self.api_key = os.environ.get(api_key_env)
        
        # Open WebUI URL (https://llm.mobile.ifi.lmu.de)
        self.base_url = config.get("base_url", "").rstrip("/")
        
        if not self.api_key:
            raise ValueError(
                f"Open WebUI API key not found. Set the {api_key_env} environment variable "
                f"or add it to .env"
            )
        if not self.base_url:
            raise ValueError(
                "Open WebUI requires a 'base_url' in config.yaml under the openwebui section."
            )

        self.endpoint = f"{self.base_url}/api/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_messages(self, img_a_path: Path, img_b_path: Path, prompt: str) -> list:
        """Build the chat messages with vision content for OpenAI-compatible endpoint."""
        b64_a = img_to_base64(img_a_path)
        b64_b = img_to_base64(img_b_path)

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image A (Left):"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_a}"
                        }
                    },
                    {"type": "text", "text": "Image B (Right):"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_b}"
                        }
                    },
                    {"type": "text", "text": prompt},
                ]
            }
        ]

    def get_preference(self, img_a_path: Path, img_b_path: Path, prompt: str) -> dict:
        messages = self._build_messages(img_a_path, img_b_path, prompt)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            # Force JSON output mode if supported by Open WebUI
            "response_format": {"type": "json_object"}
        }

        t0 = time.perf_counter()

        response = requests.post(
            self.endpoint,
            headers=self.headers,
            json=payload
        )
        
        # Raise an exception if the request failed
        response.raise_for_status()

        time_ms = (time.perf_counter() - t0) * 1000
        json_data = response.json()
        raw = json_data["choices"][0]["message"]["content"]


        # --- Clean markdown so that it is pure json ##
        raw_clean = raw.strip()
        # Find the first '{' and the last '}' to extract pure JSON
        start_idx = raw_clean.find('{')
        end_idx = raw_clean.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            raw_clean = raw_clean[start_idx:end_idx+1]
        # -----------------------------------       ##

        try:
            parsed = LLMFeedbackResponse.model_validate_json(raw_clean)
        except Exception as e:
            print(f"Failed to parse JSON response: {raw}")
            raise e

        preference = choice_to_preference(parsed.choice)

        return {
            "preference": preference,
            "x_pct": round(parsed.x_pct, 2),
            "y_pct": round(parsed.y_pct, 2),
            "time_ms": round(time_ms, 1),
            "raw": raw,
        }

    def build_batch_request(
        self,
        img_a_path: Path,
        img_b_path: Path,
        prompt: str,
        key: str,
    ) -> dict:
        """Open WebUI does not support the OpenAI Batch API natively.
        
        Instead of using a batch submit script, we will use a multi-threaded
        parallel processing script to hit the /api/chat/completions endpoint.
        """
        raise NotImplementedError("Open WebUI does not use the OpenAI Batch API endpoints.")
