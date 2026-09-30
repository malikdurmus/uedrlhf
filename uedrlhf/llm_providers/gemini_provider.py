"""Gemini LLM provider.

Wraps the google-genai SDK for single-request and batch feedback
collection using the Gemini API.
"""

import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types

from uedrlhf.llm_providers.base import LLMProvider
from uedrlhf.utils import img_to_base64, choice_to_preference, LLMFeedbackResponse
from uedrlhf.config import BASE_DIR, JSON_INSTRUCTION, LLM_CONFIG

load_dotenv(BASE_DIR / ".env")


class GeminiProvider(LLMProvider):

    def __init__(self, config: dict):
        self.model_name = config.get("model", "gemini-2.5-flash")
        api_key_env = config.get("api_key_env", "GEMINI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"Gemini API key not found. Set the {api_key_env} environment variable "
                f"or add it to .env"
            )
        self._client = genai.Client(api_key=api_key)

    def _upload_image(self, img_path: Path) -> types.Part:
        """Upload an image to the Gemini File API and return a Part."""
        uploaded = self._client.files.upload(
            file=str(img_path),
            config=types.UploadFileConfig(mime_type="image/png"),
        )
        return types.Part.from_uri(file_uri=uploaded.uri, mime_type="image/png")

    def get_preference(self, img_a_path: Path, img_b_path: Path, prompt: str) -> dict:
        part_a = self._upload_image(img_a_path)
        part_b = self._upload_image(img_b_path)

        contents = [
            "Image A (Left):",
            part_a,
            "Image B (Right):",
            part_b,
            prompt,
        ]

        t0 = time.perf_counter()

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        time_ms = (time.perf_counter() - t0) * 1000
        raw = response.text

        parsed = LLMFeedbackResponse.model_validate_json(raw)
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
        """Build a Gemini Batch API request line.

        The format follows the Gemini Batch API specification:
        https://ai.google.dev/gemini-api/docs/batch
        """
        b64_a = img_to_base64(img_a_path)
        b64_b = img_to_base64(img_b_path)

        generation_config = {
            "responseMimeType": "application/json",
        }
        
        thinking_level = LLM_CONFIG.get("thinking_level", None)
        if thinking_level:
            generation_config["thinkingConfig"] = {
                "thinkingLevel": thinking_level.upper()
            }

        return {
            "key": key,
            "request": {
                "model": f"models/{self.model_name}",
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": "Image A (Left):"},
                            {"inline_data": {"mime_type": "image/png", "data": b64_a}},
                            {"text": "Image B (Right):"},
                            {"inline_data": {"mime_type": "image/png", "data": b64_b}},
                            {"text": prompt},
                        ],
                    }
                ],
                "generationConfig": generation_config,
            },
        }
