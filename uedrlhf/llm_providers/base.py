"""Abstract base class for LLM providers.

Every LLM provider (Gemini, Open WebUI, etc.) must implement this
interface so that feedback scripts can work with any provider
without knowing its internals.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class LLMProvider(ABC):
    """Contract that every LLM provider must fulfill."""

    @abstractmethod
    def get_preference(self, img_a_path: Path, img_b_path: Path, prompt: str) -> dict:
        """Send two environment images and a prompt to the LLM.

        Args:
            img_a_path: Path to image A (left).
            img_b_path: Path to image B (right).
            prompt: The full evaluation prompt.

        Returns:
            A dict with keys:
                "preference": float (0.0 = A, 1.0 = B, 0.5 = TIE)
                "x_pct": float (0-100)
                "y_pct": float (0-100)
                "time_ms": float (response latency in ms)
                "raw": str (raw LLM response text)
        """
        ...

    @abstractmethod
    def build_batch_request(
        self,
        img_a_path: Path,
        img_b_path: Path,
        prompt: str,
        key: str,
    ) -> dict:
        """Build a single batch request line for batch processing APIs.

        Args:
            img_a_path: Path to image A.
            img_b_path: Path to image B.
            prompt: The full evaluation prompt.
            key: Unique identifier for this request (e.g. "pair_42_swapped_0").

        Returns:
            A dict suitable for writing as a JSONL line.
        """
        ...
