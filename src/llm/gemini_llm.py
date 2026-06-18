"""Gemini LLM implementation using google-genai package (new API).
Connects to Google Gemini API free tier as fallback."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.llm.base import BaseLLM


class GeminiLLM(BaseLLM):
    """LLM backend using Google Gemini API (free tier)."""

    def __init__(
        self, model: str = "gemini-2.0-flash", api_key: str | None = None, temperature: float = 0.1
    ):
        """
        Initialize Gemini client using google.genai.

        Args:
            model: Gemini model name (gemini-2.0-flash or gemini-1.5-flash)
            api_key: Google API key (or set GEMINI_API_KEY env var)
            temperature: Response randomness (0-1)
        """
        self._model = model
        self._api_key = api_key or self._load_api_key()
        self._temperature = temperature
        self._client = None

        if self._api_key:
            try:
                from google import genai

                self._client = genai.Client(api_key=self._api_key)
            except Exception as e:
                print(f"[!] Failed to initialize Gemini client: {e}")

    def _load_api_key(self) -> str | None:
        """Load API key from .env file."""
        # Try loading from .env in project root
        env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(str(env_path))
        return os.getenv("GEMINI_API_KEY")

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        Send prompt to Gemini and get response.

        Args:
            prompt: Input text
            **kwargs: Override defaults (temperature, etc.)

        Returns:
            Generated text response

        Raises:
            ValueError: If API key is not set or client not initialized
            ConnectionError: If API is unreachable
        """
        if not self._client:
            if not self._api_key:
                raise ValueError("GEMINI_API_KEY not set in .env")
            raise ConnectionError(
                "Gemini client not initialized. Check API key and internet connection."
            )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                raise ConnectionError(
                    "Gemini API quota exhausted. Wait a minute or use a new API key."
                )
            raise ConnectionError(f"Gemini API error: {e}")

    def is_available(self) -> bool:
        """Check if API key is configured and client initialized."""
        return self._client is not None and self._api_key is not None

    @property
    def name(self) -> str:
        return f"Gemini/{self._model}"
