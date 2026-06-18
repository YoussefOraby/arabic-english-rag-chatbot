"""Ollama LLM implementation using the official ollama Python client.
Connects to a local Ollama server for inference."""

from collections.abc import Generator
from typing import Any

import ollama

from src.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    """LLM backend using local Ollama server."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        num_ctx: int = 4096,
        timeout: int = 120,
    ):
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._timeout = timeout
        ollama.host = base_url

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Send prompt to Ollama and get the complete response."""
        options = {
            "temperature": kwargs.get("temperature", self._temperature),
            "num_ctx": kwargs.get("num_ctx", self._num_ctx),
        }
        try:
            response = ollama.chat(
                model=kwargs.get("model", self._model),
                messages=[{"role": "user", "content": prompt}],
                options=options,
            )
            return response["message"]["content"]
        except ollama.ResponseError as e:
            raise RuntimeError(f"Ollama error: {e.error}") from e
        except Exception as e:
            raise ConnectionError(
                f"Ollama server not reachable at {self._base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            ) from e

    def invoke_stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, str]:
        """
        Send prompt to Ollama and yield tokens as they arrive.

        Yields:
            Each text token from the stream.

        Returns:
            The complete response text (after generator exhausts).
        """
        options = {
            "temperature": kwargs.get("temperature", self._temperature),
            "num_ctx": kwargs.get("num_ctx", self._num_ctx),
        }
        try:
            stream = ollama.chat(
                model=kwargs.get("model", self._model),
                messages=[{"role": "user", "content": prompt}],
                options=options,
                stream=True,
            )
            full = ""
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                full += token
                yield token
            return full
        except ollama.ResponseError as e:
            raise RuntimeError(f"Ollama error: {e.error}") from e
        except Exception as e:
            raise ConnectionError(
                f"Ollama server not reachable at {self._base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            ) from e

    def is_available(self) -> bool:
        try:
            ollama.list()
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"Ollama/{self._model}"
