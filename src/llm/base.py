"""Base LLM interface — all LLM implementations inherit from this.
Provides a uniform interface regardless of backend (Ollama, Gemini, etc.)."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        Send a prompt to the LLM and get a response.

        Args:
            prompt: The input text/prompt
            **kwargs: Model-specific parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text response
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM backend is reachable.

        Returns:
            True if the model is ready for inference
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this LLM (e.g., 'Ollama/llama3.1')."""
        ...
