"""Tests for the Ollama LLM implementation.

All tests skip if Ollama is not running.
"""

import pytest

from src.llm.ollama_llm import OllamaLLM

ollama_available = OllamaLLM().is_available()


@pytest.mark.skipif(not ollama_available, reason="Ollama not running")
def test_ollama_llm_import():
    """Verify OllamaLLM can be imported."""
    from src.llm.base import BaseLLM
    from src.llm.ollama_llm import OllamaLLM

    assert issubclass(OllamaLLM, BaseLLM)


@pytest.mark.skipif(not ollama_available, reason="Ollama not running")
def test_ollama_llm_init():
    """Verify OllamaLLM initialization with default model."""
    from src.llm.ollama_llm import OllamaLLM

    llm = OllamaLLM()
    assert llm.name == "Ollama/llama3.2:3b"
    assert llm.is_available() is True


@pytest.mark.skipif(not ollama_available, reason="Ollama not running")
def test_ollama_llm_invoke():
    """Verify OllamaLLM can generate a response."""
    from src.llm.ollama_llm import OllamaLLM

    llm = OllamaLLM()
    response = llm.invoke("Say hello in one word")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.skipif(not ollama_available, reason="Ollama not running")
def test_ollama_llm_arabic():
    """Verify OllamaLLM handles Arabic."""
    from src.llm.ollama_llm import OllamaLLM

    llm = OllamaLLM()
    response = llm.invoke("قل مرحبا بكلمة واحدة")
    assert isinstance(response, str)
    assert len(response) > 0
