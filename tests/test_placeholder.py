"""Placeholder tests — Phase 0 scaffolding verification.
Real tests come in each phase when modules are implemented."""

from pathlib import Path


def test_project_structure():
    """Verify essential directories exist."""
    root = Path(__file__).resolve().parent.parent
    assert (root / "src").is_dir()
    assert (root / "config").is_dir()
    assert (root / "scripts").is_dir()
    assert (root / "tests").is_dir()
    assert (root / "data" / "raw").is_dir()


def test_config_import():
    """Verify config module can be imported without errors."""
    from src.config import get_settings, settings

    assert settings is not None
    assert get_settings() is settings  # singleton


def test_helpers_import():
    """Verify utility functions are importable."""
    from src.utils.helpers import is_arabic_text, truncate_text

    # max_length=5: 2 chars + "..." = 5 total
    assert truncate_text("hello world", 5) == "he..."
    assert is_arabic_text("مرحبا") is True
    assert is_arabic_text("hello") is False


def test_logging_import():
    """Verify logging setup works."""
    from src.utils.logging import get_logger

    logger = get_logger("test")
    assert logger.name == "test"


def test_llm_base_import():
    """Verify LLM base class is importable."""
    from src.llm.base import BaseLLM

    # Can't instantiate abstract class directly—just check it exists
    assert "invoke" in BaseLLM.__abstractmethods__


def test_placeholder_pass():
    """Ensure pytest is working correctly."""
    assert True
