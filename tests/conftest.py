"""Shared test fixtures and configuration."""

import sys
from pathlib import Path

import pytest

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def data_dir() -> Path:
    """Path to test fixture directory."""
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_pdf_en(data_dir: Path) -> Path:
    """Path to English sample PDF."""
    return data_dir / "sample_arxiv.pdf"


@pytest.fixture
def sample_pdf_ar(data_dir: Path) -> Path:
    """Path to Arabic sample PDF."""
    return data_dir / "sample_ar.pdf"
