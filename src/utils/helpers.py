"""Common utility functions."""

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file (for cache invalidation)."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(filepath: Path) -> Any:
    """Load JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: Path, indent: int = 2) -> None:
    """Save data to JSON file."""
    ensure_dir(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for display."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def is_arabic_text(text: str) -> bool:
    """
    Heuristic: check if text contains Arabic characters.
    Returns True if >30% chars are in Arabic Unicode blocks.
    """
    if not text:
        return False
    arabic_chars = sum(
        1
        for c in text
        if "\u0600" <= c <= "\u06ff"  # Arabic
        or "\u0750" <= c <= "\u077f"  # Arabic Supplement
        or "\u08a0" <= c <= "\u08ff"  # Arabic Extended-A
        or "\ufb50" <= c <= "\ufdff"  # Arabic Presentation Forms-A
        or "\ufe70" <= c <= "\ufeff"  # Arabic Presentation Forms-B
    )
    return (arabic_chars / len(text)) > 0.3
