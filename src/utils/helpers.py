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


# Arabic diacritics (tashkeel) to strip for retrieval normalization
_ARABIC_DIACRITICS = dict.fromkeys(
    ord(c) for c in "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u0653\u0654\u0655"
)
_TATWEEL = "\u0640"


def normalize_arabic_for_retrieval(text: str) -> str:
    """Normalize Arabic text for retrieval matching only (not for display).

    Operations:
      - Strip diacritics (tashkeel)
      - Remove tatweel (kashida)
      - Unify alef forms: أ إ آ → ا
      - Unify yeh: ى → ي
      - Unify teh marbuta: ة → ه
    """
    text = text.translate(_ARABIC_DIACRITICS)
    text = text.replace(_TATWEEL, "")
    text = text.replace("\u0623", "\u0627").replace("\u0625", "\u0627").replace("\u0622", "\u0627")
    text = text.replace("\u0649", "\u064a")  # ى → ي
    text = text.replace("\u0629", "\u0647")  # ة → ه
    return text


def extract_arabic_keywords(text: str) -> set[str]:
    """Extract significant Arabic keywords from text for retrieval boosting.

    Returns normalized (for retrieval) keywords from the text.
    """
    if not text:
        return set()
    norm = normalize_arabic_for_retrieval(text)
    # Arabic "stop words" to filter out
    stop_words = {
        "في", "من", "إلى", "عن", "على", "مع", "هل", "ما", "هو", "هي",
        "هم", "هن", "كان", "هذا", "هذه", "ذلك", "تلك", "ال", "التي",
        "الذين", "الذي", "ل", "ب", "ك", "و", "ا", "ف", "س",
        "قد", "لم", "لن", "ان", "أن", "إن", "لا", "ما",
    }
    tokens = set()
    for word in norm.split():
        word = word.strip()
        # Remove leading/trailing non-word chars
        word = word.strip(".,:;!?()[]{}\"'«»؟")
        if len(word) > 1 and word not in stop_words:
            tokens.add(word)
    return tokens


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
