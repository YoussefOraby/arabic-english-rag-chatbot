"""Tests for pyramids.pdf chunk output."""
import json
from pathlib import Path

CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "pyramids_chunks.json"


def test_chunks_json_exists():
    assert CHUNKS_PATH.exists(), f"{CHUNKS_PATH} not found"


def test_chunks_count():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    assert len(chunks) == 321, f"Expected 321 chunks, got {len(chunks)}"


def test_no_empty_text():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    empty = [c for c in chunks if not c.get("text", "").strip()]
    assert not empty, f"{len(empty)} chunks have empty text"


def test_metadata_present():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    missing = []
    for i, c in enumerate(chunks):
        if c.get("page_num") is None:
            missing.append((i, "page_num"))
        if not c.get("source_file"):
            missing.append((i, "source_file"))
    assert not missing, f"Missing metadata: {missing}"
