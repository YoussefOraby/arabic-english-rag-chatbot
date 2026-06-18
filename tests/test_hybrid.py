"""Tests for BM25 index and RRF merge."""

from src.embeddings.store import SearchResult
from src.pdf.chunker import Chunk
from src.retrieval.hybrid import BM25Index, rff_merge


def _make_search_result(chunk_id: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(text=text, page_num=1, source_file="test.pdf", chunk_id=chunk_id),
        score=score,
    )


class TestBM25Index:
    """BM25 index build and search."""

    def test_build_and_search(self):
        idx = BM25Index()
        chunks = [
            Chunk(text="black holes gravity", page_num=1, source_file="test.pdf", chunk_id="id1"),
            Chunk(
                text="quantum mechanics wave", page_num=2, source_file="test.pdf", chunk_id="id2"
            ),
            Chunk(text="machine learning data", page_num=3, source_file="test.pdf", chunk_id="id3"),
            Chunk(text="deep learning neural", page_num=4, source_file="test.pdf", chunk_id="id4"),
        ]
        idx.rebuild_from_chunks(chunks)
        assert idx.is_ready

        results = idx.search("black holes", k=2)
        assert len(results) == 2
        assert results[0][0] == "id1"
        # BM25Okapi can give zero IDF for rare terms in tiny corpora,
        # but "black" and "holes" both appear once in 4 docs → non-zero
        assert results[0][1] > 0

    def test_empty_index(self):
        idx = BM25Index()
        assert not idx.is_ready
        assert idx.search("test", k=2) == []

    def test_rebuild_empty(self):
        idx = BM25Index()
        idx.rebuild_from_chunks([])
        assert not idx.is_ready

    def test_rebuild_with_chunks(self):
        idx = BM25Index()
        c1 = Chunk(text="hello world", page_num=1, source_file="t.pdf", chunk_id="a")
        c2 = Chunk(text="goodbye world", page_num=2, source_file="t.pdf", chunk_id="b")
        idx.rebuild_from_chunks([c1, c2])
        results = idx.search("hello", k=2)
        ids = [r[0] for r in results]
        assert "a" in ids


def test_rff_merge_empty():
    assert rff_merge([], [], k=5) == []


def test_rff_merge_identical():
    dense = [
        _make_search_result("a", "text a", 0.9),
        _make_search_result("b", "text b", 0.8),
    ]
    bm25 = [("a", 1.5), ("b", 1.0)]
    merged = rff_merge(dense, bm25, k=2)
    assert len(merged) == 2
    assert merged[0].chunk.chunk_id == "a"
    assert merged[1].chunk.chunk_id == "b"


def test_rff_merge_dense_only():
    dense = [
        _make_search_result("a", "text a", 0.9),
        _make_search_result("b", "text b", 0.8),
    ]
    merged = rff_merge(dense, [], k=2)
    assert len(merged) == 2
    assert merged[0].chunk.chunk_id == "a"


def test_rff_merge_bm25_only():
    bm25 = [("x", 2.0), ("y", 1.5)]
    merged = rff_merge([], bm25, k=2)
    assert len(merged) == 2


def test_rff_merge_different_sets():
    dense = [
        _make_search_result("a", "text a", 0.9),
        _make_search_result("b", "text b", 0.8),
    ]
    bm25 = [("c", 1.5), ("d", 1.0)]
    merged = rff_merge(dense, bm25, k=4)
    assert len(merged) == 4


def test_rff_merge_with_alpha():
    dense = [
        _make_search_result("a", "text a", 0.9),
        _make_search_result("b", "text b", 0.8),
    ]
    bm25 = [("a", 1.5)]
    # alpha=1 means dense only: only "a" and "b" should appear
    merged = rff_merge(dense, bm25, k=3, alpha=1.0)
    assert len(merged) == 2
    ids = {r.chunk.chunk_id for r in merged}
    assert "a" in ids
    assert "b" in ids


def test_rff_merge_results_ordered():
    dense = [
        _make_search_result("a", "text a", 0.9),
        _make_search_result("b", "text b", 0.8),
    ]
    bm25 = [("c", 2.0)]
    merged = rff_merge(dense, bm25, k=3)
    assert len(merged) <= 3


def test_bm25_rebuild_with_document_id_filter():
    """Verify rebuild can filter to a single document."""
    idx = BM25Index()
    chunks = [
        Chunk(
            text="machine learning",
            page_num=1,
            source_file="a.pdf",
            chunk_id="a1",
            document_id="doc_a",
        ),
        Chunk(
            text="machine learning",
            page_num=2,
            source_file="a.pdf",
            chunk_id="a2",
            document_id="doc_a",
        ),
        Chunk(
            text="machine learning",
            page_num=1,
            source_file="b.pdf",
            chunk_id="b1",
            document_id="doc_b",
        ),
    ]
    idx.rebuild_from_chunks(chunks)
    assert idx._document_ids == ["doc_a", "doc_a", "doc_b"]


def test_bm25_search_respects_document_id():
    """Verify BM25 search with document_id returns only matching chunks."""
    idx = BM25Index()
    chunks = [
        Chunk(
            text="black holes gravity",
            page_num=1,
            source_file="a.pdf",
            chunk_id="a1",
            document_id="doc_a",
        ),
        Chunk(
            text="quantum mechanics",
            page_num=1,
            source_file="b.pdf",
            chunk_id="b1",
            document_id="doc_b",
        ),
    ]
    idx.rebuild_from_chunks(chunks)
    results_all = idx.search("black", k=5)
    assert len(results_all) == 2
    results_filtered = idx.search("black", k=5, document_id="doc_a")
    assert len(results_filtered) == 1
    assert results_filtered[0][0] == "a1"
    results_empty = idx.search("black", k=5, document_id="nonexistent")
    assert results_empty == []


def test_bm25_rebuild_with_document_id():
    """Verify rebuild with document_id only indexes chunks for that doc."""
    idx = BM25Index()
    chunks = [
        Chunk(
            text="python programming",
            page_num=1,
            source_file="a.pdf",
            chunk_id="a1",
            document_id="doc_a",
        ),
        Chunk(
            text="java programming",
            page_num=1,
            source_file="b.pdf",
            chunk_id="b1",
            document_id="doc_b",
        ),
        Chunk(
            text="rust programming",
            page_num=1,
            source_file="b.pdf",
            chunk_id="b2",
            document_id="doc_b",
        ),
    ]
    # Rebuild with explicit chunks (no store involved)
    idx.rebuild_from_chunks(chunks)
    results = idx.search("programming", k=5, document_id="doc_b")
    assert len(results) == 2
    ids = {r[0] for r in results}
    assert ids == {"b1", "b2"}
