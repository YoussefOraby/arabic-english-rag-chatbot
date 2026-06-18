"""Tests for cross-encoder reranker (with mock)."""

from unittest.mock import patch

from src.embeddings.store import SearchResult
from src.pdf.chunker import Chunk
from src.retrieval.reranker import Reranker


def _make_search_result(chunk_id: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(text=text, page_num=1, source_file="test.pdf", chunk_id=chunk_id),
        score=score,
    )


def test_reranker_empty_candidates():
    r = Reranker("fake-model")
    assert r.rerank("query", []) == []


@patch("sentence_transformers.CrossEncoder")
def test_reranker_orders_by_score(mock_ce):
    # Mock predict to return scores
    instance = mock_ce.return_value
    instance.predict.return_value = [0.1, 0.9, 0.5]

    r = Reranker("fake-model")
    candidates = [
        _make_search_result("a", "text aaa", 0.5),
        _make_search_result("b", "text bbb", 0.5),
        _make_search_result("c", "text ccc", 0.5),
    ]
    result = r.rerank("query", candidates, top_k=3)
    assert len(result) == 3
    assert result[0].chunk.chunk_id == "b"
    assert result[1].chunk.chunk_id == "c"
    assert result[2].chunk.chunk_id == "a"


@patch("sentence_transformers.CrossEncoder")
def test_reranker_top_k_limits(mock_ce):
    instance = mock_ce.return_value
    instance.predict.return_value = [0.9, 0.1, 0.5]

    r = Reranker("fake-model")
    candidates = [
        _make_search_result("a", "text aaa", 0.5),
        _make_search_result("b", "text bbb", 0.5),
        _make_search_result("c", "text ccc", 0.5),
    ]
    result = r.rerank("query", candidates, top_k=2)
    assert len(result) == 2


@patch("sentence_transformers.CrossEncoder")
def test_reranker_scores_replaced(mock_ce):
    instance = mock_ce.return_value
    instance.predict.return_value = [0.8, 0.3]

    r = Reranker("fake-model")
    candidates = [
        _make_search_result("a", "text aaa", 0.5),
        _make_search_result("b", "text bbb", 0.5),
    ]
    result = r.rerank("query", candidates, top_k=2)
    assert result[0].score == 0.8
    assert result[1].score == 0.3


def test_reranker_lazy_load():
    """Verify model is None before first call."""
    r = Reranker("fake-model")
    assert r._model is None


@patch("sentence_transformers.CrossEncoder")
def test_reranker_ensure_loaded(mock_ce):
    r = Reranker("fake-model")
    r.ensure_loaded()
    assert r._model is not None
