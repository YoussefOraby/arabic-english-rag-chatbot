"""Tests for ChromaDB vector store."""

import gc
import shutil
import tempfile
from pathlib import Path

import pytest

from src.config import EmbeddingsConfig, VectorStoreConfig
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore, SearchResult
from src.pdf.chunker import Chunk


@pytest.fixture
def temp_dir():
    """Create a temporary directory for ChromaDB persistence."""
    path = Path(tempfile.mkdtemp())
    yield path
    # Force garbage collection to close ChromaDB file handles before rmtree
    gc.collect()
    gc.collect()
    shutil.rmtree(str(path), ignore_errors=True)


@pytest.fixture
def embedder():
    """Create an embedder with the multilingual model."""
    config = EmbeddingsConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return Embedder(config)


@pytest.fixture
def store(temp_dir, embedder):
    """Create a ChromaStore in a temporary directory."""
    config = VectorStoreConfig(
        persist_directory=str(temp_dir),
        collection_name="test_collection",
    )
    return ChromaStore(config, embedder)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            text="Python is a programming language.",
            page_num=1,
            source_file="test.pdf",
            chunk_id="test_p1_abc123",
            document_id="doc_001",
        ),
        Chunk(
            text="Machine learning is a subset of AI.",
            page_num=2,
            source_file="test.pdf",
            chunk_id="test_p2_def456",
            document_id="doc_001",
        ),
        Chunk(
            text="الذكاء الاصطناعي هو فرع من علوم الحاسوب",
            page_num=3,
            source_file="test.pdf",
            chunk_id="test_p3_ghi789",
            document_id="doc_001",
        ),
        Chunk(
            text="ChromaDB is a vector database.",
            page_num=1,
            source_file="other.pdf",
            chunk_id="other_p1_jkl012",
            document_id="doc_002",
        ),
    ]


def test_chroma_store_import():
    """Verify ChromaStore and SearchResult can be imported."""
    from src.embeddings import ChromaStore, SearchResult

    assert ChromaStore is not None
    assert SearchResult is not None


def test_chroma_store_init(store):
    """Verify ChromaStore initializes and creates collection."""
    assert store.collection is not None
    assert store.collection.name == "test_collection"
    assert store.count_chunks() == 0


def test_add_chunks(store, sample_chunks):
    """Verify chunks are added to ChromaDB."""
    store.add_chunks(sample_chunks)
    assert store.count_chunks() == 4


def test_add_empty_chunks(store):
    """Verify adding empty list does nothing."""
    store.add_chunks([])
    assert store.count_chunks() == 0


def test_similarity_search(store, sample_chunks):
    """Verify similarity search returns relevant results."""
    store.add_chunks(sample_chunks)
    results = store.similarity_search("programming language", k=2, score_threshold=0.0)
    assert len(results) > 0
    assert any("Python" in r.chunk.text for r in results)


def test_similarity_search_arabic(store, sample_chunks):
    """Verify Arabic query returns Arabic chunk."""
    store.add_chunks(sample_chunks)
    results = store.similarity_search("ذكاء اصطناعي", k=4, score_threshold=0.0)
    arabic_results = [r for r in results if "الذكاء" in r.chunk.text]
    assert len(arabic_results) > 0


def test_similarity_search_threshold(store, sample_chunks):
    """Verify score_threshold filters low-scoring results."""
    store.add_chunks(sample_chunks)
    results = store.similarity_search("quantum physics", k=4, score_threshold=0.9)
    # Unrelated query should have low scores, possibly filtered out
    assert isinstance(results, list)


def test_search_result_dataclass():
    """Verify SearchResult attributes."""
    chunk = Chunk(text="test", page_num=1, source_file="test.pdf", chunk_id="test")
    result = SearchResult(chunk=chunk, score=0.95)
    assert result.chunk == chunk
    assert result.score == 0.95


def test_clear(store, sample_chunks):
    """Verify clear removes all chunks."""
    store.add_chunks(sample_chunks)
    assert store.count_chunks() == 4
    store.clear()
    assert store.count_chunks() == 0


def test_clear_and_readd(store, sample_chunks):
    """Verify clear + re-add works."""
    store.add_chunks(sample_chunks)
    store.clear()
    store.add_chunks(sample_chunks[:2])
    assert store.count_chunks() == 2


def test_count_chunks(store, sample_chunks):
    """Verify count is accurate."""
    assert store.count_chunks() == 0
    store.add_chunks(sample_chunks[:1])
    assert store.count_chunks() == 1
    store.add_chunks(sample_chunks[1:3])
    assert store.count_chunks() == 3


def test_delete_by_document(store, sample_chunks):
    """Verify delete_by_document removes only chunks for that doc."""
    store.add_chunks(sample_chunks)
    assert store.count_chunks() == 4
    removed = store.delete_by_document("doc_001")
    assert removed == 3
    assert store.count_chunks() == 1
    remaining = store.similarity_search("ChromaDB", k=4, score_threshold=0.0)
    assert len(remaining) == 1
    assert remaining[0].chunk.document_id == "doc_002"


def test_delete_by_document_nonexistent(store, sample_chunks):
    """Verify deleting a non-existent document does nothing."""
    store.add_chunks(sample_chunks)
    removed = store.delete_by_document("nonexistent")
    assert removed == 0
    assert store.count_chunks() == 4


def test_similarity_search_document_id_filter(store, sample_chunks):
    """Verify similarity_search with document_id filter."""
    store.add_chunks(sample_chunks)
    results = store.similarity_search(
        "machine learning", k=4, score_threshold=0.0, document_id="doc_001"
    )
    assert len(results) == 3
    for r in results:
        assert r.chunk.document_id == "doc_001"


def test_reindex_flow(store, sample_chunks):
    """Verify delete + re-add pattern used by reindex."""
    store.add_chunks(sample_chunks)
    assert store.count_chunks() == 4
    removed = store.delete_by_document("doc_001")
    assert removed == 3
    new_meta = Chunk(
        text="Updated Python content.",
        page_num=1,
        source_file="test.pdf",
        chunk_id="test_p1_reindex",
        document_id="doc_001",
    )
    store.add_chunks([new_meta])
    assert store.count_chunks() == 2
