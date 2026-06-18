"""Tests for sentence-transformers embedding wrapper."""

import numpy as np

from src.config import EmbeddingsConfig
from src.embeddings.embedder import Embedder


def test_embedder_import():
    """Verify Embedder can be imported."""
    from src.embeddings import Embedder

    assert Embedder is not None


def test_embedder_init():
    """Verify Embedder initializes and loads model."""
    config = EmbeddingsConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedder = Embedder(config)
    assert embedder.model is not None
    assert embedder.dim > 0


def test_embedder_embed_texts():
    """Verify embed_texts returns correct shape."""
    config = EmbeddingsConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedder = Embedder(config)
    texts = ["Hello world", "Test sentence two"]
    embeddings = embedder.embed_texts(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == embedder.dim
    assert len(embeddings[1]) == embedder.dim


def test_embedder_embed_query():
    """Verify embed_query returns a single vector."""
    config = EmbeddingsConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedder = Embedder(config)
    embedding = embedder.embed_query("What is RAG?")
    assert len(embedding) == embedder.dim


def test_embedder_arabic():
    """Verify Arabic text is handled."""
    config = EmbeddingsConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedder = Embedder(config)
    embedding = embedder.embed_query("ما هو الذكاء الاصطناعي؟")
    assert len(embedding) == embedder.dim


def test_embedder_dimension():
    """Verify embedding dimension matches expected."""
    config = EmbeddingsConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedder = Embedder(config)
    assert embedder.dim == 384  # This model outputs 384-dim vectors


def test_embedder_normalized():
    """Verify embeddings are unit-normalized (cosine ready)."""
    config = EmbeddingsConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        normalize_embeddings=True,
    )
    embedder = Embedder(config)
    embedding = embedder.embed_query("test")
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-5
