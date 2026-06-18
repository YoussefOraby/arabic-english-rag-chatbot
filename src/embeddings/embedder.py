"""Sentence-transformers wrapper for generating embeddings."""

from sentence_transformers import SentenceTransformer

from src.config import EmbeddingsConfig


class Embedder:
    """Wrapper around sentence-transformers for document and query embeddings."""

    def __init__(self, config: EmbeddingsConfig):
        self._model: SentenceTransformer | None = None
        self.config = config
        self._load_model()

    def _load_model(self) -> None:
        self._model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._load_model()
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings into vectors."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )
        return embedding.tolist()

    @property
    def dim(self) -> int:
        """Return embedding dimension."""
        return self.model.get_embedding_dimension()
