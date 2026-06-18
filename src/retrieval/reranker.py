"""Cross-encoder reranker for re-scoring retrieved chunks."""

from src.config import settings
from src.embeddings.store import SearchResult


class Reranker:
    """Lazy-loaded cross-encoder reranker.

    Model is loaded on first call to ``rerank`` (or ``ensure_loaded``),
    so it does not block application startup.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.retrieval.reranker_model
        self._model = None

    def ensure_loaded(self) -> None:
        """Force model load (useful for pre-warming)."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """Re-score candidate chunks with cross-encoder and return top-k.

        If no candidates, returns empty list.
        """
        if not candidates:
            return []

        self.ensure_loaded()
        top_k = top_k or len(candidates)

        pairs = [(query, r.chunk.text) for r in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(candidates, scores, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )

        # Assign new scores and strip excess
        result = []
        for r, score in ranked[:top_k]:
            r.score = float(score)
            result.append(r)

        return result
