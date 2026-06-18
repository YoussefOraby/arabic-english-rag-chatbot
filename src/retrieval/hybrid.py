"""BM25 sparse retrieval index with Reciprocal Rank Fusion (RRF) for hybrid search."""

from rank_bm25 import BM25Okapi

from src.embeddings.store import ChromaStore, SearchResult
from src.pdf.chunker import Chunk


class BM25Index:
    """BM25Okapi index built from ChromaDB chunk texts.

    Must be rebuilt (``rebuild``) after new documents are added.
    Supports optional ``document_id`` filtering on both rebuild and search.
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._document_ids: list[str] = []

    def rebuild(self, store: ChromaStore, document_id: str | None = None) -> None:
        """Fetch all chunks from ChromaDB and build the BM25 index.

        If ``document_id`` is given, only chunks for that document are indexed.
        """
        chunks = store.get_all_chunks()
        if document_id:
            chunks = [c for c in chunks if c.document_id == document_id]
        if not chunks:
            self.bm25 = None
            self._ids = []
            self._texts = []
            self._document_ids = []
            return

        self._texts = [c.text for c in chunks]
        self._ids = [c.chunk_id for c in chunks]
        self._document_ids = [c.document_id for c in chunks]
        tokenized = [self._tokenize(t) for t in self._texts]
        self.bm25 = BM25Okapi(tokenized)

    def rebuild_from_chunks(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from an explicit chunk list."""
        if not chunks:
            self.bm25 = None
            self._ids = []
            self._texts = []
            self._document_ids = []
            return

        self._texts = [c.text for c in chunks]
        self._ids = [c.chunk_id for c in chunks]
        self._document_ids = [c.document_id for c in chunks]
        tokenized = [self._tokenize(t) for t in self._texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int, document_id: str | None = None) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, score) pairs from BM25.

        If ``document_id`` is given, only results for that document are returned.
        Returns empty list if no index is built.
        """
        if self.bm25 is None or not self._ids:
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Build indexed list, optionally filtering by document_id
        indexed = []
        for idx, s in enumerate(scores):
            if document_id and self._document_ids[idx] != document_id:
                continue
            indexed.append((idx, float(s)))

        indexed.sort(key=lambda x: x[1], reverse=True)
        results = [(self._ids[idx], float(score)) for idx, score in indexed[:k]]
        return results

    @property
    def is_ready(self) -> bool:
        return self.bm25 is not None

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer (works for Ar + En)."""
        import re

        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if len(t) > 1]


def rff_merge(
    dense_results: list[SearchResult],
    bm25_id_scores: list[tuple[str, float]],
    k: int,
    alpha: float = 0.5,
) -> list[SearchResult]:
    """Merge dense and BM25 results using weighted Reciprocal Rank Fusion.

    ``dense_results``: list from ChromaStore.similarity_search
    ``bm25_id_scores``: list of (chunk_id, score) from BM25Index.search
    ``k``: number of final results to return

    Returns:
        Merged list of SearchResult (ranked by RRF score).
    """
    dense_ids = {r.chunk.chunk_id for r in dense_results}
    bm25_ids = {sid for sid, _ in bm25_id_scores}

    merged_ids = dense_ids | bm25_ids
    if not merged_ids:
        return []

    # Build rank positions (0-indexed)
    dense_rank = {r.chunk.chunk_id: idx for idx, r in enumerate(dense_results)}
    bm25_rank = {sid: idx for idx, (sid, _) in enumerate(bm25_id_scores)}

    scored = []
    for cid in merged_ids:
        dr = dense_rank.get(cid, 1000)
        br = bm25_rank.get(cid, 1000)

        # RRF with alpha weight: higher weight for dense
        rrf_score = alpha * (1.0 / (60 + dr)) + (1.0 - alpha) * (1.0 / (60 + br))
        scored.append((cid, rrf_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = {sid for sid, _ in scored[:k]}

    # Return original SearchResult objects preserving dense score for non-hybrid display
    seen = set()
    result: list[SearchResult] = []
    # Dense results first (preserve their SearchResult objects)
    for r in dense_results:
        if r.chunk.chunk_id in top_ids and r.chunk.chunk_id not in seen:
            seen.add(r.chunk.chunk_id)
            result.append(r)
    # Then BM25-only results (create SearchResult with score=0)
    for sid, _ in bm25_id_scores:
        if sid in top_ids and sid not in seen:
            seen.add(sid)
            result.append(
                SearchResult(
                    chunk=Chunk(chunk_id=sid, text="", page_num=0, source_file=""), score=0.0
                )
            )

    return result
