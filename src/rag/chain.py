"""RAG chain: query → retrieve → prompt → generate → answer with citations.

Supports both sync (``query``) and streaming (``query_stream``) modes.
"""

import re
from collections.abc import Generator
from typing import Any

from src.config import settings
from src.embeddings.store import ChromaStore, SearchResult
from src.llm.base import BaseLLM
from src.rag.prompts import format_chunks_for_context, format_history, get_system_prompt
from src.retrieval.hybrid import BM25Index, rff_merge
from src.retrieval.reranker import Reranker


class RAGChain:
    """
    Retrieval-Augmented Generation chain.

    Pipeline:
        1. Embed user query
        2. Retrieve top-K chunks from ChromaDB (BM25 hybrid if enabled)
        3. Optionally rerank with cross-encoder
        4. Build prompt with context + question
        5. Send to LLM
        6. Parse citations from response
    """

    def __init__(self, store: ChromaStore, llm: BaseLLM):
        self.store = store
        self.llm = llm
        self.prompt_template = settings.rag.system_prompt_template
        self.top_k = settings.retrieval.top_k
        self.max_tokens = settings.rag.max_tokens

        # Lazy-built retrieval components
        self._bm25: BM25Index | None = None
        self._reranker: Reranker | None = None
        self._bm25_built = False

    def rebuild_index(self) -> None:
        """Force rebuild of BM25 index after new chunks are added to the store."""
        if not settings.retrieval.enable_hybrid:
            return
        if self._bm25 is None:
            self._bm25 = BM25Index()
        self._bm25.rebuild(self.store)
        self._bm25_built = True

    def query(
        self,
        question: str,
        history: list[dict] | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Answer a question using RAG (non-streaming).

        Returns:
            Dict with:
                - "answer": Generated answer text with [page X] citations
                - "chunks": List of retrieved chunks (for debugging)
                - "source_documents": Dict of filename -> set of page_numbers
                - "citations": List of citation dicts with verification
                - "insufficient_data": True if retrieval evidence was too weak
        """
        chunks = self._retrieve(question, document_id=document_id)

        insufficient = (
            len(chunks) == 0 or max(c.score for c in chunks) < settings.retrieval.score_threshold
        )

        if insufficient:
            return {
                "answer": "Insufficient data to verify from the uploaded documents.",
                "chunks": [],
                "source_documents": {},
                "citations": [],
                "insufficient_data": True,
            }

        prompt = self._build_prompt(question, chunks, history=history)

        raw_answer = self.llm.invoke(prompt, max_tokens=self.max_tokens)

        answer, citations = self._parse_citations(raw_answer)

        source_documents: dict[str, set[int]] = {}
        for c in chunks:
            chunk = c.chunk if hasattr(c, "chunk") else c
            source_documents.setdefault(chunk.source_file, set()).add(chunk.page_num)

        citations_verified = self._verify_citations(citations, chunks)

        return {
            "answer": answer,
            "chunks": chunks,
            "source_documents": {k: sorted(v) for k, v in source_documents.items()},
            "citations": citations_verified,
            "insufficient_data": insufficient,
        }

    def query_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        document_id: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Streaming version of ``query``.

        Yields dicts:
            ``{"type": "token", "content": "..."}`` for each LLM token.
            ``{"type": "done", "answer": ..., "source_documents": ..., "chunks": ...}``
            when generation is complete.
        """
        chunks = self._retrieve(question, document_id=document_id)

        insufficient = (
            len(chunks) == 0 or max(c.score for c in chunks) < settings.retrieval.score_threshold
        )

        if insufficient:
            yield {
                "type": "done",
                "answer": "Insufficient data to verify from the uploaded documents.",
                "source_documents": {},
                "chunks": [],
                "citations": [],
                "insufficient_data": True,
            }
            return

        prompt = self._build_prompt(question, chunks, history=history)

        # Build source_documents from retrieved chunks
        source_documents: dict[str, set[int]] = {}
        for c in chunks:
            chunk = c.chunk if hasattr(c, "chunk") else c
            source_documents.setdefault(chunk.source_file, set()).add(chunk.page_num)
        source_documents = {k: sorted(v) for k, v in source_documents.items()}

        # Stream tokens
        full = ""
        for token in self.llm.invoke_stream(prompt, max_tokens=self.max_tokens):
            full += token
            yield {"type": "token", "content": token}

        # Parse citations from the complete response
        answer, citations = self._parse_citations(full)

        citations_verified = self._verify_citations(citations, chunks)

        yield {
            "type": "done",
            "answer": answer,
            "source_documents": source_documents,
            "chunks": chunks,
            "citations": citations_verified,
            "insufficient_data": insufficient,
        }

    # ── Query intent detection ──────────────────────────────────────────────

    @staticmethod
    def _detect_query_intent(question: str) -> str:
        """Classify query intent for adaptive retrieval.

        Returns one of: ``"identity"``, ``"list"``, ``"default"``.
        """
        q = question.lower().strip()

        identity_tokens = [
            "name",
            "candidate",
            "owner",
            "email",
            "phone",
            "contact",
            "called",
            "who is",
            "who am i",
            "full name",
        ]
        if any(t in q for t in identity_tokens):
            return "identity"

        # Use noun-based tokens to avoid false positives on generic questions
        list_tokens = [
            "list",
            "projects",
            "skills",
            "experience",
            "education",
            "certifications",
            "all the",
            "each",
            "every",
            "enumerate",
        ]
        if any(t in q for t in list_tokens):
            return "list"

        return "default"

    # ── Query expansion ─────────────────────────────────────────────────────

    @staticmethod
    def _expand_query(question: str, intent: str) -> str:
        """Expand query with intent-aware keywords to improve retrieval.

        Does not alter the original question — only appends generic terms that
        help the embedding model focus on the relevant section of a document.
        """
        if intent == "identity":
            return f"{question} name email contact header personal information"
        if intent == "list":
            return f"{question} list details summary"
        return question

    # ── Identity-chunk injection ────────────────────────────────────────────

    def _ensure_identity_in_context(
        self, results: list[SearchResult], document_id: str | None
    ) -> list[SearchResult]:
        """If no identity info (name / email / phone) is present in results,
        inject a header chunk from the top document."""
        if not results:
            return results

        # Heuristic: if any result contains an email, header is already present
        for r in results:
            if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", r.chunk.text[:200]):
                return results

        # Find the header chunk of the top document
        top_doc = results[0].chunk.source_file
        all_chunks = self.store.get_all_chunks()
        doc_chunks = [c for c in all_chunks if c.source_file == top_doc]
        if not doc_chunks:
            return results

        # Prefer a chunk containing an email address; fall back to earliest page
        header = next(
            (c for c in doc_chunks if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", c.text)),
            None,
        )
        if header is None:
            header = min(doc_chunks, key=lambda c: (c.page_num, c.chunk_id))

        if any(r.chunk.text == header.text for r in results):
            return results

        out = results[:-1]
        out.insert(0, SearchResult(chunk=header, score=1.0))
        return out

    def _retrieve(self, question: str, document_id: str | None = None) -> list[Any]:
        """Retrieve relevant chunks — dense only, hybrid (BM25+RRF), or hybrid+rerank.

        Args:
            question: The user query
            document_id: Optional — restrict retrieval to a single document
        """
        intent = self._detect_query_intent(question)
        q = self._expand_query(question, intent)

        # Determine how many candidates to fetch before reranking / fusion
        if settings.retrieval.enable_hybrid or settings.retrieval.enable_reranker:
            candidate_k = max(settings.retrieval.rerank_top_k, self.top_k * 3)
        else:
            candidate_k = self.top_k

        # Step 1: Dense retrieval (always)
        dense_results = self.store.similarity_search(
            q,
            k=candidate_k,
            score_threshold=settings.retrieval.score_threshold,
            document_id=document_id,
        )

        # Step 2: Hybrid — merge with BM25 via RRF
        if settings.retrieval.enable_hybrid:
            if not self._bm25_built:
                self.rebuild_index()

            if self._bm25 is not None and self._bm25.is_ready:
                bm25_results = self._bm25.search(q, k=candidate_k, document_id=document_id)
                merged = rff_merge(
                    dense_results,
                    bm25_results,
                    k=candidate_k,
                    alpha=settings.retrieval.hybrid_alpha,
                )
            else:
                merged = dense_results
        else:
            merged = dense_results

        # Step 3: Rerank with cross-encoder
        if settings.retrieval.enable_reranker and merged:
            if self._reranker is None:
                self._reranker = Reranker()
            merged = self._reranker.rerank(q, merged, top_k=self.top_k)

        # Step 4: Trim to final top_k
        results = merged[: self.top_k]

        # Step 5: Intent-aware post-processing
        if intent == "identity":
            results = self._ensure_identity_in_context(results, document_id)

        return results

    def _build_prompt(
        self, question: str, chunks: list[Any], history: list[dict] | None = None
    ) -> str:
        """Build the prompt with context + question + optional history."""
        system_prompt = get_system_prompt(question, custom_template=self.prompt_template)
        context = format_chunks_for_context(chunks)
        history_str = format_history(history or [], max_pairs=settings.rag.max_history_pairs)

        parts = [system_prompt]
        if history_str:
            parts.append(history_str)
        parts.append(f"Context:\n{context}")
        parts.append(f"Question: {question}")
        parts.append("Answer (with citations):")
        return "\n\n".join(parts)

    # Regex that captures ANY bracketed content containing "page" or "صفحة" + numbers.
    # Handles both clean and noisy formats:
    #   [page X]  — simple English
    #   [صفحة X]  — simple Arabic
    #   [filename.pdf, page X]  — noisy with filename
    #   [page X from filename.pdf]  — swapped order
    #   [page 2, 3]  — multiple pages
    #   [Context N [filename.pdf, page X]]  — nested (inner bracket eaten by non-greedy)
    _CITATION_BRACKET_RE = re.compile(
        r"\[([^\]]*?(?:page|صفحة)\s+\d+(?:\s*,\s*\d+)*[^\]]*?)\]",
        re.IGNORECASE,
    )

    # Inner page-number extractor within bracket content
    _PAGE_NUM_RE = re.compile(r"(?:page|صفحة)\s+(\d+(?:\s*,\s*\d+)*)", re.IGNORECASE)

    def _parse_citations(self, response: str) -> tuple[str, list[dict]]:
        """
        Extract page citations from LLM response.

        Supports all formats:
          - ``[page X]``
          - ``[صفحة X]``
          - ``[filename.pdf, page X]``
          - ``[page X from filename.pdf]``
          - ``[page 2, 3]``
          - ``[Context N [filename.pdf, page X]]``

        Returns:
            Tuple of (sanitized_answer, list_of_citation_dicts),
            where the answer has been normalized to clean ``[page X]`` / ``[صفحة X]``.
        """
        citations = []

        for match in self._CITATION_BRACKET_RE.finditer(response):
            content = match.group(1)
            page_match = self._PAGE_NUM_RE.search(content)
            if not page_match:
                continue

            page_numbers = [int(p.strip()) for p in page_match.group(1).split(",")]

            filename = self._extract_filename_from_bracket(content, page_match)

            citations.append(
                {
                    "pages": page_numbers,
                    "text_before": response[max(0, match.start() - 40) : match.start()],
                    "filename": filename,
                }
            )

        clean_answer = self._CITATION_BRACKET_RE.sub(
            self._normalize_citation_bracket, response
        ).strip()

        return clean_answer, citations

    @staticmethod
    def _extract_filename_from_bracket(content: str, page_match: re.Match) -> str | None:
        """Try to extract a filename from bracket content (before or after page keyword)."""
        before = content[: page_match.start()].strip()
        before = re.sub(r"^Context\s+\d+\s*", "", before, flags=re.IGNORECASE).strip()
        before = before.lstrip("[").strip()
        m = re.search(r"\b([\w\-\.\(\)]+\.pdf)", before)
        if m:
            return m.group(1).strip()
        after = content[page_match.end() :].strip()
        m = re.search(r"\b([\w\-\.\(\)]+\.pdf)", after)
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def _normalize_citation_bracket(match: re.Match) -> str:
        """Replace a noisy citation bracket with clean ``[page X]`` / ``[صفحة X]``."""
        inner = match.group(1)
        p = re.search(r"(?:page|صفحة)\s+(\d+(?:\s*,\s*\d+)*)", inner, re.IGNORECASE)
        if p:
            pages = p.group(1)
            arabic = "صفحة" in inner
            return f"[{'صفحة' if arabic else 'page'} {pages}]"
        return match.group(0)

    def _verify_citations(self, citations: list[dict], chunks: list[Any]) -> list[dict]:
        """Check each citation against retrieved chunks.

        A citation is verified if every page number it references exists
        in at least one of the retrieved chunks.
        """
        valid_pages: set[int] = set()
        for c in chunks:
            chunk = c.chunk if hasattr(c, "chunk") else c
            valid_pages.add(chunk.page_num)

        verified = []
        for cit in citations:
            all_found = all(p in valid_pages for p in cit["pages"])
            verified.append({**cit, "verified": all_found})

        return verified
