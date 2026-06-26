"""RAG chain: query → retrieve → prompt → generate → answer with citations.

Supports both sync (``query``) and streaming (``query_stream``) modes.
"""

import re
from collections.abc import Generator
from typing import Any

from src.config import settings
from src.embeddings.store import ChromaStore
from src.llm.base import BaseLLM
from src.rag.prompts import format_chunks_for_context, format_history, get_system_prompt
from src.retrieval.hybrid import BM25Index, rff_merge
from src.retrieval.reranker import Reranker
from src.utils.helpers import is_arabic_text, normalize_arabic_for_retrieval, extract_arabic_keywords

# Regex patterns for deterministic field extraction
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-\s.]?)?\(?\d{2,4}\)?[-\s.]?\d{3,4}[-\s.]?\d{3,4}")


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

        print(f"DEBUG: chunks found = {len(chunks)}")
        if chunks:
            print(f"DEBUG: scores = {[round(c.score, 4) for c in chunks]}")
            print(f"DEBUG: max score = {max(c.score for c in chunks)}")
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

        # Reorder chunks for short documents (≤10 total chunks)
        chunks = self._reorder_short_document(chunks)

        prompt = self._build_prompt(question, chunks, history=history)

        raw_answer = self.llm.invoke(prompt, max_tokens=self.max_tokens)

        answer, citations = self._parse_citations(raw_answer)

        source_documents: dict[str, set[int]] = {}
        for c in chunks:
            chunk = c.chunk if hasattr(c, "chunk") else c
            source_documents.setdefault(chunk.source_file, set()).add(chunk.page_num)

        citations_verified = self._verify_citations(citations, chunks)

        # Post-process: correct wrong page numbers if all evidence is from one page
        known_pages: set[int] = set()
        for c in chunks:
            chunk = c.chunk if hasattr(c, "chunk") else c
            known_pages.add(chunk.page_num)
        answer, citations_verified = self._correct_citations(
            answer, citations_verified, known_pages, source_documents
        )

        # Deterministic override for contact-style structured fields
        contact_type = self._detect_contact_question(question)
        if contact_type:
            extracted = self._extract_structured_field(chunks, contact_type)
            if extracted:
                if extracted.lower() not in answer.lower():
                    answer = extracted

        # Strip inline citations from the user-facing answer (they are in metadata)
        clean_answer = self._strip_inline_citations(answer)

        return {
            "answer": clean_answer,
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

        # Reorder chunks for short documents (≤10 total chunks)
        chunks = self._reorder_short_document(chunks)

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

        # Post-process: correct wrong page numbers if all evidence is from one page
        known_pages: set[int] = set()
        for c in chunks:
            chunk = c.chunk if hasattr(c, "chunk") else c
            known_pages.add(chunk.page_num)
        answer, citations_verified = self._correct_citations(
            answer, citations_verified, known_pages, source_documents
        )

        # Deterministic override for contact-style structured fields
        contact_type = self._detect_contact_question(question)
        if contact_type:
            extracted = self._extract_structured_field(chunks, contact_type)
            if extracted:
                if extracted.lower() not in answer.lower():
                    answer = extracted

        # Strip inline citations from the user-facing answer (they are in metadata)
        clean_answer = self._strip_inline_citations(answer)

        yield {
            "type": "done",
            "answer": clean_answer,
            "source_documents": source_documents,
            "chunks": chunks,
            "citations": citations_verified,
            "insufficient_data": insufficient,
        }

    def _normalize_query(self, question: str) -> str:
        """Normalize query for retrieval — Arabic normalization if Arabic, otherwise identity."""
        if is_arabic_text(question):
            return normalize_arabic_for_retrieval(question)
        return question

    def _retrieve(self, question: str, document_id: str | None = None) -> list[Any]:
        """Retrieve relevant chunks — dense only, hybrid (BM25+RRF), or hybrid+rerank.

        Args:
            question: The user query
            document_id: Optional — restrict retrieval to a single document
        """

        # Normalize query for Arabic retrieval (strip diacritics/tatweel, unify alef)
        search_query = self._normalize_query(question)

        # Determine how many candidates to fetch before reranking / fusion
        if settings.retrieval.enable_hybrid or settings.retrieval.enable_reranker:
            candidate_k = max(settings.retrieval.rerank_top_k, self.top_k * 3)
        else:
            candidate_k = self.top_k

        # Step 1: Dense retrieval (always)
        dense_results = self.store.similarity_search(
            search_query,
            k=candidate_k,
            score_threshold=settings.retrieval.score_threshold,
            document_id=document_id,
        )

        # Step 2: Hybrid — merge with BM25 via RRF
        if settings.retrieval.enable_hybrid:
            if not self._bm25_built:
                self.rebuild_index()

            if self._bm25 is not None and self._bm25.is_ready:
                bm25_results = self._bm25.search(search_query, k=candidate_k, document_id=document_id)
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
            merged = self._reranker.rerank(search_query, merged, top_k=self.top_k)

        # Step 4: Trim to final top_k
        trimmed = merged[: self.top_k]

        # Step 5: Keyword-match reordering for Arabic questions
        if is_arabic_text(question):
            trimmed = self._reorder_by_keywords(question, trimmed)

        return trimmed

    def _reorder_short_document(self, chunks: list[Any]) -> list[Any]:
        """Reorder chunks by document position for short documents.

        When a document has ≤10 total chunks in the store, retrieval
        by score can place a semantically similar but positionally later
        chunk before the header chunk (e.g. projects before contact info).
        Reordering by document order (page, then insertion order) puts
        header / contact info before less relevant sections, which helps
        the LLM answer contact-style questions correctly.
        """
        if not chunks:
            return chunks

        # Determine which source files are involved
        source_files: set[str] = set()
        for r in chunks:
            chunk = r.chunk if hasattr(r, "chunk") else r
            source_files.add(chunk.source_file)

        # Check total chunks per source_file
        all_in_store = self.store.get_all_chunks()
        doc_chunk_count: dict[str, int] = {}
        for c in all_in_store:
            doc_chunk_count[c.source_file] = doc_chunk_count.get(c.source_file, 0) + 1

        should_reorder = any(doc_chunk_count.get(sf, 0) <= 10 for sf in source_files)
        if not should_reorder:
            return chunks

        # Build a position map from the store's insertion order (document order)
        position: dict[str, int] = {}
        for pos, c in enumerate(all_in_store):
            position[c.chunk_id] = pos

        def sort_key(r: Any) -> tuple:
            chunk = r.chunk if hasattr(r, "chunk") else r
            return (chunk.page_num, position.get(chunk.chunk_id, 999999))

        return sorted(chunks, key=sort_key)

    @staticmethod
    def _reorder_by_keywords(question: str, chunks: list[Any]) -> list[Any]:
        """Reorder retrieved chunks so those containing question keywords appear first.

        For Arabic questions, extracts significant keywords from the question
        and prioritizes chunks that contain them. Within keyword-matched chunks,
        preserves original rank order.
        """
        if not chunks:
            return chunks

        keywords = extract_arabic_keywords(question)
        if not keywords:
            return chunks

        def match_count(r: Any) -> int:
            chunk = r.chunk if hasattr(r, "chunk") else r
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            text_norm = normalize_arabic_for_retrieval(text)
            return sum(1 for kw in keywords if kw in text_norm)

        # Separate into matched and unmatched, preserving relative order within each group
        matched = [r for r in chunks if match_count(r) > 0]
        unmatched = [r for r in chunks if match_count(r) == 0]

        # Sort matched by (descending match_count, original score descending)
        matched.sort(key=lambda r: (match_count(r), r.score if hasattr(r, "score") else 0), reverse=True)

        return matched + unmatched

    @staticmethod
    def _strip_inline_citations(text: str) -> str:
        """Remove inline citation brackets like [page X], [صفحة X] from answer text.

        The citations are already captured in the ``citations`` metadata field.
        This prevents invented or noisy page numbers from reaching the user.
        """
        stripped = re.sub(
            r"\[(?:page|صفحة)\s*:?\s*\d+(?:\s*,\s*\d+)*[^\]]*?\]",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # Also strip bare [Source N] brackets that sometimes leak through
        stripped = re.sub(
            r"\[\s*Source\s+\d+(?:\s*\|[^\]]*)?\]",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        # Clean up double spaces, leading/trailing whitespace, and empty brackets
        stripped = re.sub(r"\s{2,}", " ", stripped).strip()
        stripped = re.sub(r"\[\s*\]", "", stripped).strip()
        return stripped

    @staticmethod
    def _detect_contact_question(question: str) -> str | None:
        """Detect if this is an email or phone contact-style question.

        Returns ``"email"``, ``"phone"``, or ``None``.
        """
        q = question.lower()
        if re.search(r"\b(email|e-mail|mail|البريد|الإلكتروني|إيميل)\b", q, re.IGNORECASE):
            return "email"
        if re.search(r"\b(phone|telephone|mobile|cell|رقم|تليفون|هاتف|موبايل)\b", q, re.IGNORECASE):
            return "phone"
        return None

    @staticmethod
    def _extract_structured_field(chunks: list[Any], field_type: str) -> str | None:
        """Deterministically extract email or phone from retrieved chunk texts.

        This runs over all retrieved chunk texts and returns the first match.
        Used to verify / correct the LLM answer for structured contact fields.
        """
        all_text = ""
        for r in chunks:
            chunk = r.chunk if hasattr(r, "chunk") else r
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            all_text += "\n" + text

        if field_type == "email":
            matches = _EMAIL_RE.findall(all_text)
            return matches[0] if matches else None

        if field_type == "phone":
            matches = _PHONE_RE.findall(all_text)
            if matches:
                # Return the longest match (most complete phone number)
                return max(matches, key=len)
            return None

        return None

    def _build_prompt(
        self, question: str, chunks: list[Any], history: list[dict] | None = None
    ) -> str:
        """Build the prompt with context + question + optional history."""
        system_prompt = get_system_prompt(question, custom_template=self.prompt_template)
        context = format_chunks_for_context(chunks)
        history_str = format_history(history or [], max_pairs=settings.rag.max_history_pairs)

        # Add extra precision instructions for contact-style questions
        contact_type = self._detect_contact_question(question)
        extra_instruction = ""
        if contact_type == "email":
            extra_instruction = (
                "\n\nPrecision instruction: The question asks for an email address. "
                "Look for a string like 'name@domain.com' in the context. "
                "Do NOT return a phone number for an email question."
            )
        elif contact_type == "phone":
            extra_instruction = (
                "\n\nPrecision instruction: The question asks for a phone number. "
                "Look for a numeric phone number in the context. "
                "Do NOT return an email address for a phone question."
            )

        parts = [system_prompt]
        if extra_instruction:
            parts.append(extra_instruction)
        if history_str:
            parts.append(history_str)
        parts.append(f"Context:\n{context}")
        parts.append(f"Question: {question}")
        parts.append("Answer:")
        return "\n\n".join(parts)

    # Regex that captures ANY bracketed content containing "page" or "صفحة" + numbers.
    # Handles both clean and noisy formats:
    #   [page X]  — simple English
    #   [صفحة X]  — simple Arabic
    #   [filename.pdf, page X]  — noisy with filename
    #   [page X from filename.pdf]  — swapped order
    #   [page 2, 3]  — multiple pages
    #   [page: X] or [page:X] — with colon (LLM mimicking header format)
    #   [Context N [filename.pdf, page X]]  — nested (inner bracket eaten by non-greedy)
    _CITATION_BRACKET_RE = re.compile(
        r"\[([^\]]*?(?:page|صفحة)\s*:?\s*\d+(?:\s*,\s*\d+)*[^\]]*?)\]",
        re.IGNORECASE,
    )

    # Inner page-number extractor within bracket content
    _PAGE_NUM_RE = re.compile(r"(?:page|صفحة)\s*:?\s*(\d+(?:\s*,\s*\d+)*)", re.IGNORECASE)

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
        p = re.search(r"(?:page|صفحة)\s*:?\s*(\d+(?:\s*,\s*\d+)*)", inner, re.IGNORECASE)
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

    @staticmethod
    def _correct_citations(
        answer: str,
        citations: list[dict],
        known_pages: set[int],
        source_documents: dict[str, list[int]],
    ) -> tuple[str, list[dict]]:
        """Post-process answer to fix invented page numbers.

        If:
          - A citation references page(s) NOT in ``known_pages`` (LLM hallucinated),
          - AND all retrieved chunks come from a single page,
        then rewrite the invented page number to the known page.

        This is a safety net when the LLM confuses source index with page number.
        """
        if not citations or not known_pages:
            return answer, citations

        # Determine the one true page if all evidence is from a single page
        all_same_page = len(known_pages) == 1
        if not all_same_page:
            return answer, citations

        true_page = next(iter(known_pages))

        fixed_citations = []
        modified = False
        for cit in citations:
            original_pages = cit["pages"]
            invented = [p for p in original_pages if p not in known_pages]
            if invented:
                # Replace invented pages with the one true page
                new_pages = [true_page if p in invented else p for p in original_pages]
                fixed_citations.append({**cit, "pages": new_pages, "verified": True})
                modified = True
            else:
                fixed_citations.append(cit)

        if not modified:
            return answer, citations

        # Rewrite page numbers in the answer text
        def _replace_page(match: re.Match) -> str:
            inner = match.group(1)
            p = re.search(r"(?:page|صفحة)\s*:?\s*(\d+(?:\s*,\s*\d+)*)", inner, re.IGNORECASE)
            if not p:
                return match.group(0)
            old_pages = [int(x.strip()) for x in p.group(1).split(",")]
            if any(pn not in known_pages for pn in old_pages):
                arabic = "صفحة" in inner
                return f"[{'صفحة' if arabic else 'page'} {true_page}]"
            return match.group(0)

        corrected = re.sub(RAGChain._CITATION_BRACKET_RE, _replace_page, answer)

        return corrected, fixed_citations
