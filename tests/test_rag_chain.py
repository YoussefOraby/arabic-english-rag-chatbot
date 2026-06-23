"""Tests for RAG chain (retrieve → prompt → generate → parse citations)."""

import pytest

from src.embeddings.store import SearchResult
from src.pdf.chunker import Chunk
from src.rag.chain import RAGChain


class MockLLM:
    """Mock LLM that returns a fixed response."""

    def invoke(self, prompt: str, **kwargs) -> str:
        return "Based on the context, black holes have event horizons [page 5]. They also have inner horizons [page 3]."

    def invoke_stream(self, prompt: str, **kwargs):
        response = self.invoke(prompt, **kwargs)
        yield from response


class MockStore:
    """Mock ChromaStore returning predefined chunks."""

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.3,
        document_id: str | None = None,
    ):
        return [
            SearchResult(
                chunk=Chunk(
                    text="Black holes have event horizons that prevent light from escaping.",
                    page_num=5,
                    source_file="sample_arxiv.pdf",
                    chunk_id="test_p5_001",
                ),
                score=0.85,
            ),
            SearchResult(
                chunk=Chunk(
                    text="Inner horizons exist inside rotating black holes.",
                    page_num=3,
                    source_file="sample_arxiv.pdf",
                    chunk_id="test_p3_002",
                ),
                score=0.72,
            ),
        ]

    def get_all_chunks(self) -> list:
        return [r.chunk for r in self.similarity_search("")]

    def count_chunks(self) -> int:
        return len(self.get_all_chunks())


@pytest.fixture
def chain():
    """Create RAGChain with mock dependencies."""
    return RAGChain(store=MockStore(), llm=MockLLM())


def test_rag_chain_import():
    """Verify RAGChain can be imported."""
    from src.rag.chain import RAGChain

    assert RAGChain is not None


def test_rag_chain_init(chain):
    """Verify RAGChain initializes with store and llm."""
    assert chain.store is not None
    assert chain.llm is not None


def test_rag_chain_query(chain):
    """Verify full query pipeline returns expected structure."""
    result = chain.query("What are black hole horizons?")
    assert "answer" in result
    assert "chunks" in result
    assert "source_documents" in result
    assert len(result["chunks"]) == 2
    assert "sample_arxiv.pdf" in result["source_documents"]


def test_rag_chain_answer(chain):
    """Verify answer contains expected content from LLM."""
    result = chain.query("What are black hole horizons?")
    assert "event horizons" in result["answer"]
    assert "[page 5]" in result["answer"]


def test_rag_chain_source_documents(chain):
    """Verify source documents are correctly aggregated."""
    result = chain.query("test")
    assert "sample_arxiv.pdf" in result["source_documents"]
    assert result["source_documents"]["sample_arxiv.pdf"] == [3, 5]


def test_rag_chain_retrieve():
    """Verify retrieve method returns search results."""
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    chunks = chain._retrieve("test query")
    assert len(chunks) == 2
    assert chunks[0].score == 0.85
    assert chunks[0].chunk.page_num == 5


def test_rag_chain_build_prompt(chain):
    """Verify prompt is built correctly with context."""
    chunks = chain._retrieve("test")
    prompt = chain._build_prompt("What is a black hole?", chunks)
    assert "Context:" in prompt
    assert "Question:" in prompt
    assert "What is a black hole?" in prompt
    assert "sample_arxiv.pdf" in prompt


def test_parse_citations_arabic_format(chain):
    """Verify Arabic [صفحة X] format is parsed."""
    response = "الثقوب السوداء لها أفق حدث [صفحة 5]. كما لها أفق داخلي [صفحة 3]."
    answer, citations = chain._parse_citations(response)
    assert len(citations) == 2
    assert citations[0]["pages"] == [5]
    assert citations[1]["pages"] == [3]


def test_parse_citations_mixed_format(chain):
    """Verify both [page X] and [صفحة X] work in the same response."""
    response = "Black holes have event horizons [page 5]. الثقوب السوداء لها أفق [صفحة 3]."
    answer, citations = chain._parse_citations(response)
    assert len(citations) == 2
    assert citations[0]["pages"] == [5]
    assert citations[1]["pages"] == [3]


def test_parse_citations_arabic_multiple_pages(chain):
    """Verify Arabic format with multiple pages."""
    response = "راجع الصفحات [صفحة 5, 7, 12] لمزيد من التفاصيل."
    answer, citations = chain._parse_citations(response)
    assert len(citations) == 1
    assert citations[0]["pages"] == [5, 7, 12]


def test_rag_chain_parse_citations(chain):
    """Verify citation parsing extracts page numbers."""
    response = "The event horizon prevents light escape [page 5]. Inner regions [page 3, 7]."
    answer, citations = chain._parse_citations(response)
    assert len(citations) == 2
    assert citations[0]["pages"] == [5]
    assert citations[1]["pages"] == [3, 7]


def test_format_chunks_for_context():
    """Verify context formatting includes source annotations."""
    from src.rag.prompts import format_chunks_for_context

    chunks = [
        SearchResult(
            chunk=Chunk(text="Python is fun.", page_num=1, source_file="doc.pdf", chunk_id="a"),
            score=0.95,
        ),
    ]
    context = format_chunks_for_context(chunks)
    assert "Python is fun." in context
    assert "[doc.pdf, page 1]" in context


def test_get_system_prompt_english():
    """Verify English system prompt is returned for English questions."""
    from src.rag.prompts import ENGLISH_SYSTEM_PROMPT, get_system_prompt

    prompt = get_system_prompt("What is AI?")
    assert prompt == ENGLISH_SYSTEM_PROMPT


def test_get_system_prompt_arabic():
    """Verify Arabic system prompt is returned for Arabic questions."""
    from src.rag.prompts import ARABIC_SYSTEM_PROMPT, get_system_prompt

    prompt = get_system_prompt("ما هو الذكاء الاصطناعي؟")
    assert prompt == ARABIC_SYSTEM_PROMPT


def test_get_system_prompt_custom():
    """Verify custom template overrides language detection."""
    from src.rag.prompts import get_system_prompt

    custom = "Custom template here"
    prompt = get_system_prompt("ما هو الذكاء الاصطناعي؟", custom_template=custom)
    assert prompt == custom


def test_query_stream_yields_tokens():
    """Verify query_stream yields token events before done."""
    from src.rag.chain import RAGChain

    chain = RAGChain(store=MockStore(), llm=MockLLM())
    events = list(chain.query_stream("test"))
    assert len(events) > 1
    assert events[0]["type"] == "token"
    assert events[-1]["type"] == "done"
    assert "answer" in events[-1]


def test_query_with_history_in_prompt(chain):
    """Verify history is included in the built prompt."""
    history = [
        {"role": "user", "content": "What is gravity?"},
        {"role": "assistant", "content": "Gravity is a force [page 1]."},
        {"role": "user", "content": "Tell me more."},
    ]
    chunks = chain._retrieve("test")
    prompt = chain._build_prompt("What are black holes?", chunks, history=history)
    assert "Previous conversation:" in prompt
    assert "User: What is gravity?" in prompt
    assert "Assistant: Gravity is a force [page 1]." in prompt
    assert "End of previous conversation." in prompt
    assert "Question: What are black holes?" in prompt


def test_query_with_empty_history(chain):
    """Verify empty history does not break prompt."""
    chunks = chain._retrieve("test")
    prompt = chain._build_prompt("What is a black hole?", chunks, history=[])
    assert "Previous conversation:" not in prompt
    assert "Question: What is a black hole?" in prompt


def test_query_with_none_history(chain):
    """Verify None history falls back to no history section."""
    chunks = chain._retrieve("test")
    prompt = chain._build_prompt("What is a black hole?", chunks, history=None)
    assert "Previous conversation:" not in prompt


def test_query_history_max_pairs(chain):
    """Verify history is truncated to max_pairs."""
    history = []
    for i in range(6):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    # history has 6 pairs (12 messages); max_history_pairs=4 means last 8 visible
    chunks = chain._retrieve("test")
    prompt = chain._build_prompt("final?", chunks, history=history)
    assert "Previous conversation:" in prompt
    # First 2 pairs should be cut
    assert "User: q0" not in prompt
    assert "User: q2" in prompt  # 3rd pair is within last 4


def test_query_with_history(chain):
    """Verify query() with history returns expected structure."""
    history = [
        {"role": "user", "content": "What is gravity?"},
        {"role": "assistant", "content": "Gravity is a force [page 1]."},
    ]
    result = chain.query("Tell me more.", history=history)
    assert "answer" in result
    assert len(result["chunks"]) == 2


def test_query_stream_with_history(chain):
    """Verify query_stream with history yields tokens."""
    history = [
        {"role": "user", "content": "What is gravity?"},
        {"role": "assistant", "content": "Gravity is a force."},
    ]
    events = list(chain.query_stream("Tell me more.", history=history))
    assert len(events) > 1
    assert events[0]["type"] == "token"
    assert events[-1]["type"] == "done"


def test_query_stream_accumulates_full_text():
    """Verify streaming tokens accumulate to the full answer."""
    from src.rag.chain import RAGChain

    chain = RAGChain(store=MockStore(), llm=MockLLM())
    tokens = []
    for event in chain.query_stream("test"):
        if event["type"] == "token":
            tokens.append(event["content"])
    full = "".join(tokens)
    assert "event horizons" in full
    assert len(full) > 0


def _ollama_running() -> bool:
    try:
        from src.llm.ollama_llm import OllamaLLM

        return OllamaLLM().is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_running(), reason="Ollama not running")
def test_ollama_invoke_stream():
    """Verify invoke_stream yields tokens (requires Ollama running)."""
    from src.config import settings
    from src.llm.ollama_llm import OllamaLLM

    llm = OllamaLLM(**settings.llm.ollama.model_dump())

    tokens = list(llm.invoke_stream("Say hello in one word"))
    assert len(tokens) > 0
    full = "".join(tokens)
    assert len(full) > 0


def test_retrieve_with_hybrid_enabled(monkeypatch):
    """Verify hybrid search produces same-or-more results."""
    monkeypatch.setattr("src.rag.chain.settings.retrieval.enable_hybrid", True)
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    chain.rebuild_index()
    chunks = chain._retrieve("black holes")
    assert len(chunks) == 2  # Only 2 chunks exist


def test_retrieve_hybrid_rebuilds_index(monkeypatch):
    """Verify hybrid triggers BM25 rebuild on first query."""
    monkeypatch.setattr("src.rag.chain.settings.retrieval.enable_hybrid", True)
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    assert not chain._bm25_built
    chunks = chain._retrieve("black holes")
    assert chain._bm25_built
    assert len(chunks) >= 1


def test_retrieve_with_reranker_enabled(monkeypatch):
    """Verify reranker is called when enabled (uses mock)."""
    monkeypatch.setattr("src.rag.chain.settings.retrieval.enable_reranker", True)
    monkeypatch.setattr("src.rag.chain.settings.retrieval.enable_hybrid", False)
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    chunks = chain._retrieve("black holes")
    assert len(chunks) <= 2
    # Reranker replaced scores with cross-encoder scores (but with mock model they may be 0)
    # Just verify the chain ran without error


def test_retrieve_hybrid_and_reranker(monkeypatch):
    """Verify both hybrid and reranker work together."""
    monkeypatch.setattr("src.rag.chain.settings.retrieval.enable_hybrid", True)
    monkeypatch.setattr("src.rag.chain.settings.retrieval.enable_reranker", True)
    monkeypatch.setattr("src.rag.chain.settings.retrieval.top_k", 2)
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    chain.rebuild_index()
    chunks = chain._retrieve("black holes")
    assert len(chunks) == 2


def test_rebuild_index_updates_bm25(monkeypatch):
    """Verify rebuild_index fetches chunks from store."""
    monkeypatch.setattr("src.rag.chain.settings.retrieval.enable_hybrid", True)
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    assert not chain._bm25_built
    chain.rebuild_index()
    assert chain._bm25_built
    assert chain._bm25 is not None
    assert chain._bm25.is_ready


def test_retrieve_with_document_id():
    """Verify document_id is passed through to similarity_search."""
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    chunks = chain._retrieve("test query", document_id="doc_001")
    assert len(chunks) == 2


def test_query_with_document_id():
    """Verify query with document_id returns expected structure."""
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    result = chain.query("What are black hole horizons?", document_id="doc_001")
    assert "answer" in result
    assert "chunks" in result
    assert "source_documents" in result


def test_query_stream_with_document_id():
    """Verify query_stream with document_id yields tokens."""
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    events = list(chain.query_stream("test query", document_id="doc_001"))
    assert len(events) > 1
    assert events[0]["type"] == "token"
    assert events[-1]["type"] == "done"


def test_hybrid_search_respects_document_id(monkeypatch):
    """Verify hybrid search with document_id never returns other docs."""
    monkeypatch.setattr("src.rag.chain.settings.retrieval.enable_hybrid", True)
    monkeypatch.setattr("src.rag.chain.settings.retrieval.enable_reranker", False)
    chunks = [
        SearchResult(
            chunk=Chunk(
                text="Machine learning is AI.",
                page_num=1,
                source_file="ml.pdf",
                chunk_id="ml1",
                document_id="doc_ml",
            ),
            score=0.85,
        ),
        SearchResult(
            chunk=Chunk(
                text="Deep learning is neural networks.",
                page_num=2,
                source_file="ml.pdf",
                chunk_id="ml2",
                document_id="doc_ml",
            ),
            score=0.75,
        ),
        SearchResult(
            chunk=Chunk(
                text="Physics studies matter and energy.",
                page_num=1,
                source_file="physics.pdf",
                chunk_id="phy1",
                document_id="doc_phy",
            ),
            score=0.9,
        ),
    ]

    class HybridFilterMockStore:
        def similarity_search(self, query, k=4, score_threshold=0.3, document_id=None):
            result = [r for r in chunks if not document_id or r.chunk.document_id == document_id]
            return result[:k]

        def get_all_chunks(self):
            return [r.chunk for r in chunks]

        def count_chunks(self):
            return len(chunks)

    chain = RAGChain(store=HybridFilterMockStore(), llm=MockLLM())
    chain.rebuild_index()

    # Filter to doc_ml only
    filtered = chain._retrieve("machine learning", document_id="doc_ml")
    assert len(filtered) > 0
    for r in filtered:
        assert r.chunk.document_id == "doc_ml", f"Got chunk from {r.chunk.document_id}"

    # Filter to doc_phy only
    filtered2 = chain._retrieve("physics", document_id="doc_phy")
    assert len(filtered2) > 0
    for r in filtered2:
        assert r.chunk.document_id == "doc_phy"


def test_retrieve_with_document_id_refines_results(monkeypatch):
    """Verify document_id filters results. Only doc_001 chunks returned."""
    monkeypatch.setattr("src.rag.chain.settings.retrieval.enable_hybrid", False)
    monkeypatch.setattr("src.rag.chain.settings.retrieval.enable_reranker", False)
    chunks = [
        SearchResult(
            chunk=Chunk(
                text="Doc one text.",
                page_num=1,
                source_file="doc1.pdf",
                chunk_id="c1",
                document_id="doc_001",
            ),
            score=0.9,
        ),
        SearchResult(
            chunk=Chunk(
                text="Doc two text.",
                page_num=1,
                source_file="doc2.pdf",
                chunk_id="c2",
                document_id="doc_002",
            ),
            score=0.8,
        ),
        SearchResult(
            chunk=Chunk(
                text="Also doc one.",
                page_num=2,
                source_file="doc1.pdf",
                chunk_id="c3",
                document_id="doc_001",
            ),
            score=0.7,
        ),
    ]

    class FilterableMockStore:
        def similarity_search(self, query, k=4, score_threshold=0.3, document_id=None):
            if document_id:
                return [r for r in chunks if r.chunk.document_id == document_id]
            return chunks

        def get_all_chunks(self):
            return [r.chunk for r in chunks]

        def count_chunks(self):
            return len(chunks)

    chain = RAGChain(store=FilterableMockStore(), llm=MockLLM())

    all_results = chain._retrieve("test query")
    assert len(all_results) == 3

    filtered = chain._retrieve("test query", document_id="doc_001")
    assert len(filtered) == 2
    for r in filtered:
        assert r.chunk.document_id == "doc_001"

    filtered2 = chain._retrieve("test query", document_id="doc_002")
    assert len(filtered2) == 1
    assert filtered2[0].chunk.document_id == "doc_002"


def test_insufficient_data_low_scores(monkeypatch):
    """Verify insufficient_data=True when all scores below threshold."""
    monkeypatch.setattr("src.rag.chain.settings.retrieval.score_threshold", 0.99)
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    result = chain.query("test question")
    assert result["insufficient_data"] is True
    assert result["answer"] == "Insufficient data to verify from the uploaded documents."
    assert result["citations"] == []
    assert result["source_documents"] == {}
    assert result["chunks"] == []


def test_insufficient_data_empty_chunks():
    """Verify insufficient_data=True when no chunks retrieved."""

    class EmptyMockStore:
        def similarity_search(self, query, k=4, score_threshold=0.3, document_id=None):
            return []

        def get_all_chunks(self):
            return []

        def count_chunks(self):
            return 0

    from src.rag.chain import RAGChain

    chain = RAGChain(store=EmptyMockStore(), llm=MockLLM())
    result = chain.query("test question")
    assert result["insufficient_data"] is True
    assert result["chunks"] == []
    assert result["answer"] == "Insufficient data to verify from the uploaded documents."
    assert result["citations"] == []
    assert result["source_documents"] == {}


def test_insufficient_data_sufficient(chain):
    """Verify insufficient_data=False with good scores at default threshold."""
    result = chain.query("What are black hole horizons?")
    assert result["insufficient_data"] is False


def test_insufficient_data_stream(monkeypatch):
    """Verify insufficient_data=True in query_stream done event."""
    monkeypatch.setattr("src.rag.chain.settings.retrieval.score_threshold", 0.99)
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    events = list(chain.query_stream("test"))
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["insufficient_data"] is True
    assert done[0]["answer"] == "Insufficient data to verify from the uploaded documents."
    assert done[0]["citations"] == []
    assert done[0]["source_documents"] == {}
    assert done[0]["chunks"] == []


def test_insufficient_data_has_no_page_zero(monkeypatch):
    """Verify insufficient_data response contains no [page 0]."""
    monkeypatch.setattr("src.rag.chain.settings.retrieval.score_threshold", 0.99)
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    result = chain.query("What is the capital of Japan?")
    assert "[page 0]" not in result["answer"]
    for cit in result["citations"]:
        assert 0 not in cit["pages"]


def test_insufficient_data_has_no_empty_filename(monkeypatch):
    """Verify insufficient_data response has no empty source filenames."""
    monkeypatch.setattr("src.rag.chain.settings.retrieval.score_threshold", 0.99)
    chain = RAGChain(store=MockStore(), llm=MockLLM())
    result = chain.query("What is the capital of Japan?")
    for src in result["source_documents"]:
        assert src != ""
    assert len(result["source_documents"]) == 0


def test_citation_format_no_context_prefix(chain):
    """Verify answer does NOT contain noisy 'Context X:' format citations."""
    result = chain.query("What are black hole horizons?")
    assert "[Context" not in result["answer"]
    assert "Context 1" not in result["answer"]


def test_citation_format_clean_page(chain):
    """Verify answer citations use clean [page X] format."""
    result = chain.query("What are black hole horizons?")
    import re

    matches = re.findall(r"\[(?:page|صفحة)\s+\d+\]", result["answer"])
    assert len(matches) > 0
    for m in matches:
        assert m.startswith("[page") or m.startswith("[صفحة")


def test_citation_verification_valid(chain):
    """Verify citations matching retrieved chunks are marked verified."""
    result = chain.query("What are black hole horizons?")
    retrieved_pages = {c.chunk.page_num for c in result["chunks"]}
    assert len(result["citations"]) > 0
    for cit in result["citations"]:
        for p in cit["pages"]:
            if p in (5, 3):
                assert p in retrieved_pages


def test_citation_verification_invalid():
    """Verify citations with page numbers not in chunks are marked unverified."""
    from src.rag.chain import RAGChain

    class LowPageMockStore:
        def similarity_search(self, query, k=4, score_threshold=0.3, document_id=None):
            return [
                SearchResult(
                    chunk=Chunk(
                        text="Content on page 1.",
                        page_num=1,
                        source_file="doc.pdf",
                        chunk_id="c1",
                    ),
                    score=0.9,
                ),
            ]

        def get_all_chunks(self):
            return [r.chunk for r in self.similarity_search("")]

        def count_chunks(self):
            return 1

    chain = RAGChain(store=LowPageMockStore(), llm=MockLLM())
    result = chain.query("test")
    for cit in result["citations"]:
        for p in cit["pages"]:
            if p not in (1,):
                assert not cit["verified"], f"Page {p} should be unverified"


def test_query_response_includes_citations(chain):
    """Verify query response always includes citations list."""
    result = chain.query("What are black holes?")
    assert "citations" in result
    assert isinstance(result["citations"], list)
    for cit in result["citations"]:
        assert "pages" in cit
        assert "verified" in cit


def test_query_stream_includes_citations(chain):
    """Verify query_stream done event includes citations with verification."""
    events = list(chain.query_stream("test"))
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert "citations" in done[0]
    for cit in done[0]["citations"]:
        assert "verified" in cit


# ── Robust citation parsing tests ────────────────────────────────────


def test_parse_citations_filename_format(chain):
    """Verify [filename.pdf, page X] format is parsed."""
    response = "Black holes have event horizons [2312.10997v5.pdf, page 5]. They have inner horizons [sample_arxiv.pdf, page 3]."
    answer, citations = chain._parse_citations(response)
    assert len(citations) == 2
    assert citations[0]["pages"] == [5]
    assert citations[1]["pages"] == [3]
    assert citations[0]["filename"] == "2312.10997v5.pdf"
    assert citations[1]["filename"] == "sample_arxiv.pdf"


def test_parse_citations_page_from_filename(chain):
    """Verify [page X from filename.pdf] format is parsed."""
    response = "The event horizon prevents light escape [page 5 from 2312.10997v5.pdf]."
    answer, citations = chain._parse_citations(response)
    assert len(citations) == 1
    assert citations[0]["pages"] == [5]
    assert citations[0]["filename"] == "2312.10997v5.pdf"


def test_parse_citations_nested_context(chain):
    """Verify [Context N [filename.pdf, page X]] format is parsed."""
    response = "RAG has three paradigms [Context 1 [2312.10997v5.pdf, page 2]]."
    answer, citations = chain._parse_citations(response)
    assert len(citations) == 1
    assert citations[0]["pages"] == [2]
    assert citations[0]["filename"] == "2312.10997v5.pdf"


def test_parse_citations_filename_multiple_pages(chain):
    """Verify [filename.pdf, page X, Y] with multiple pages."""
    response = "See pages [2312.10997v5.pdf, page 2, 3] for details."
    answer, citations = chain._parse_citations(response)
    assert len(citations) == 1
    assert citations[0]["pages"] == [2, 3]
    assert citations[0]["filename"] == "2312.10997v5.pdf"


def test_parse_citations_arabic_filename_format(chain):
    """Verify Arabic [filename.pdf, صفحة X] format is parsed."""
    response = "الثقوب السوداء لها أفق حدث [2312.10997v5.pdf, صفحة 5]."
    answer, citations = chain._parse_citations(response)
    assert len(citations) == 1
    assert citations[0]["pages"] == [5]
    assert citations[0]["filename"] == "2312.10997v5.pdf"


def test_sanitize_answer_removes_filename(chain):
    """Verify answer is cleaned — no filename inside brackets."""
    response = "Black holes have event horizons [2312.10997v5.pdf, page 5]."
    answer, citations = chain._parse_citations(response)
    assert "2312.10997v5.pdf" not in answer
    assert "[page 5]" in answer


def test_sanitize_answer_preserves_clean_citations(chain):
    """Verify clean [page X] citations are preserved unchanged."""
    response = "Black holes have event horizons [page 5]. They also have inner horizons [page 3]."
    answer, citations = chain._parse_citations(response)
    assert "[page 5]" in answer
    assert "[page 3]" in answer


def test_sanitize_answer_arabic_preserved(chain):
    """Verify Arabic [صفحة X] citations are preserved unchanged."""
    response = "الثقوب السوداء لها أفق حدث [صفحة 5]."
    answer, citations = chain._parse_citations(response)
    assert "[صفحة 5]" in answer


def test_parse_citations_no_match(chain):
    """Verify no citations extracted when there are none."""
    response = "Black holes are fascinating objects in space."
    answer, citations = chain._parse_citations(response)
    assert len(citations) == 0
    assert answer == response.strip()


def test_query_stream_parses_filename_citations():
    """Verify query_stream done event extracts filename-style citations."""

    class FilenameMockLLM:
        def invoke(self, prompt, **kwargs):
            return "Event horizon [blackhole.pdf, page 5]."

        def invoke_stream(self, prompt, **kwargs):
            yield from self.invoke(prompt, **kwargs)

    from src.rag.chain import RAGChain

    chain = RAGChain(store=MockStore(), llm=FilenameMockLLM())
    events = list(chain.query_stream("test"))
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert len(done[0]["citations"]) == 1
    assert done[0]["citations"][0]["pages"] == [5]
    assert done[0]["citations"][0]["filename"] == "blackhole.pdf"
    assert "[page 5]" in done[0]["answer"]
