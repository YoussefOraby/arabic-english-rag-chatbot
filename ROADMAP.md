# Project Roadmap

## V1 — Completed (Current Release)

The following features are fully implemented and tested:

| Feature | Status | Details |
|---------|--------|---------|
| PDF upload | Done | Drag-and-drop PDFs via Streamlit UI or REST API |
| Text extraction | Done | PyMuPDF with Arabic RTL support |
| OCR fallback | Done | Tesseract fallback for scanned PDFs |
| Semantic chunking | Done | Sentence-boundary + similarity merging |
| Embeddings | Done | sentence-transformers multilingual (384-dim) |
| ChromaDB vector store | Done | Local persistent storage |
| Semantic retrieval | Done | Cosine similarity search |
| Hybrid retrieval | Done | BM25 + dense score fusion (RFF) |
| FastAPI REST API | Done | /query, /upload, /documents, /reindex, /health, /stats |
| Streamlit UI | Done | Chat interface with RTL Arabic support |
| Document management | Done | List, delete, reindex documents |
| Citations | Done | [page X] and [صفحة X] citation format |
| Language-aware prompts | Done | Separate English/Arabic system prompts |
| Unsupported question handling | Done | Refuses to answer when context is insufficient |
| Evaluation framework | Done | 22-question golden dataset with 6 metrics |
| Docker support | Done | Dockerfile + docker-compose (app + Ollama) |
| Tests | Done | 163+ tests across all modules |
| GitHub Actions CI | Done | Automated lint, format check, test runner |

## Current Limitations

- **LLM quality**: Small local model (llama3.2:3b) is fast but can miss details. Upgrade to llama3.3:70b for production.
- **Reranker**: Current cross-encoder is English-only; degrades Arabic retrieval. Needs multilingual reranker (e.g. BAAI/bge-reranker-v2-m3).
- **Pass rate**: 45–50% on evaluation dataset. Remaining failures are retrieval gaps (expected pages not in top-k) and LLM answer quality.
- **No auth**: Single-user, no login, no user isolation.
- **No persistent memory**: Chat history is per-session in the UI; API accepts history per-request only.

## Future Work / Phase B

| Feature | Priority | Notes |
|---------|----------|-------|
| Persistent conversation memory | Medium | Store chat turns in a database for cross-session continuity |
| Multilingual reranker | Medium | Replace English-only cross-encoder with BAAI/bge-reranker-v2-m3 |
| Authentication & user isolation | Low | Multi-user support with login |
| Qdrant option | Low | Alternative vector store for larger-scale deployments |
| Observability / Langfuse | Low | Tracing and monitoring for RAG pipelines |
| Response caching | Low | Cache frequent queries for lower latency |
| Production deployment | Low | Cloud deployment (Streamlit Community Cloud or VPS) |

## Success Gates

- **V1**: All 163+ tests pass, 22 evaluation questions runnable, CI green on every push
