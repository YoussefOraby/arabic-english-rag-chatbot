# Project Roadmap

## Phases Overview

| Phase | Name | Status | Key Deliverable |
|-------|------|--------|-----------------|
| 0 | Scaffolding | ✅ Complete | Repo structure, CI, config, sample PDFs |
| 1 | Environment Setup | 🔜 Next | Verified Python, Ollama/Gemini, imports |
| 2 | PDF Extraction & Chunking | ⏳ Pending | extract_pages(), chunk_pages() work on test PDFs |
| 3 | Embeddings & Vector Store | ⏳ Pending | ChromaDB populated, similarity search works |
| 4 | Streamlit UI + RAG Chain | ⏳ Pending | Chat interface, Q&A with citations |
| 5 | FastAPI Backend (v2) | ⏳ Pending | /chat endpoint, API docs |
| 6 | Evaluation & Metrics | ⏳ Pending | docs/evaluation.md with retrieval/answer quality |
| 7 | Deploy & Polish | ⏳ Pending | Live URL, README complete, demo video |

## Phase Details

### Phase 0 — Scaffolding
- [x] Folder structure
- [x] pyproject.toml with dependencies
- [x] Config system (YAML + env vars)
- [x] Docker support (optional)
- [x] GitHub Actions CI
- [x] Makefile with common commands
- [x] Sample PDFs (arXiv + Arabic)
- [x] README skeleton

### Phase 1 — Environment Setup
- [ ] Check Python 3.11+
- [ ] Install dependencies
- [ ] Verify imports
- [ ] Pull Ollama model or set Gemini API key
- [ ] Hello-world LLM call works

### Phase 2 — PDF Extraction & Chunking
- [ ] Implement extract_pages() with PyMuPDF
- [ ] Implement chunk_pages() with LangChain
- [ ] Test on all sample PDFs (English, Arabic)
- [ ] Print chunks to verify correctness

### Phase 3 — Embeddings & Vector Store
- [ ] Implement ChromaStore.__init__()
- [ ] Implement add_chunks() with sentence-transformers
- [ ] Implement similarity_search()
- [ ] Test retrieval with Arabic and English queries

### Phase 4 — Streamlit UI + RAG Chain
- [ ] Implement build_rag_chain()
- [ ] Implement prompts.py (Ar/En templates)
- [ ] Build Streamlit chat interface
- [ ] Wire: query → retrieve → generate → display with citations

### Phase 5 — FastAPI Backend (v2)
- [ ] FastAPI app with /chat endpoint
- [ ] POST /chat accepts {query, history}
- [ ] Returns {answer, citations, sources}
- [ ] Swagger docs at /docs

### Phase 6 — Evaluation & Metrics
- [ ] Define test queries (Ar + En)
- [ ] Measure retrieval@k, answer relevance
- [ ] Document in docs/evaluation.md

### Phase 7 — Deploy & Polish
- [ ] Deploy to Streamlit Community Cloud
- [ ] Finalize README with screenshots
- [ ] Record demo video
- [ ] GitHub repo cleanup

## Dependencies Between Phases

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
                                            ↓
                                       Phase 5 (optional)
                                            ↓
                                       Phase 6 → Phase 7
```

## Success Gates

- **Phase 0**: `make test` passes, 2 PDFs in `data/raw/`
- **Phase 1**: `python -c "from src.llm.ollama_llm import OllamaLLM; print(OllamaLLM().invoke('hi'))"` works
- **Phase 2**: All test PDFs produce correct chunks (Ar + En)
- **Phase 3**: Top-3 retrieval returns relevant chunks for test queries
- **Phase 4**: Chat answers with [page X] citations
- **Phase 7**: Live URL, `make test` passes on CI
