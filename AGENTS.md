# Evaluation & RAG Quality Framework

## Quick Reference

```bash
# Run full evaluation against running API
python evaluation/evaluate.py --run-name "my-run"

# Run a quick smoke test (use --verbose to see failures)
python evaluation/evaluate.py --run-name "quick" --verbose

# Run all unit tests
python -m pytest tests/test_evaluate.py tests/test_golden_resume.py -v

# Debug a specific question
python scripts/debug_query.py --question "What is RAG?"

# Debug the full RAG pipeline
python evaluation/debug/debug_chatbot.py
```

## Architecture

```
evaluation/
  golden_questions.json   # 39 curated questions (RAG paper + synthetic resume + unsupported)
  evaluate.py             # Runs eval against the live API, categorizes failures, writes reports
  runs/                   # Per-run results (results.json, report.md, summary.csv)
  debug/
    debug_chatbot.py      # Interactive debug dashboard for single questions
    plan.md               # RAG debugging guide

tests/
  test_golden_resume.py   # 37 tests: PDF parseability, eval engine correctness, answer logic
  test_evaluate.py        # 43 tests: keyword matching, Arabic normalization, refusal detection

fixtures/
  golden_resume.pdf       # Synthetic resume (Ahmed Hassan) for automated testing
  sample_arxiv.pdf        # English RAG survey paper (source: arxiv 2312.10997)
  sample_ar.pdf           # Arabic sample PDF
```

## Evaluation Methodology

### Test Suite (37 automated tests in test_golden_resume.py)

| Category | Tests | What It Verifies |
|----------|-------|------------------|
| PDF parseability | 2 | golden_resume.pdf text extraction via fitz |
| JSON validity | 6 | Golden questions schema, unique IDs, required fields |
| evaluate_answer() | 15 | FALSE_NEGATIVE, FALSE_POSITIVE, MISSING_KEYWORD, FORBIDDEN_KEYWORD, WRONG_CITATION, Arabic/mixed support, empty answers |
| is_insufficient_data() | 7 | Refusal phrase detection, negative cases |
| extract_citations() | 7 | Bracketed page citation parsing |

**Run:** `python -m pytest tests/test_golden_resume.py -v`

### Golden Questions Dataset (golden_questions.json)

| Group | Count | Description |
|-------|-------|-------------|
| research_paper | 16 | English, Arabic, mixed questions about the RAG survey paper (2312.10997) |
| synthetic_resume | 12 | Identity, list, mixed questions about the golden resume (Ahmed Hassan) |
| unsupported | 11 | Questions NOT answerable from any uploaded document |

### Failure Categories

| Category | Meaning |
|----------|---------|
| `FALSE_NEGATIVE` | System said "insufficient data" but should have answered |
| `FALSE_POSITIVE` | System answered a question that has no support in documents |
| `MISSING_KEYWORD` | Required keywords not present in answer |
| `FORBIDDEN_KEYWORD` | Answer contains terms it shouldn't (e.g. listing projects as achievements) |
| `WRONG_CITATION` | Cited pages don't match expected pages |

### Running an Evaluation

1. Start the API: `python -m uvicorn src.api:app --host 0.0.0.0 --port 8000`
2. Ensure documents are ingested (upload golden_resume.pdf and sample_arxiv.pdf via API)
3. Run: `python evaluation/evaluate.py --run-name "baseline-v2"`
4. View results: `evaluation/runs/baseline-v2/report.md`

### Evaluating the Synthetic Resume

Before running resume questions, ingest the golden resume PDF:

```python
import requests
requests.post("http://localhost:8000/upload", files={"file": open("tests/fixtures/golden_resume.pdf", "rb")})
```

### Baseline Results (2026-06-23)

| Metric | Value |
|--------|-------|
| **Overall pass rate** | **35.9%** (14/39) |
| Research paper (English) | 31.2% (5/16) |
| Synthetic resume | 0% (0/12) — golden resume NOT ingested |
| Unsupported | 81.8% (9/11) |

## Debugging Guide

### Common Failure Patterns

1. **FALSE_NEGATIVE on in-document questions** → Check document is ingested, check chunk retrieval
2. **FALSE_NEGATIVE on English paper questions** → Often a retrieval issue; try `debug_query.py` to inspect chunks
3. **FALSE_POSITIVE** → Insufficient data guard not triggering; check `_is_data_sufficient` logic
4. **MISSING_KEYWORD** → Answer is correct but doesn't contain specific expected terms
5. **WRONG_CITATION** → Correct content but wrong page reference

### Debug Scripts

- `scripts/debug_query.py "What is RAG?"` — Shows retrieved chunks, scores, and final answer
- `evaluation/debug/debug_chatbot.py` — Interactive dashboard with expansion panels
- `scripts/verify_search.py` — Verify search results for specific queries

### Key Source Files

- `src/rag/chain.py` — Main RAG chain: retrieval, context building, LLM call
- `src/rag/retriever.py` — Embedding & retrieval logic
- `src/llm/ollama_llm.py` — LLM call formatting and response parsing
- `src/embeddings/store.py` — ChromaDB vector store operations
- `src/api/main.py` — API endpoints (/query, /upload, /health)

## Adding New Golden Questions

1. Add question to `evaluation/golden_questions.json`
2. Ensure JSON schema includes: `id`, `question`, `should_answer`, `required_keywords`
3. For resume questions, add `expected_keywords` (OR matching)
4. For precision checks, add `forbidden_keywords`
5. Run tests: `python -m pytest tests/test_golden_resume.py -v`
6. Run eval: `python evaluation/evaluate.py --run-name "verify-new-questions"`

## Run Comparison

To compare runs, the evaluation engine writes per-run results to `evaluation/runs/{name}/`.
The `latest` symlink points to the most recent run, and delta reports are generated
automatically when a new run detects differences from the previous run.
