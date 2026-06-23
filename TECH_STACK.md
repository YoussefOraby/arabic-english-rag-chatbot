# Tech Stack Decisions

## Overview

Every tool chosen with three criteria:
1. **Free tier** — zero cost for portfolio and learning
2. **Local-first** — works offline, no API dependency
3. **Multilingual** — handles Arabic + English correctly

---

## Core Stack

### PyMuPDF (fitz)
| Aspect | Detail |
|--------|--------|
| **Role** | PDF text extraction |
| **Why this one** | Fastest PDF parser for clean PDFs. Best Arabic/RTL support among free tools. Handles ligatures and bidirectional text properly. |
| **Alternatives** | pdfplumber (slower, better for tables), pdfminer.six (more accurate but complex), unstructured (heavy dependencies) |
| **Free limits** | Unlimited (Apache 2.0) |
| **Fallback** | OCR via Tesseract for scanned pages |

### langchain-text-splitters
| Aspect | Detail |
|--------|--------|
| **Role** | Recursive character text splitting for chunking |
| **Why this one** | Reliable sentence-boundary-aware splitting with configurable separators (including Arabic punctuation) |
| **Alternatives** | Custom splitter (more work), spaCy sentence tokenizer (heavier) |
| **Free limits** | Unlimited (MIT) |
| **Note** | This is the only LangChain package used. The RAG pipeline is hand-built for full control over hybrid retrieval, citation parsing, and language-aware prompts. |

### sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
| Aspect | Detail |
|--------|--------|
| **Role** | Create vector embeddings from text chunks |
| **Why this one** | 384-dimension multilingual embeddings supporting 50+ languages. 120MB model — runs on CPU. Arabic support is excellent. |
| **Alternatives** | multilingual-e5-large (better quality, 1GB, slower), bge-m3 (best multilingual, 2GB, GPU recommended) |
| **Free limits** | Unlimited (Apache 2.0) |
| **Why not OpenAI** | Paid API, no offline mode |

### ChromaDB
| Aspect | Detail |
|--------|--------|
| **Role** | Vector database for similarity search |
| **Why this one** | Simplest free local vector DB. Persistent storage as files. Python-native API. Good enough for single-user portfolio. |
| **Alternatives** | FAISS (faster, no built-in persistence), Qdrant (better for production, heavier), Weaviate (requires Docker) |
| **Free limits** | Unlimited (Apache 2.0) |

### Ollama (llama3.2:3b)
| Aspect | Detail |
|--------|--------|
| **Role** | Local LLM for answer generation |
| **Why this one** | 3B parameter model — ~2GB RAM, fast on CPU. Good Arabic + English. One-command install (`ollama pull`). |
| **Alternatives** | llama3.1:8b (stronger but slower), mistral:7b (stronger English, weaker Arabic) |
| **Free limits** | Unlimited (local, MIT) |
| **Risk** | Small model may miss details in complex answers. Upgrade to 8B+ for production. |

### Google Gemini 1.5 Flash (Optional Fallback)
| Aspect | Detail |
|--------|--------|
| **Role** | Cloud LLM fallback when local model is too slow (not currently wired as default) |
| **Why this one** | Best free tier: 1500 requests/day, 1M token context. Built-in multilingual support. No credit card required. |
| **Alternatives** | Groq (fast, no Arabic optimization), Together AI ($1 free credit) |
| **Free limits** | 1500 requests/day, rate-limited |

### Streamlit
| Aspect | Detail |
|--------|--------|
| **Role** | Chat UI for asking questions |
| **Why this one** | Fastest path to a portfolio-ready UI. Built-in chat components. RTL support for Arabic. |
| **Alternatives** | Gradio (more ML-focused UI), Chainlit (RAG-specific, less known to hiring managers) |
| **Free limits** | Unlimited |

### FastAPI
| Aspect | Detail |
|--------|--------|
| **Role** | REST API backend for programmatic access |
| **Why this one** | Auto-generates OpenAPI docs. Async support. Standard for AI/ML backends. Portfolio signal. |
| **Endpoints** | /upload, /documents, /documents/{id}, /documents/{id}/reindex, /query, /health, /stats |
| **Free limits** | Unlimited (MIT) |

---

## Infrastructure

| Tool | Role | Why |
|------|------|-----|
| GitHub Actions | CI/CD | Free for public repos. Runs lint, format check, tests on every push. |
| Docker (optional) | Containerization | Consistent environment for deployment. |
| Makefile | Task automation | `make test`, `make run`, `make ingest` — standard for developers. |
| Ruff | Linter/Formatter | Fastest Python linter. Catches bugs early. |
| Pytest | Testing | Industry standard. 163+ tests. |

---

## Cost Breakdown

| Tool | Cost | Notes |
|------|------|-------|
| Python, PyMuPDF, langchain-text-splitters, ChromaDB, sentence-transformers | $0 | Local, open source |
| Ollama | $0 | Local, open source |
| Gemini Flash | $0 | Free tier: 1500 req/day (optional fallback) |
| Streamlit | $0 | Free |
| GitHub | $0 | Free: public repos + Actions |
| **Total** | **$0** | — |

---

## What We'd Change for Production

1. **Ollama → OpenAI / Anthropic** — Better quality, lower latency
2. **ChromaDB → Qdrant / Pinecone** — Scalable, hosted vector DB
3. **sentence-transformers → OpenAI embeddings** — Better quality, simpler
4. **Small model → llama3.3:70b or hosted API** — Higher answer quality
5. **Streamlit → React/Next.js** — More customizable UI
6. **Single machine → Cloud deployment (AWS/GCP)** — Scalable, monitored
7. **No auth → Authentication + user isolation** — Multi-user support
