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
| **Fallback** | We'll add `unstructured` later for noisy/scanned PDFs |

### LangChain (core + community + text-splitters)
| Aspect | Detail |
|--------|--------|
| **Role** | RAG orchestration, text splitting, prompt templates |
| **Why this one** | Industry standard for RAG. Modular: import only what we need. Active community. Builds portfolio familiarity. |
| **Alternatives** | LlamaIndex (equally good, different API), custom implementation (more work, less portfolio value) |
| **Free limits** | Unlimited (MIT) |
| **Note** | We use only `langchain-core`, `langchain-community`, `langchain-text-splitters` — not the full monolith |

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

### Ollama (llama3.1:8b-instruct-q4_K_M)
| Aspect | Detail |
|--------|--------|
| **Role** | Local LLM for answer generation |
| **Why this one** | Quantized (4-bit) 8B model — runs on 4.7GB RAM. Good Arabic + English. One-command install (`ollama pull`). |
| **Alternatives** | llama3.2:3b (faster, weaker), mistral:7b (stronger English, weaker Arabic) |
| **Free limits** | Unlimited (local, MIT) |
| **Risk** | May be slow on CPU-only machines → Gemini fallback |

### Google Gemini 1.5 Flash (Fallback)
| Aspect | Detail |
|--------|--------|
| **Role** | Cloud LLM fallback when local model is too slow |
| **Why this one** | Best free tier: 1500 requests/day, 1M token context. Built-in multilingual support. No credit card required. |
| **Alternatives** | Groq (fast, no Arabic optimization), Together AI ($1 free credit) |
| **Free limits** | 1500 requests/day, rate-limited |

### Streamlit
| Aspect | Detail |
|--------|--------|
| **Role** | Chat UI for asking questions |
| **Why this one** | Fastest path to a portfolio-ready UI. Built-in chat components. Free hosting on Streamlit Community Cloud. |
| **Alternatives** | Gradio (more ML-focused UI), Chainlit (RAG-specific, less known to hiring managers) |
| **Free limits** | Unlimited public apps on Community Cloud |

### FastAPI (Phase 5, optional)
| Aspect | Detail |
|--------|--------|
| **Role** | Backend API for programmatic access |
| **Why this one** | Auto-generates OpenAPI docs. Async support. Standard for AI/ML backends. Portfolio signal. |
| **Alternatives** | Flask (simpler, less modern), Django (overkill) |
| **Free limits** | Unlimited (MIT) |
| **Note** | Deferred to Phase 5. Not needed for core portfolio demo. |

---

## Infrastructure

| Tool | Role | Why |
|------|------|-----|
| GitHub Actions | CI/CD | Free for public repos. Runs lint, typecheck, tests on every push. |
| Docker (optional) | Containerization | Consistent environment. Not required for development. |
| Streamlit Community Cloud | Deployment | Free hosting, one-click from GitHub. Better than HF Spaces (no sleep). |
| Makefile | Task automation | `make test`, `make run`, `make ingest` — standard for developers. |
| Ruff | Linter/Formatter | Fastest Python linter. Catches bugs early. |
| MyPy | Type checker | Catches type errors. Portfolio signal for code quality. |
| Pytest | Testing | Industry standard. |

---

## Cost Breakdown

| Tool | Cost | Notes |
|------|------|-------|
| Python, PyMuPDF, LangChain, ChromaDB, sentence-transformers | $0 | Local, open source |
| Ollama | $0 | Local, open source |
| Gemini Flash | $0 | Free tier: 1500 req/day |
| Streamlit | $0 | Free hosting: public apps |
| GitHub | $0 | Free: public repos + Actions |
| **Total** | **$0** | — |

---

## What We'd Change for Production

1. **Ollama → OpenAI / Anthropic** — Better quality, lower latency
2. **ChromaDB → Qdrant / Pinecone** — Scalable, hosted vector DB
3. **sentence-transformers → OpenAI embeddings** — Better quality, simpler
4. **Streamlit → React/Next.js** — More customizable UI
5. **Single machine → Cloud deployment (AWS/GCP)** — Scalable, monitored
