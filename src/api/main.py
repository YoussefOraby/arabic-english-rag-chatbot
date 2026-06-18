"""FastAPI REST API for the RAG chatbot.
Usage: uvicorn src.api.main:app --reload
   or: python -m uvicorn src.api.main:app --reload
"""

import hashlib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    Citation,
    DeleteResponse,
    DocumentResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    StatsResponse,
    UploadResponse,
)
from src.config import settings
from src.document.processor import process_document
from src.document.registry import DocumentRegistry
from src.document.schemas import DocumentStatus
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore
from src.llm.ollama_llm import OllamaLLM
from src.rag.chain import RAGChain

app = FastAPI(
    title="RAG Chatbot API",
    description="Arabic-English RAG chatbot powered by Ollama + ChromaDB",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy singletons ──
_chain: RAGChain | None = None
_db_path = Path(
    os.environ.get("DOCUMENTS_DB_PATH")
    or (Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "documents.db")
)
_registry: DocumentRegistry | None = None


def _get_chain() -> RAGChain:
    global _chain
    if _chain is None:
        embedder = Embedder(settings.embeddings)
        store = ChromaStore(settings.vector_store, embedder)
        llm = OllamaLLM(**settings.llm.ollama.model_dump())
        _chain = RAGChain(store=store, llm=llm)
    return _chain


def _get_registry() -> DocumentRegistry:
    global _registry
    if _registry is None:
        _registry = DocumentRegistry(_db_path)
    return _registry


# ── Helpers ───────────────────────────────────────────────────────────


def _preview_text(text: str, max_chars: int = 120) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.6:
        return truncated[:last_space] + "..."
    return truncated.rstrip() + "..."


def _sanitize_filename(name: str) -> str:
    """Strip path separators and dangerous characters from a filename."""
    name = Path(name).name  # Strip any directory components
    name = re.sub(r"[^\w\.\-\(\) ]", "_", name)
    return name or "uploaded.pdf"


def _validate_file(file: UploadFile) -> None:
    max_bytes = settings.document.max_file_size_mb * 1024 * 1024
    ext = Path(file.filename or "").suffix.lower()
    allowed = settings.document.allowed_extensions
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}",
        )
    # Read only the first bytes to check size (we need to seek back)
    contents = file.file.read()
    file.file.seek(0)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents) / 1024 / 1024:.1f} MB). Max: {settings.document.max_file_size_mb} MB",
        )


# ── System Endpoints ──────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service health."""
    try:
        chain = _get_chain()
        count = chain.store.count_chunks()
        ollama_ok = chain.llm.is_available()
        return HealthResponse(status="ok", ollama=ollama_ok, chroma_chunks=count)
    except Exception as e:
        return HealthResponse(status=f"error: {e}", ollama=False, chroma_chunks=0)


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return system statistics."""
    try:
        chain = _get_chain()
        count = chain.store.count_chunks()
    except Exception:
        count = 0
    return StatsResponse(
        total_chunks=count,
        collection_name=settings.vector_store.collection_name,
        persist_directory=settings.vector_store.persist_directory,
        embedding_model=settings.embeddings.model_name,
        llm_model=settings.llm.ollama.model,
        top_k=settings.retrieval.top_k,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Ask a question and get an answer with citations."""
    try:
        chain = _get_chain()
        hist = [m.model_dump() for m in req.history] if req.history else None
        result = chain.query(
            req.question,
            history=hist,
            document_id=req.document_id,
        )

        chunks = [
            RetrievedChunk(
                source=c.chunk.source_file,
                page=c.chunk.page_num,
                score=round(c.score, 4),
                preview=_preview_text(c.chunk.text, 120),
            )
            for c in result["chunks"]
        ]

        citations = [
            Citation(
                pages=c["pages"],
                text_before=c.get("text_before", ""),
                verified=c.get("verified", False),
            )
            for c in result.get("citations", [])
        ]

        return QueryResponse(
            answer=result["answer"],
            source_documents=result["source_documents"],
            chunks=chunks,
            citations=citations,
            insufficient_data=result.get("insufficient_data", False),
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


# ── Document Management Endpoints ─────────────────────────────────────


@app.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Upload a PDF — saves, creates registry entry, and processes in background."""
    _validate_file(file)

    safe_name = _sanitize_filename(file.filename or "document.pdf")
    raw_dir = Path(settings.vector_store.persist_directory).parent.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_path = raw_dir / safe_name

    # Resolve to prevent path traversal
    save_path = save_path.resolve()
    if not str(save_path).startswith(str(raw_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid file path")

    contents = await file.read()

    # Calculate SHA256 for duplicate detection
    file_hash = hashlib.sha256(contents).hexdigest()

    registry = _get_registry()
    existing = registry.get_by_hash(file_hash)
    if existing and existing.status in (
        DocumentStatus.INDEXED,
        DocumentStatus.PROCESSING,
    ):
        return UploadResponse(
            document_id=existing.document_id,
            filename=existing.filename,
            status=existing.status.value,
            message="Document already uploaded and processed.",
        )

    save_path.write_bytes(contents)

    doc = registry.create(safe_name, file_size_bytes=len(contents), file_hash=file_hash)

    background_tasks.add_task(
        process_document,
        doc.document_id,
        save_path,
        safe_name,
        registry,
    )

    return UploadResponse(
        document_id=doc.document_id,
        filename=safe_name,
        status=doc.status.value,
        message="Upload accepted. Document is being processed in the background.",
    )


@app.get("/documents", response_model=list[DocumentResponse])
async def list_documents():
    """List all uploaded documents with their status."""
    registry = _get_registry()
    docs = registry.list()
    return [
        DocumentResponse(
            document_id=d.document_id,
            filename=d.filename,
            upload_time=d.upload_time.isoformat(),
            status=d.status.value,
            file_size_bytes=d.file_size_bytes,
            num_pages=d.num_pages,
            num_chunks=d.num_chunks,
            error_message=d.error_message,
        )
        for d in docs
    ]


@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get a single document by ID."""
    registry = _get_registry()
    doc = registry.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        upload_time=doc.upload_time.isoformat(),
        status=doc.status.value,
        file_size_bytes=doc.file_size_bytes,
        num_pages=doc.num_pages,
        num_chunks=doc.num_chunks,
        error_message=doc.error_message,
    )


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Delete a document and all its chunks/vectors."""
    registry = _get_registry()
    doc = registry.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    chain = _get_chain()
    chunks_removed = chain.store.delete_by_document(document_id)
    registry.delete(document_id)

    # Rebuild BM25 if hybrid is enabled
    if settings.retrieval.enable_hybrid:
        chain.rebuild_index()

    return DeleteResponse(
        document_id=document_id,
        deleted=True,
        chunks_removed=chunks_removed,
        message=f"Document '{doc.filename}' deleted with {chunks_removed} chunks.",
    )


@app.post("/documents/{document_id}/reindex", response_model=UploadResponse)
async def reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Re-index a document: delete existing vectors, re-process from source PDF."""
    registry = _get_registry()
    doc = registry.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    raw_dir = Path(settings.vector_store.persist_directory).parent.parent / "raw"
    pdf_path = (raw_dir / doc.filename).resolve()
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Source PDF '{doc.filename}' not found on disk",
        )

    # Remove existing vectors
    chain = _get_chain()
    chain.store.delete_by_document(document_id)

    # Reset status and re-process
    registry.update_status(document_id, DocumentStatus.UPLOADED)
    background_tasks.add_task(
        process_document,
        document_id,
        pdf_path,
        doc.filename,
        registry,
    )

    return UploadResponse(
        document_id=document_id,
        filename=doc.filename,
        status="processing",
        message="Re-index started. Document is being re-processed in the background.",
    )
