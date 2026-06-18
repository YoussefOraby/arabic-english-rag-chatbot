"""Pydantic schemas for the RAG chatbot API."""

from pydantic import BaseModel, Field

# ── Document Management ───────────────────────────────────────────────


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    upload_time: str
    status: str
    file_size_bytes: int
    num_pages: int | None = None
    num_chunks: int | None = None
    error_message: str | None = None


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    chunks_removed: int
    message: str


# ── Query ─────────────────────────────────────────────────────────────


class HistoryMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question (Arabic or English)")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve")
    document_id: str | None = Field(
        default=None, description="Restrict search to a single document"
    )
    history: list[HistoryMessage] | None = Field(
        default=None, description="Previous conversation turns for context"
    )


class SourceDocument(BaseModel):
    filename: str
    pages: list[int]


class RetrievedChunk(BaseModel):
    source: str
    page: int
    score: float
    preview: str


class Citation(BaseModel):
    pages: list[int]
    text_before: str = ""
    verified: bool = False


class QueryResponse(BaseModel):
    answer: str
    source_documents: dict[str, list[int]]
    chunks: list[RetrievedChunk]
    citations: list[Citation] = []
    insufficient_data: bool = False


# ── System ────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    ollama: bool = False
    chroma_chunks: int = 0


class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    persist_directory: str
    embedding_model: str
    llm_model: str
    top_k: int
