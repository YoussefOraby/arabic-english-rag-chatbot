"""Pydantic models for document management."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    document_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    filename: str
    upload_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: DocumentStatus = DocumentStatus.UPLOADED
    file_size_bytes: int = 0
    file_hash: str = ""
    num_pages: int | None = None
    num_chunks: int | None = None
    error_message: str | None = None
