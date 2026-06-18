"""Document registry backed by SQLite.

Tracks document metadata and status (uploaded → processing → indexed / failed).
"""

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.document.schemas import DocumentRecord, DocumentStatus


class DocumentRegistry:
    """SQLite-backed document registry.

    Simple, no extra dependencies, portable.  Thread-safe for single-process
    usage (FastAPI + BackgroundTasks run in the same process).
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── connection helpers ──────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id    TEXT PRIMARY KEY,
                    filename       TEXT NOT NULL,
                    upload_time    TEXT NOT NULL,
                    status         TEXT NOT NULL DEFAULT 'uploaded',
                    file_size_bytes INTEGER NOT NULL DEFAULT 0,
                    file_hash      TEXT NOT NULL DEFAULT '',
                    num_pages      INTEGER,
                    num_chunks     INTEGER,
                    error_message  TEXT
                )
            """)
            # Migrate existing databases that lack the file_hash column
            try:
                conn.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # column already exists

    # ── CRUD ────────────────────────────────────────────────────────────

    def create(
        self, filename: str, file_size_bytes: int = 0, file_hash: str = ""
    ) -> DocumentRecord:
        """Insert a new document record and return it."""
        doc = DocumentRecord(
            document_id=uuid.uuid4().hex[:12],
            filename=filename,
            upload_time=datetime.now(UTC),
            status=DocumentStatus.UPLOADED,
            file_size_bytes=file_size_bytes,
            file_hash=file_hash,
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO documents
                   (document_id, filename, upload_time, status,
                    file_size_bytes, file_hash)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    doc.document_id,
                    doc.filename,
                    doc.upload_time.isoformat(),
                    doc.status.value,
                    doc.file_size_bytes,
                    doc.file_hash,
                ),
            )
        return doc

    def get(self, document_id: str) -> DocumentRecord | None:
        """Fetch a single document by ID, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_filename(self, filename: str) -> DocumentRecord | None:
        """Fetch the most recent document with the given filename."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE filename = ? ORDER BY upload_time DESC LIMIT 1",
                (filename,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_hash(self, file_hash: str) -> DocumentRecord | None:
        """Fetch the most recent document with the given SHA256 hash."""
        if not file_hash:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_hash = ? ORDER BY upload_time DESC LIMIT 1",
                (file_hash,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list(self) -> list[DocumentRecord]:
        """Return all documents ordered by upload time (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY upload_time DESC",
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        num_pages: int | None = None,
        num_chunks: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update document status and optional metadata."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE documents
                   SET status = ?,
                       num_pages       = COALESCE(?, num_pages),
                       num_chunks      = COALESCE(?, num_chunks),
                       error_message   = ?
                   WHERE document_id = ?""",
                (status.value, num_pages, num_chunks, error_message, document_id),
            )

    def clear(self) -> None:
        """Delete all document records."""
        with self._connect() as conn:
            conn.execute("DELETE FROM documents")

    def delete(self, document_id: str) -> bool:
        """Remove document record. Returns True if a row was deleted."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
            return cur.rowcount > 0

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            upload_time=datetime.fromisoformat(row["upload_time"]),
            status=DocumentStatus(row["status"]),
            file_size_bytes=row["file_size_bytes"] or 0,
            file_hash=row["file_hash"] if row["file_hash"] else "",
            num_pages=row["num_pages"],
            num_chunks=row["num_chunks"],
            error_message=row["error_message"],
        )
