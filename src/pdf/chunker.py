"""Text chunking — mixed semantic + recursive strategy.
Splits each page individually to preserve page numbers for citations."""

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import PDFConfig
from src.pdf.extractor import PageText


@dataclass
class Chunk:
    """A single chunk of text or table from a document."""

    text: str
    page_num: int
    source_file: str
    chunk_id: str
    chunk_type: str = "text"  # "text" or "table"
    table_index: int | None = None
    document_id: str = ""
    original_text: str = ""
    processed_text: str = ""
    embedding_model: str = ""
    created_at: str = ""


# ── Sentence-boundary regex for English + Arabic ──
_SENTENCE_PATTERN = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=[.?!؟])\s+(?=[A-Z\u0600-\u06FF"\'(])',
    re.UNICODE,
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences preserving punctuation."""
    parts = _SENTENCE_PATTERN.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_pages(
    pages: list[PageText],
    config: PDFConfig,
    source_file: str = "unknown.pdf",
    embedder: object | None = None,
    pdf_path: Path | None = None,
) -> list[Chunk]:
    """
    Split extracted pages into chunks. When *pdf_path* is provided, also
    extracts tables and includes them as table-type chunks.

    When *embedder* is provided, uses semantic chunking (sentence-split
    then merge-by-similarity).  Otherwise falls back to
    RecursiveCharacterTextSplitter.

    Args:
        pages: List of PageText objects from extract_pages()
        config: PDF configuration
        source_file: Name of the source PDF file
        embedder: Optional Embedder instance for semantic chunking
        pdf_path: Optional path to PDF for table extraction

    Returns:
        List of Chunk objects ready for embedding (text + tables)
    """
    if embedder is not None:
        chunks = _semantic_chunk_pages(pages, config, source_file, embedder)
    else:
        chunks = _recursive_chunk_pages(pages, config, source_file)

    if pdf_path is not None:
        table_chunks = extract_table_chunks(pdf_path, source_file)
        chunks.extend(table_chunks)

        if config.enable_chart_detection:
            try:
                from src.pdf.charts import extract_chart_chunks

                chart_chunks = extract_chart_chunks(pdf_path, source_file)
                chunks.extend(chart_chunks)
            except Exception:
                pass

    return chunks


def extract_table_chunks(pdf_path: Path, source_file: str) -> list[Chunk]:
    """Extract tables from a PDF and return them as Chunk objects."""
    from src.pdf.tables import extract_tables

    try:
        tables = extract_tables(pdf_path)
    except Exception:
        return []

    if not tables:
        return []

    chunks: list[Chunk] = []
    for tbl in tables:
        chunk_id = f"{source_file}_p{tbl.page_num}_t{tbl.table_index}_{uuid.uuid4().hex[:8]}"
        chunks.append(
            Chunk(
                text=tbl.markdown,
                page_num=tbl.page_num,
                source_file=source_file,
                chunk_id=chunk_id,
                chunk_type="table",
                table_index=tbl.table_index,
            )
        )
    return chunks


# ────────────────────── Recursive (fallback) ───────────────────────


def _recursive_chunk_pages(
    pages: list[PageText],
    config: PDFConfig,
    source_file: str,
) -> list[Chunk]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
        length_function=len,
    )

    chunks = []
    for page in pages:
        if not page.text.strip():
            continue
        page_chunks = text_splitter.split_text(page.text)
        for chunk_text in page_chunks:
            if len(chunk_text.strip()) < config.min_chunk_length:
                continue
            chunks.append(
                Chunk(
                    text=chunk_text.strip(),
                    page_num=page.page_num,
                    source_file=source_file,
                    chunk_id=f"{source_file}_p{page.page_num}_{uuid.uuid4().hex[:8]}",
                )
            )
    return chunks


# ────────────────────── Semantic chunker ───────────────────────────


def _semantic_chunk_pages(
    pages: list[PageText],
    config: PDFConfig,
    source_file: str,
    embedder: object,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        page_chunks = _split_page_semantically(
            page.text,
            config,
            embedder,
        )
        for chunk_text in page_chunks:
            if len(chunk_text.strip()) < config.min_chunk_length:
                continue
            chunks.append(
                Chunk(
                    text=chunk_text.strip(),
                    page_num=page.page_num,
                    source_file=source_file,
                    chunk_id=f"{source_file}_p{page.page_num}_{uuid.uuid4().hex[:8]}",
                )
            )
    return chunks


def _split_page_semantically(
    text: str,
    config: PDFConfig,
    embedder: object,
    similarity_threshold: float = 0.55,
) -> list[str]:
    """
    Split a single page's text into semantically coherent chunks.

    Strategy:
      1. Split text into sentences.
      2. Batch-embed sentences.
      3. Walk left-to-right, merging sentences into a candidate window.
      4. When the window length exceeds *chunk_size* (or the next
         sentence has low cosine similarity to the window centroid),
         emit the current window and start a new one.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return sentences or [text]

    # Batch-embed all sentences at once
    try:
        embeddings = np.array(embedder.embed_texts(sentences))
    except Exception:
        # Fallback to recursive split if embeddings fail
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )
        return splitter.split_text(text)

    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        # Greedily grow a chunk
        end = start
        window_chars = 0
        centroid: np.ndarray | None = None

        while end < len(sentences):
            next_len = len(sentences[end])
            if window_chars + next_len > config.chunk_size and end > start:
                break  # don't exceed max size (unless single sentence)

            if end > start and centroid is not None:
                # Check similarity with the next sentence
                sim = float(embeddings[end] @ centroid)
                if sim < similarity_threshold and window_chars > config.chunk_size // 2:
                    break

            # Accept sentence
            window_chars += next_len
            if centroid is None:
                centroid = embeddings[end].copy()
            else:
                centroid = (centroid * (end - start) + embeddings[end]) / (end - start + 1)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            end += 1

        chunk_text = " ".join(sentences[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start = end

    return chunks if chunks else [text]
