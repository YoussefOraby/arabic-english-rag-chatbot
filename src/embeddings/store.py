"""ChromaDB vector store for storing and retrieving document chunks."""

from datetime import UTC, datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import VectorStoreConfig, settings
from src.embeddings.embedder import Embedder
from src.pdf.chunker import Chunk


class SearchResult:
    """Result from a similarity search."""

    def __init__(self, chunk: Chunk, score: float):
        self.chunk = chunk
        self.score = score  # Similarity score (0-1)


class ChromaStore:
    """Wrapper around ChromaDB for storing and retrieving document chunks."""

    def __init__(self, config: VectorStoreConfig, embedder: Embedder):
        self.config = config
        self.embedder = embedder

        persist_dir = Path(config.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": config.distance_metric},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed chunks and store them in ChromaDB."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        embedding_model = settings.embeddings.model_name
        now_iso = datetime.now(UTC).isoformat()
        metadatas = [
            {
                "page_num": c.page_num,
                "source_file": c.source_file,
                "document_id": c.document_id or "",
                "chunk_type": c.chunk_type,
                "embedding_model": embedding_model,
                "created_at": now_iso,
            }
            for c in chunks
        ]

        embeddings = self.embedder.embed_texts(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count deleted."""
        existing = self.collection.get(where={"document_id": document_id})
        ids = existing["ids"]
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.3,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """Search for chunks similar to the query text.

        Args:
            query: User query text
            k: Number of results
            score_threshold: Minimum similarity score (0-1)
            document_id: Optional — restrict search to a single document
        """
        query_embedding = self.embedder.embed_query(query)

        where_filter = {"document_id": document_id} if document_id else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
            where=where_filter,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        search_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            score = 1.0 - distance

            if score < score_threshold:
                continue

            meta = results["metadatas"][0][i]
            chunk = Chunk(
                text=results["documents"][0][i],
                page_num=meta.get("page_num", 0),
                source_file=meta.get("source_file", ""),
                chunk_id=results["ids"][0][i],
                document_id=meta.get("document_id", ""),
                chunk_type=meta.get("chunk_type", "text"),
                embedding_model=meta.get("embedding_model", ""),
                created_at=meta.get("created_at", ""),
            )
            search_results.append(SearchResult(chunk=chunk, score=score))

        return search_results

    def count_chunks(self) -> int:
        """Return total number of chunks in the store."""
        return self.collection.count()

    def get_all_chunks(self) -> list[Chunk]:
        """Return every chunk stored in the ChromaDB collection."""
        all_data = self.collection.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            return []
        chunks = []
        for i in range(len(all_data["ids"])):
            meta = all_data["metadatas"][i]
            chunks.append(
                Chunk(
                    text=all_data["documents"][i],
                    page_num=meta.get("page_num", 0),
                    source_file=meta.get("source_file", ""),
                    chunk_id=all_data["ids"][i],
                    document_id=meta.get("document_id", ""),
                    chunk_type=meta.get("chunk_type", "text"),
                    embedding_model=meta.get("embedding_model", ""),
                    created_at=meta.get("created_at", ""),
                )
            )
        return chunks

    def clear(self) -> None:
        """Clear all stored chunks."""
        self.client.delete_collection(self.config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )
