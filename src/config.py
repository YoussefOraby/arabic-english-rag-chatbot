"""Configuration management using Pydantic Settings + YAML."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PDFConfig(BaseSettings):
    chunk_size: int = 500
    chunk_overlap: int = 100
    separators: list[str] = ["\n\n", "\n", "।", "؟", "،", ". ", "! ", "? ", " ", ""]
    min_chunk_length: int = 50
    ocr_threshold_chars: int = 50
    ocr_language: str = "eng+ara"
    enable_chart_detection: bool = True


class EmbeddingsConfig(BaseSettings):
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True


class VectorStoreConfig(BaseSettings):
    persist_directory: str = "data/processed/chromadb"
    collection_name: str = "documents"
    distance_metric: str = "cosine"


class OllamaConfig(BaseSettings):
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    num_ctx: int = 4096
    timeout: int = 120


class GeminiConfig(BaseSettings):
    model: str = "gemini-1.5-flash"
    temperature: float = 0.1


class LLMConfig(BaseSettings):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)


class RetrievalConfig(BaseSettings):
    top_k: int = 4
    score_threshold: float = 0.3
    include_metadata: bool = True
    enable_hybrid: bool = False
    hybrid_alpha: float = 0.5
    enable_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 10


class RAGConfig(BaseSettings):
    system_prompt_template: str = (
        "You are a helpful assistant that answers questions based ONLY on the provided context. "
        "If the context doesn't contain the answer, say \"I don't know based on the provided documents.\" "
        "Always cite sources using [page X] format at the end of each sentence. "
        "Answer in the SAME LANGUAGE as the question (Arabic or English)."
    )
    max_tokens: int = 512
    max_history_pairs: int = 4


class DocumentConfig(BaseSettings):
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf"]


class UIConfig(BaseSettings):
    page_title: str = "Arabic-English RAG Chatbot"
    page_icon: str = "📚"
    layout: str = "wide"
    max_history: int = 10
    clear_documents_when_no_upload: bool = True


class LoggingConfig(BaseSettings):
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    file: str = "logs/app.log"


class Settings(BaseSettings):
    """Root settings loaded from config.yaml + environment variables."""

    pdf: PDFConfig = Field(default_factory=PDFConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get settings instance (cached).
    Loads from config.yaml first, then overrides with .env / environment variables.
    """
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    yaml_data = load_yaml_config(config_path)
    return Settings(**yaml_data)


# Convenience singleton
settings = get_settings()
