"""
Agentic RAG System Configuration

Gemini-only configuration for optimized 2GB droplet deployment.
API key can come from .env (server default) or per-session (user-provided).
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration — Gemini-only deployment."""

    # Milvus Lite (file-based, no server needed)
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")

    # Gemini LLM
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

    # Gemini Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "3072"))

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Server-level API key (optional — users can provide their own)
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # Security
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
    allowed_origins: list[str] = field(default_factory=lambda: os.getenv(
        "ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"])


config = Config()
