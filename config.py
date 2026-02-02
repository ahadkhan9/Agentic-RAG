"""
Agentic RAG System Configuration

All values MUST be set in .env file - no fallback defaults.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def require_env(key: str) -> str:
    """Get required environment variable or raise error."""
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Required environment variable '{key}' is not set. Check your .env file.")
    return value


@dataclass
class Config:
    """Application configuration - all values from .env"""
    
    # Milvus Lite (file-based, no server needed)
    collection_name: str = os.getenv("COLLECTION_NAME", "manufacturing_docs")
    
    # LLM - Required, no defaults
    llm_provider: str = require_env("LLM_PROVIDER")
    ollama_model: str = require_env("OLLAMA_MODEL")
    gemini_model: str = require_env("GEMINI_MODEL")
    google_api_key: str = require_env("GOOGLE_API_KEY")
    
    # Embeddings
    embedding_model: str = require_env("EMBEDDING_MODEL")
    embedding_dim: int = 384  # all-MiniLM-L6-v2 dimension
    
    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


config = Config()
