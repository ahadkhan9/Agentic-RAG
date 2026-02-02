"""
Agentic RAG System Configuration

Required values depend on the selected LLM provider.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def require_env(key: str) -> str:
    """Get required environment variable or raise error."""
    value = os.getenv(key)
    if value is None or value == "":
        raise ValueError(f"Required environment variable '{key}' is not set. Check your .env file.")
    return value


@dataclass
class Config:
    """Application configuration - all values from .env"""
    
    # Milvus Lite (file-based, no server needed)
    collection_name: str = os.getenv("COLLECTION_NAME", "manufacturing_docs")
    
    # LLM - Required, no defaults
    llm_provider: str = require_env("LLM_PROVIDER")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Embeddings
    embedding_model: str = require_env("EMBEDDING_MODEL")
    embedding_dim: int = 384  # all-MiniLM-L6-v2 dimension
    
    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        provider = self.llm_provider.lower()
        if provider == "ollama":
            if not self.ollama_model:
                raise ValueError("OLLAMA_MODEL must be set when LLM_PROVIDER=ollama.")
        elif provider == "gemini":
            if not self.gemini_model:
                raise ValueError("GEMINI_MODEL must be set when LLM_PROVIDER=gemini.")
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY must be set when LLM_PROVIDER=gemini.")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER '{self.llm_provider}'. Use 'ollama' or 'gemini'.")


config = Config()
