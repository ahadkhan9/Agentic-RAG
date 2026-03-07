"""
Milvus Vector Database Client

Handles connection, collection management, and hybrid search operations.
Uses Milvus Lite for local deployment (no Docker required).
Embeddings via Gemini API (gemini-embedding-001) — no local PyTorch needed.
"""
from typing import Optional

from google import genai

from config import config
from ingestion.chunker import TextChunk
from logger import get_logger
from utils import get_gemini_client, call_with_retry

logger = get_logger("MilvusClient")


class MilvusClient:
    """Client for Milvus vector database operations with Gemini embeddings."""

    def __init__(self, api_key: Optional[str] = None):
        self.collection_name = config.collection_name
        self._api_key = api_key

        # Lazy import — pymilvus is lightweight, but defer anyway
        from pymilvus import MilvusClient as PyMilvusClient

        logger.info("🔗 Connecting to Milvus Lite (./milvus_data.db)")
        self._client = PyMilvusClient("./milvus_data.db")
        self._ensure_collection()

    def _get_embedding_client(self) -> genai.Client:
        """Get Gemini client for embedding calls."""
        return get_gemini_client(self._api_key)

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        from pymilvus import DataType

        if self._client.has_collection(self.collection_name):
            logger.debug(f"Collection '{self.collection_name}' exists")
            return

        logger.info(f"📦 Creating collection '{self.collection_name}'")

        schema = self._client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )

        schema.add_field(
            field_name="id", datatype=DataType.INT64,
            is_primary=True, auto_id=True,
        )
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR,
            dim=config.embedding_dim,
        )
        schema.add_field(
            field_name="content", datatype=DataType.VARCHAR, max_length=65535,
        )
        schema.add_field(
            field_name="source_file", datatype=DataType.VARCHAR, max_length=512,
        )
        schema.add_field(
            field_name="file_type", datatype=DataType.VARCHAR, max_length=32,
        )
        schema.add_field(
            field_name="page_number", datatype=DataType.INT32,
        )
        schema.add_field(
            field_name="section", datatype=DataType.VARCHAR, max_length=512,
        )
        schema.add_field(
            field_name="chunk_index", datatype=DataType.INT32,
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )

        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"✅ Collection '{self.collection_name}' created")

    # ------------------------------------------------------------------
    # Embedding via Gemini API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text via Gemini API."""
        client = self._get_embedding_client()

        def _call():
            return client.models.embed_content(
                model=config.embedding_model,
                contents=text,
            )

        result = call_with_retry(_call)
        return result.embeddings[0].values

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts via Gemini API.

        Batches requests to stay within API limits (max 100 per call).
        """
        all_embeddings = []
        batch_size = 100  # Gemini API limit per embed_content call

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            client = self._get_embedding_client()

            def _call(b=batch):
                return client.models.embed_content(
                    model=config.embedding_model,
                    contents=b,
                )

            result = call_with_retry(_call)
            all_embeddings.extend([e.values for e in result.embeddings])

        return all_embeddings

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def insert_chunks(self, chunks: list[TextChunk]) -> int:
        """Insert text chunks into the collection. Returns inserted count."""
        if not chunks:
            return 0

        logger.debug(f"Embedding {len(chunks)} chunks via Gemini API...")

        contents = [c.content for c in chunks]
        embeddings = self.embed_texts(contents)

        data = []
        for i, chunk in enumerate(chunks):
            data.append({
                "vector": embeddings[i],
                "content": chunk.content,
                "source_file": chunk.source_file,
                "file_type": chunk.file_type,
                "page_number": chunk.page_number or 0,
                "section": chunk.section or "",
                "chunk_index": chunk.chunk_index,
            })

        logger.debug("Inserting into Milvus...")
        self._client.insert(collection_name=self.collection_name, data=data)
        logger.info(f"✅ Inserted {len(chunks)} chunks")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """Perform semantic search on the collection."""
        query_embedding = self.embed_text(query)

        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=[
                "content", "source_file", "file_type", "page_number", "section",
            ],
        )

        documents = []
        for hits in results:
            for hit in hits:
                documents.append({
                    "id": hit["id"],
                    "score": hit["distance"],
                    "content": hit["entity"].get("content"),
                    "source_file": hit["entity"].get("source_file"),
                    "file_type": hit["entity"].get("file_type"),
                    "page_number": hit["entity"].get("page_number"),
                    "section": hit["entity"].get("section"),
                })

        return documents

    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        if not self._client.has_collection(self.collection_name):
            return {"exists": False}

        stats = self._client.get_collection_stats(self.collection_name)
        return {
            "exists": True,
            "name": self.collection_name,
            "num_entities": stats.get("row_count", 0),
        }

    def delete_collection(self):
        """Delete the entire collection."""
        if self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)
            logger.info(f"🗑️ Deleted collection '{self.collection_name}'")

    def reset_collection(self):
        """Delete and recreate the collection."""
        self.delete_collection()
        self._ensure_collection()
        logger.info(f"🔄 Collection '{self.collection_name}' reset")


# --------------------------------------------------------------------------
# Singleton with per-session API key support
# --------------------------------------------------------------------------

_client: Optional[MilvusClient] = None
_client_api_key: Optional[str] = None


def get_milvus_client(api_key: Optional[str] = None) -> MilvusClient:
    """Get or create the Milvus client singleton.

    If api_key changes, recreates the client to use the new key for embeddings.
    """
    global _client, _client_api_key

    if _client is None or api_key != _client_api_key:
        _client_api_key = api_key
        _client = MilvusClient(api_key=api_key)
    return _client


def reset_milvus_client():
    """Reset the Milvus client singleton and collection."""
    global _client
    if _client is not None:
        _client.reset_collection()
    else:
        _client = MilvusClient()
