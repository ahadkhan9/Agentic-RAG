"""
Milvus Vector Database Client

Handles connection, collection management, and hybrid search operations.
Uses Milvus Lite for local deployment (no Docker required).
Embeddings via Gemini API (gemini-embedding-001) — no local PyTorch needed.

Supports:
- doc_id field for document-level queries
- position field (0.0-1.0) for neighbor expansion
- chunk_total for coverage tracking
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

        from pymilvus import MilvusClient as PyMilvusClient

        logger.info("🔗 Connecting to Milvus Lite (./milvus_data.db)")
        self._client = PyMilvusClient("./milvus_data.db")
        self._ensure_collection()

    def _get_embedding_client(self) -> genai.Client:
        return get_gemini_client(self._api_key)

    def _ensure_collection(self):
        """Create collection with doc_id and position fields."""
        from pymilvus import DataType

        if self._client.has_collection(self.collection_name):
            logger.debug(f"Collection '{self.collection_name}' exists")
            return

        logger.info(f"📦 Creating collection '{self.collection_name}'")

        schema = self._client.create_schema(auto_id=True, enable_dynamic_field=True)

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=config.embedding_dim)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="file_type", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="page_number", datatype=DataType.INT32)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT32)
        # New fields for document-level queries
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="position", datatype=DataType.FLOAT)  # 0.0 to 1.0
        schema.add_field(field_name="chunk_total", datatype=DataType.INT32)

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

    def embed_text(self, text: str, metrics=None) -> list[float]:
        """Generate embedding for a single text via Gemini API."""
        client = self._get_embedding_client()

        def _call():
            return client.models.embed_content(
                model=config.embedding_model,
                contents=text,
            )

        result = call_with_retry(_call)
        if metrics is not None:
            metrics.record_embedding_call(1, len(text))
        return result.embeddings[0].values

    def embed_texts(self, texts: list[str], metrics=None) -> list[list[float]]:
        """Generate embeddings for multiple texts. Batches in chunks of 100."""
        all_embeddings = []
        batch_size = 100

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
            if metrics is not None:
                metrics.record_embedding_call(len(batch), sum(len(t) for t in batch))

        return all_embeddings

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def insert_chunks(self, chunks: list[TextChunk], doc_id: str = "", metrics=None) -> int:
        """Insert text chunks with doc_id and position metadata."""
        if not chunks:
            return 0

        logger.debug(f"Embedding {len(chunks)} chunks via Gemini API...")
        total = len(chunks)
        contents = [c.content for c in chunks]
        embeddings = self.embed_texts(contents, metrics=metrics)

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
                "doc_id": doc_id,
                "position": round(i / max(total - 1, 1), 4),
                "chunk_total": total,
            })

        logger.debug("Inserting into Milvus...")
        self._client.insert(collection_name=self.collection_name, data=data)
        logger.info(f"✅ Inserted {len(chunks)} chunks (doc_id={doc_id})")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        metrics=None,
    ) -> list[dict]:
        """Perform semantic search on the collection."""
        query_embedding = self.embed_text(query, metrics=metrics)

        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=[
                "content", "source_file", "file_type", "page_number",
                "section", "doc_id", "position", "chunk_total",
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
                    "doc_id": hit["entity"].get("doc_id", ""),
                    "position": hit["entity"].get("position", 0.0),
                    "chunk_total": hit["entity"].get("chunk_total", 0),
                })

        return documents

    def get_chunks_by_doc_id(self, doc_id: str) -> list[dict]:
        """Fetch ALL chunks for a given document, ordered by position."""
        filter_expr = f'doc_id == "{doc_id}"'
        results = self._client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=[
                "content", "source_file", "file_type", "page_number",
                "section", "doc_id", "position", "chunk_total", "chunk_index",
            ],
            limit=config.max_summary_chunks,
        )
        # Sort by position
        results.sort(key=lambda x: x.get("position", 0.0))
        return results

    def get_neighbor_chunks(self, doc_id: str, position: float, window: float = 0.05) -> list[dict]:
        """Fetch chunks near a given position in a document."""
        low = max(0.0, position - window)
        high = min(1.0, position + window)
        filter_expr = f'doc_id == "{doc_id}" and position >= {low} and position <= {high}'

        results = self._client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=[
                "content", "source_file", "file_type", "page_number",
                "section", "doc_id", "position", "chunk_total",
            ],
            limit=5,
        )
        results.sort(key=lambda x: x.get("position", 0.0))
        return results

    def get_collection_stats(self) -> dict:
        if not self._client.has_collection(self.collection_name):
            return {"exists": False}
        stats = self._client.get_collection_stats(self.collection_name)
        return {
            "exists": True,
            "name": self.collection_name,
            "num_entities": stats.get("row_count", 0),
        }

    def delete_collection(self):
        if self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)
            logger.info(f"🗑️ Deleted collection '{self.collection_name}'")

    def reset_collection(self):
        self.delete_collection()
        self._ensure_collection()
        logger.info(f"🔄 Collection '{self.collection_name}' reset")


# --------------------------------------------------------------------------
# Singleton with per-session API key support
# --------------------------------------------------------------------------

_client: Optional[MilvusClient] = None
_client_api_key: Optional[str] = None


def get_milvus_client(api_key: Optional[str] = None) -> MilvusClient:
    """Get or create the Milvus client singleton."""
    global _client, _client_api_key

    if _client is None or api_key != _client_api_key:
        _client_api_key = api_key
        _client = MilvusClient(api_key=api_key)
    return _client


def reset_milvus_client():
    global _client
    if _client is not None:
        _client.reset_collection()
    else:
        _client = MilvusClient()
