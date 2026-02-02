"""
Milvus Vector Database Client

Handles connection, collection management, and hybrid search operations.
Uses Milvus Lite for local development (no Docker required).
"""
from typing import Optional
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient as PyMilvusClient, DataType

from config import config
from ingestion.chunker import TextChunk
from logger import get_logger

# Initialize logger
logger = get_logger("MilvusClient")


class MilvusClient:
    """Client for Milvus vector database operations."""
    
    def __init__(self):
        self.collection_name = config.collection_name
        self.embedder = SentenceTransformer(config.embedding_model)
        # Use Milvus Lite (local file-based, no Docker needed)
        logger.info(f"🔗 Connecting to Milvus Lite (./milvus_data.db)")
        self._client = PyMilvusClient("./milvus_data.db")
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if self._client.has_collection(self.collection_name):
            logger.debug(f"Collection '{self.collection_name}' exists")
            return
            
        logger.info(f"📦 Creating collection '{self.collection_name}'")
        
        # Create schema for manufacturing docs
        schema = self._client.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )
        
        # Add fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=config.embedding_dim)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="file_type", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="page_number", datatype=DataType.INT32)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT32)
        
        # Create index params
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )
        
        # Create collection
        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"✅ Collection '{self.collection_name}' created")
    
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a text string."""
        return self.embedder.encode(text, normalize_embeddings=True).tolist()
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()
    
    def insert_chunks(self, chunks: list[TextChunk]) -> int:
        """
        Insert text chunks into the collection.
        Returns the number of inserted chunks.
        """
        if not chunks:
            return 0
        
        logger.debug(f"Embedding {len(chunks)} chunks...")
        
        # Prepare data
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
        
        logger.debug(f"Inserting into Milvus...")
        self._client.insert(collection_name=self.collection_name, data=data)
        logger.info(f"✅ Inserted {len(chunks)} chunks")
        return len(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: Optional[str] = None
    ) -> list[dict]:
        """
        Perform semantic search on the collection.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filter_expr: Optional filter expression
        
        Returns:
            List of matching documents with metadata
        """
        query_embedding = self.embed_text(query)
        
        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["content", "source_file", "file_type", "page_number", "section"]
        )
        
        documents = []
        for hits in results:
            for hit in hits:
                doc = {
                    "id": hit["id"],
                    "score": hit["distance"],
                    "content": hit["entity"].get("content"),
                    "source_file": hit["entity"].get("source_file"),
                    "file_type": hit["entity"].get("file_type"),
                    "page_number": hit["entity"].get("page_number"),
                    "section": hit["entity"].get("section"),
                }
                documents.append(doc)
        
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
        logger.info(f"🔄 Collection '{self.collection_name}' reset successfully")


# Singleton instance
_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """Get or create the Milvus client singleton."""
    global _client
    if _client is None:
        _client = MilvusClient()
    return _client


def reset_milvus_client():
    """Reset the Milvus client singleton and collection."""
    global _client
    if _client is not None:
        _client.reset_collection()
    else:
        _client = MilvusClient()
