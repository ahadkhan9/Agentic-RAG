"""
Document Registry

On upload, generates a Gemini summary and extracts key topics for each document.
Enables document-level queries like "summarize this doc" and document listing.
"""
import hashlib
from dataclasses import dataclass
from typing import Optional

from config import config
from logger import get_logger
from utils import call_gemini

logger = get_logger("DocRegistry")


@dataclass
class DocMeta:
    """Metadata for a registered document."""
    doc_id: str
    filename: str
    summary: str
    topics: str
    chunk_count: int
    total_chars: int


# In-memory registry (persisted via Milvus would need a second collection —
# for a 2GB droplet, a simple dict backed by JSON is lighter)
_registry: dict[str, DocMeta] = {}


def _make_doc_id(filename: str) -> str:
    """Generate a stable short doc_id from filename."""
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def register_document(
    filename: str,
    full_text: str,
    chunk_count: int,
    api_key: Optional[str] = None,
    metrics=None,
) -> DocMeta:
    """Generate summary & topics for a document and register it.

    Args:
        filename: Original filename
        full_text: Concatenated full text of the document
        chunk_count: Number of chunks created from this document
        api_key: Gemini API key
        metrics: Optional SessionMetrics for token tracking

    Returns:
        DocMeta with doc_id, summary, and topics
    """
    doc_id = _make_doc_id(filename)

    # Truncate text for summary generation (Gemini context is large but let's be safe)
    max_chars = 80_000  # ~20k tokens — enough for a good summary
    text_for_summary = full_text[:max_chars]

    prompt = f"""Analyze this document and provide:
1. A concise 3-5 sentence summary of the entire document
2. A comma-separated list of 5-10 key topics covered

Document text:
---
{text_for_summary}
---

Respond in this exact format:
SUMMARY: <your summary here>
TOPICS: <topic1, topic2, topic3, ...>"""

    try:
        result = call_gemini(prompt, api_key=api_key, metrics=metrics)

        summary = ""
        topics = ""
        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("SUMMARY:"):
                summary = line[8:].strip()
            elif line.upper().startswith("TOPICS:"):
                topics = line[7:].strip()

        if not summary:
            summary = result[:500]  # Fallback: use raw response
        if not topics:
            topics = "general"

    except Exception as e:
        logger.warning(f"⚠️ Failed to generate summary for {filename}: {e}")
        summary = f"Document: {filename} ({chunk_count} chunks)"
        topics = "unknown"

    meta = DocMeta(
        doc_id=doc_id,
        filename=filename,
        summary=summary,
        topics=topics,
        chunk_count=chunk_count,
        total_chars=len(full_text),
    )

    _registry[doc_id] = meta
    logger.info(f"📝 Registered document: {filename} (doc_id={doc_id}, {chunk_count} chunks)")
    return meta


def get_doc_meta(doc_id: str) -> Optional[DocMeta]:
    """Get metadata for a registered document."""
    return _registry.get(doc_id)


def get_doc_meta_by_filename(filename: str) -> Optional[DocMeta]:
    """Look up document metadata by filename."""
    doc_id = _make_doc_id(filename)
    return _registry.get(doc_id)


def list_documents() -> list[DocMeta]:
    """List all registered documents."""
    return list(_registry.values())


def find_doc_for_query(query: str) -> Optional[DocMeta]:
    """Try to identify which document a query is referring to.

    Checks if any registered filename appears in the query.
    Returns the best match or None.
    """
    query_lower = query.lower()
    for meta in _registry.values():
        # Check if filename (without extension) appears in query
        name_no_ext = meta.filename.rsplit(".", 1)[0].lower()
        if name_no_ext in query_lower or meta.filename.lower() in query_lower:
            return meta

    # If only one document is registered, assume the query is about it
    if len(_registry) == 1:
        return list(_registry.values())[0]

    return None


def clear_registry():
    """Clear all registered documents."""
    _registry.clear()
    logger.info("🗑️ Document registry cleared")
