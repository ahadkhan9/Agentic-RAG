"""
Retriever Agent

Handles document retrieval and context building.
Uses semantic search with optional re-ranking.
"""
from dataclasses import dataclass
from typing import Optional

from vectordb.milvus_client import get_milvus_client
from logger import get_logger

# Initialize logger
logger = get_logger("Retriever")


@dataclass
class RetrievalResult:
    """A retrieved document with relevance info."""
    content: str
    source_file: str
    file_type: str
    page_number: Optional[int]
    section: Optional[str]
    score: float


@dataclass
class Citation:
    """Citation information for a source."""
    source_file: str
    page_number: Optional[int]
    section: Optional[str]
    excerpt: str


def retrieve_documents(
    query: str,
    top_k: int = 5,
    file_type_filter: Optional[str] = None
) -> list[RetrievalResult]:
    """
    Retrieve relevant documents for a query.
    
    Args:
        query: The search query
        top_k: Number of results to return
        file_type_filter: Optional filter by file type (pdf, docx, etc.)
    
    Returns:
        List of RetrievalResult objects sorted by relevance
    """
    logger.debug(f"🔍 Searching: '{query[:50]}...' (top_k={top_k})")
    client = get_milvus_client()
    
    # Build filter expression if needed
    filter_expr = None
    if file_type_filter:
        filter_expr = f'file_type == "{file_type_filter}"'
        logger.debug(f"Filter: {filter_expr}")
    
    # Search
    results = client.search(query=query, top_k=top_k, filter_expr=filter_expr)
    
    # Convert to RetrievalResult objects
    retrieval_results = []
    for doc in results:
        retrieval_results.append(RetrievalResult(
            content=doc["content"],
            source_file=doc["source_file"],
            file_type=doc["file_type"],
            page_number=doc["page_number"] if doc["page_number"] > 0 else None,
            section=doc["section"] if doc["section"] else None,
            score=doc["score"]
        ))
    
    logger.info(f"📚 Found {len(retrieval_results)} documents")
    for i, r in enumerate(retrieval_results[:3], 1):
        logger.debug(f"  [{i}] {r.source_file} (score: {r.score:.3f})")
    
    # Simple re-ranking: penalize template files (they often have headers but empty data)
    for result in retrieval_results:
        if 'template' in result.source_file.lower():
            result.score *= 0.7  # 30% penalty for templates
    
    # Re-sort after penalty
    retrieval_results.sort(key=lambda x: x.score, reverse=True)
    
    return retrieval_results


def retrieve_for_multiple_queries(queries: list[str], top_k_per_query: int = 5) -> list[RetrievalResult]:
    """
    Retrieve documents for multiple queries and deduplicate.
    Used for multi-part query decomposition.
    Uses higher top_k to ensure we get enough context for all sub-queries.
    """
    all_results = []
    seen_contents = set()
    
    for query in queries:
        results = retrieve_documents(query, top_k=top_k_per_query)
        for result in results:
            # Deduplicate by content
            content_key = result.content[:100]  # First 100 chars as key
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                all_results.append(result)
    
    # Sort by score
    all_results.sort(key=lambda x: x.score, reverse=True)
    
    return all_results


def build_context(results: list[RetrievalResult], max_length: int = 4000) -> tuple[str, list[Citation]]:
    """
    Build context string from retrieval results.
    Returns context and citations for source attribution.
    """
    context_parts = []
    citations = []
    current_length = 0
    
    for i, result in enumerate(results, 1):
        # Build source reference
        source_ref = f"[Source {i}: {result.source_file}"
        if result.page_number:
            source_ref += f", Page {result.page_number}"
        if result.section:
            source_ref += f", Section: {result.section}"
        source_ref += "]"
        
        # Add to context if within limit
        entry = f"{source_ref}\n{result.content}\n"
        if current_length + len(entry) > max_length:
            break
            
        context_parts.append(entry)
        current_length += len(entry)
        
        # Create citation
        citations.append(Citation(
            source_file=result.source_file,
            page_number=result.page_number,
            section=result.section,
            excerpt=result.content[:100] + "..." if len(result.content) > 100 else result.content
        ))
    
    context = "\n".join(context_parts)
    
    # Return all citations (no deduplication - LLM references by index)
    return context, citations


def format_citations(citations: list[Citation]) -> str:
    """Format citations for display in response."""
    if not citations:
        return ""
    
    formatted = "\n\n**Sources:**\n"
    for i, cit in enumerate(citations, 1):
        source_info = f"{i}. {cit.source_file}"
        if cit.page_number:
            source_info += f", Page {cit.page_number}"
        if cit.section:
            source_info += f" ({cit.section})"
        formatted += f"{source_info}\n"
    
    return formatted
