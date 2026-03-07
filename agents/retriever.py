"""
Retriever Agent

Handles document retrieval and context building.
Supports multiple retrieval strategies:
- Standard top-k with neighbor expansion
- Full-document retrieval for summaries
- Dual-concept retrieval for comparisons
- Broad retrieval for synthesis
"""
from dataclasses import dataclass
from typing import Optional

from vectordb.milvus_client import get_milvus_client
from config import config
from logger import get_logger

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
    doc_id: str = ""
    position: float = 0.0


@dataclass
class Citation:
    """Citation information for a source."""
    source_file: str
    page_number: Optional[int]
    section: Optional[str]
    excerpt: str


def _raw_to_result(doc: dict, default_score: float = 1.0) -> RetrievalResult:
    """Convert a raw Milvus document dict to RetrievalResult."""
    return RetrievalResult(
        content=doc.get("content", ""),
        source_file=doc.get("source_file", ""),
        file_type=doc.get("file_type", ""),
        page_number=doc.get("page_number") if doc.get("page_number", 0) > 0 else None,
        section=doc.get("section") if doc.get("section") else None,
        score=doc.get("score", default_score),
        doc_id=doc.get("doc_id", ""),
        position=doc.get("position", 0.0),
    )


# ------------------------------------------------------------------
# Retrieval Strategies
# ------------------------------------------------------------------

def retrieve_documents(
    query: str,
    top_k: int = None,
    file_type_filter: Optional[str] = None,
    api_key: Optional[str] = None,
    metrics=None,
) -> list[RetrievalResult]:
    """Standard retrieval: top-k vector search with optional neighbor expansion."""
    top_k = top_k or config.default_top_k
    logger.debug(f"🔍 Searching: '{query[:50]}...' (top_k={top_k})")
    client = get_milvus_client(api_key=api_key)

    filter_expr = None
    if file_type_filter:
        filter_expr = f'file_type == "{file_type_filter}"'

    results = client.search(query=query, top_k=top_k, filter_expr=filter_expr, metrics=metrics)

    retrieval_results = [_raw_to_result(doc) for doc in results]
    retrieval_results.sort(key=lambda x: x.score, reverse=True)

    logger.info(f"📚 Found {len(retrieval_results)} documents")
    return retrieval_results


def retrieve_with_neighbors(
    query: str,
    top_k: int = None,
    api_key: Optional[str] = None,
    metrics=None,
) -> list[RetrievalResult]:
    """Retrieve top-k results plus their neighboring chunks for context continuity."""
    top_k = top_k or config.default_top_k
    client = get_milvus_client(api_key=api_key)

    results = client.search(query=query, top_k=top_k, metrics=metrics)
    base_results = [_raw_to_result(doc) for doc in results]

    # Expand: fetch neighbors for top-3 results
    expanded = list(base_results)
    seen_content = {r.content[:80] for r in expanded}

    for result in base_results[:3]:
        if result.doc_id:
            neighbors = client.get_neighbor_chunks(result.doc_id, result.position)
            for n in neighbors:
                key = n.get("content", "")[:80]
                if key not in seen_content:
                    seen_content.add(key)
                    expanded.append(_raw_to_result(n, default_score=result.score * 0.9))

    expanded.sort(key=lambda x: x.score, reverse=True)
    logger.info(f"📚 Retrieved {len(base_results)} + {len(expanded) - len(base_results)} neighbors")
    return expanded


def retrieve_for_summary(
    doc_id: str,
    api_key: Optional[str] = None,
) -> list[RetrievalResult]:
    """Retrieve ALL chunks for a document, ordered by position. For summarization."""
    client = get_milvus_client(api_key=api_key)
    chunks = client.get_chunks_by_doc_id(doc_id)

    results = [_raw_to_result(c, default_score=1.0) for c in chunks]
    # Already sorted by position from get_chunks_by_doc_id
    logger.info(f"📚 Retrieved all {len(results)} chunks for doc_id={doc_id}")
    return results


def retrieve_for_comparison(
    concept_a: str,
    concept_b: str,
    top_k_per: int = 8,
    api_key: Optional[str] = None,
    metrics=None,
) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
    """Dual retrieval for comparison: one search per concept."""
    results_a = retrieve_documents(concept_a, top_k=top_k_per, api_key=api_key, metrics=metrics)
    results_b = retrieve_documents(concept_b, top_k=top_k_per, api_key=api_key, metrics=metrics)

    logger.info(f"📚 Comparison: {len(results_a)} for A, {len(results_b)} for B")
    return results_a, results_b


def retrieve_broad(
    query: str,
    api_key: Optional[str] = None,
    metrics=None,
) -> list[RetrievalResult]:
    """Broad retrieval for synthesis: high top-k with diversity."""
    return retrieve_with_neighbors(
        query, top_k=config.broad_top_k, api_key=api_key, metrics=metrics
    )


def retrieve_for_multiple_queries(
    queries: list[str],
    top_k_per_query: int = 5,
    api_key: Optional[str] = None,
    metrics=None,
) -> list[RetrievalResult]:
    """Retrieve documents for multiple queries and deduplicate."""
    all_results = []
    seen_contents = set()

    for query in queries:
        results = retrieve_documents(query, top_k=top_k_per_query, api_key=api_key, metrics=metrics)
        for result in results:
            content_key = result.content[:100]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                all_results.append(result)

    all_results.sort(key=lambda x: x.score, reverse=True)
    return all_results


# ------------------------------------------------------------------
# Context Building
# ------------------------------------------------------------------

def build_context(
    results: list[RetrievalResult],
    max_length: int = None,
) -> tuple[str, list[Citation]]:
    """Build context string from retrieval results with citations."""
    max_length = max_length or config.max_context_chars
    context_parts = []
    citations = []
    current_length = 0

    for i, result in enumerate(results, 1):
        source_ref = f"[Source {i}: {result.source_file}"
        if result.page_number:
            source_ref += f", Page {result.page_number}"
        if result.section:
            source_ref += f", Section: {result.section}"
        source_ref += "]"

        entry = f"{source_ref}\n{result.content}\n"
        if current_length + len(entry) > max_length:
            break

        context_parts.append(entry)
        current_length += len(entry)

        citations.append(
            Citation(
                source_file=result.source_file,
                page_number=result.page_number,
                section=result.section,
                excerpt=(
                    result.content[:100] + "..."
                    if len(result.content) > 100
                    else result.content
                ),
            )
        )

    context = "\n".join(context_parts)
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
