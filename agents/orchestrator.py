"""
Agent Orchestrator

Coordinates the flow between Router, Retriever, Generator, and Doc Registry.
Routes to intent-specific retrieval and generation strategies.
All LLM/embedding calls route through the per-session API key.
"""
from dataclasses import dataclass
from typing import Optional

from agents.router import route_query, QueryIntent
from agents.retriever import (
    retrieve_documents,
    retrieve_with_neighbors,
    retrieve_for_summary,
    retrieve_for_comparison,
    retrieve_broad,
    retrieve_for_multiple_queries,
    build_context,
    Citation,
)
from agents.generator import (
    generate_response,
    generate_direct_response,
    generate_clarification_request,
    generate_summary,
    generate_comparison,
    generate_synthesis,
    GeneratedResponse,
)
from vectordb.doc_registry import (
    register_document,
    find_doc_for_query,
    list_documents as list_registered_docs,
)
from logger import get_logger

logger = get_logger("Orchestrator")


@dataclass
class QueryResult:
    """Complete result of processing a user query."""
    query: str
    intent: QueryIntent
    response: str
    citations: list[Citation]
    sub_queries: Optional[list[str]] = None


def process_query(
    query: str,
    top_k: int = 10,
    api_key: Optional[str] = None,
    metrics=None,
    chat_history: list[dict] = None,
) -> QueryResult:
    """Process a user query through the intent-aware agent pipeline.

    Flow:
    1. Router: Classify intent (6 types)
    2. Retriever: Intent-specific retrieval strategy
    3. Generator: Intent-specific generation strategy
    """
    logger.info(f"📥 Processing query: '{query}'")

    if metrics is not None:
        metrics.record_query()

    # Check if documents exist
    has_documents = bool(list_registered_docs())

    # Step 1: Route
    intent, queries, concepts = route_query(
        query, api_key=api_key, metrics=metrics, has_documents=has_documents,
    )
    logger.info(f"🎯 Intent: {intent.name}")

    # Step 2 + 3: Intent-specific retrieval + generation
    if intent == QueryIntent.CLARIFY:
        clarification = generate_clarification_request(query)
        return QueryResult(query=query, intent=intent, response=clarification, citations=[])

    if intent == QueryIntent.DIRECT:
        result = generate_direct_response(query, api_key=api_key, metrics=metrics)
        return QueryResult(query=query, intent=intent, response=result.full_response, citations=[])

    if intent == QueryIntent.SUMMARY:
        return _handle_summary(query, api_key=api_key, metrics=metrics)

    if intent == QueryIntent.COMPARISON:
        return _handle_comparison(query, concepts, api_key=api_key, metrics=metrics)

    if intent == QueryIntent.SYNTHESIS:
        return _handle_synthesis(query, api_key=api_key, metrics=metrics, chat_history=chat_history)

    # Default: RETRIEVAL with neighbor expansion
    return _handle_retrieval(
        query, queries, intent, top_k=top_k,
        api_key=api_key, metrics=metrics, chat_history=chat_history,
    )


def _handle_retrieval(
    query: str,
    queries: list[str],
    intent: QueryIntent,
    top_k: int = 10,
    api_key: Optional[str] = None,
    metrics=None,
    chat_history: list[dict] = None,
) -> QueryResult:
    """Standard retrieval with neighbor expansion."""
    if len(queries) > 1:
        logger.info(f"🔍 Multi-part retrieval for {len(queries)} queries...")
        retrieval_results = retrieve_for_multiple_queries(
            queries, top_k_per_query=5, api_key=api_key, metrics=metrics,
        )
    else:
        logger.info(f"🔍 Retrieval with neighbor expansion (top_k={top_k})...")
        retrieval_results = retrieve_with_neighbors(
            query, top_k=top_k, api_key=api_key, metrics=metrics,
        )

    if not retrieval_results:
        return _no_results(query, intent, queries)

    context, citations = build_context(retrieval_results)
    generated = generate_response(
        query=query, context=context, citations=citations,
        chat_history=chat_history, api_key=api_key, metrics=metrics,
    )

    return QueryResult(
        query=query, intent=intent, response=generated.full_response,
        citations=generated.citations,
        sub_queries=queries if len(queries) > 1 else None,
    )


def _handle_summary(
    query: str,
    api_key: Optional[str] = None,
    metrics=None,
) -> QueryResult:
    """Full-document summarization."""
    doc_meta = find_doc_for_query(query)

    if not doc_meta:
        # No specific doc found — try retrieval-based summary
        logger.info("📝 No specific doc identified, using broad retrieval for summary...")
        results = retrieve_broad(query, api_key=api_key, metrics=metrics)
        if not results:
            return _no_results(query, QueryIntent.SUMMARY)

        generated = generate_summary(results, query=query, api_key=api_key, metrics=metrics)
        return QueryResult(
            query=query, intent=QueryIntent.SUMMARY,
            response=generated.full_response, citations=generated.citations,
        )

    logger.info(f"📝 Summarizing document: {doc_meta.filename} (doc_id={doc_meta.doc_id})")
    results = retrieve_for_summary(doc_meta.doc_id, api_key=api_key)

    if not results:
        return _no_results(query, QueryIntent.SUMMARY)

    generated = generate_summary(
        results, doc_summary=doc_meta.summary, query=query,
        api_key=api_key, metrics=metrics,
    )

    return QueryResult(
        query=query, intent=QueryIntent.SUMMARY,
        response=generated.full_response, citations=generated.citations,
    )


def _handle_comparison(
    query: str,
    concepts: list[str],
    api_key: Optional[str] = None,
    metrics=None,
) -> QueryResult:
    """Dual-retrieval comparison."""
    if len(concepts) < 2:
        # Fallback: treat as retrieval
        logger.warning("⚠️ Comparison requires 2 concepts, falling back to retrieval")
        return _handle_retrieval(query, [query], QueryIntent.COMPARISON, api_key=api_key, metrics=metrics)

    concept_a, concept_b = concepts[0], concepts[1]
    logger.info(f"⚖️ Comparing: '{concept_a}' vs '{concept_b}'")

    results_a, results_b = retrieve_for_comparison(
        concept_a, concept_b, api_key=api_key, metrics=metrics,
    )

    if not results_a and not results_b:
        return _no_results(query, QueryIntent.COMPARISON)

    context_a, citations_a = build_context(results_a)
    context_b, citations_b = build_context(results_b)

    generated = generate_comparison(
        context_a=context_a, context_b=context_b,
        citations_a=citations_a, citations_b=citations_b,
        concept_a=concept_a, concept_b=concept_b,
        query=query, api_key=api_key, metrics=metrics,
    )

    return QueryResult(
        query=query, intent=QueryIntent.COMPARISON,
        response=generated.full_response, citations=generated.citations,
    )


def _handle_synthesis(
    query: str,
    api_key: Optional[str] = None,
    metrics=None,
    chat_history: list[dict] = None,
) -> QueryResult:
    """Broad retrieval for synthesis/connection-finding."""
    logger.info("🔗 Broad retrieval for synthesis...")
    results = retrieve_broad(query, api_key=api_key, metrics=metrics)

    if not results:
        return _no_results(query, QueryIntent.SYNTHESIS)

    context, citations = build_context(results)
    generated = generate_synthesis(
        context=context, citations=citations, query=query,
        chat_history=chat_history, api_key=api_key, metrics=metrics,
    )

    return QueryResult(
        query=query, intent=QueryIntent.SYNTHESIS,
        response=generated.full_response, citations=generated.citations,
    )


def _no_results(query: str, intent: QueryIntent, queries: list[str] = None) -> QueryResult:
    """Standard no-results response."""
    return QueryResult(
        query=query, intent=intent,
        response=(
            "I couldn't find any relevant information in the documents. "
            "Please try rephrasing your question or check if the relevant "
            "documents have been uploaded."
        ),
        citations=[],
        sub_queries=queries,
    )


# ------------------------------------------------------------------
# Document Ingestion (with registry)
# ------------------------------------------------------------------

def ingest_document(
    file_path: str,
    api_key: Optional[str] = None,
    metrics=None,
) -> dict:
    """Ingest a document: load → chunk → register → embed → store."""
    from ingestion.loader import load_document
    from ingestion.chunker import chunk_documents
    from vectordb.milvus_client import get_milvus_client

    logger.info(f"📄 Ingesting document: {file_path}")

    # Load and chunk
    doc_chunks = load_document(file_path)
    text_chunks = chunk_documents(doc_chunks)
    logger.info(f"📝 Created {len(text_chunks)} chunks")

    # Register document (generate summary)
    import os
    filename = os.path.basename(file_path)
    full_text = "\n\n".join(c.content for c in doc_chunks)

    doc_meta = register_document(
        filename=filename,
        full_text=full_text,
        chunk_count=len(text_chunks),
        api_key=api_key,
        metrics=metrics,
    )

    # Insert into Milvus with doc_id
    client = get_milvus_client(api_key=api_key)
    num_inserted = client.insert_chunks(text_chunks, doc_id=doc_meta.doc_id, metrics=metrics)
    logger.info(f"✅ Inserted {num_inserted} chunks into Milvus")

    if metrics is not None:
        metrics.record_upload()

    return {
        "file": file_path,
        "filename": filename,
        "doc_id": doc_meta.doc_id,
        "summary": doc_meta.summary,
        "topics": doc_meta.topics,
        "chunks_created": len(text_chunks),
        "chunks_inserted": num_inserted,
    }
