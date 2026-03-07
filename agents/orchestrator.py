"""
Agent Orchestrator

Coordinates Router, Retriever, Generator, and Doc Registry.
Supports both blocking and streaming response modes.
"""
from dataclasses import dataclass
from typing import Optional, Generator

from agents.router import route_query, QueryIntent
from agents.retriever import (
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
    stream_response,
    stream_direct_response,
    stream_summary,
    stream_comparison,
    stream_synthesis,
    format_citations,
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


@dataclass
class StreamContext:
    """Context needed alongside a stream for citations/intent."""
    intent: QueryIntent
    citations: list[Citation]
    sub_queries: Optional[list[str]] = None


def process_query(
    query: str, top_k: int = 10,
    api_key: Optional[str] = None, metrics=None,
    chat_history: list[dict] = None,
) -> QueryResult:
    """Process a query (blocking). Returns full result."""
    intent, queries, concepts, retrieval_results, context, citations = _resolve(
        query, top_k, api_key, metrics, chat_history,
    )

    if intent == QueryIntent.CLARIFY:
        return QueryResult(query=query, intent=intent, response=generate_clarification_request(query), citations=[])

    if intent == QueryIntent.DIRECT:
        r = generate_direct_response(query, api_key=api_key, metrics=metrics)
        return QueryResult(query=query, intent=intent, response=r.full_response, citations=[])

    if intent == QueryIntent.SUMMARY:
        doc_meta = find_doc_for_query(query)
        results = retrieve_for_summary(doc_meta.doc_id, api_key=api_key) if doc_meta else retrieval_results
        if not results:
            return _no_results(query, intent)
        r = generate_summary(results, doc_summary=getattr(doc_meta, 'summary', ''), query=query, api_key=api_key, metrics=metrics)
        return QueryResult(query=query, intent=intent, response=r.full_response, citations=r.citations)

    if intent == QueryIntent.COMPARISON:
        if len(concepts) >= 2:
            results_a, results_b = retrieve_for_comparison(concepts[0], concepts[1], api_key=api_key, metrics=metrics)
            ctx_a, cit_a = build_context(results_a)
            ctx_b, cit_b = build_context(results_b)
            r = generate_comparison(ctx_a, ctx_b, cit_a, cit_b, concepts[0], concepts[1], query, api_key=api_key, metrics=metrics)
            return QueryResult(query=query, intent=intent, response=r.full_response, citations=r.citations)

    if intent == QueryIntent.SYNTHESIS:
        r = generate_synthesis(context, citations, query, chat_history=chat_history, api_key=api_key, metrics=metrics)
        return QueryResult(query=query, intent=intent, response=r.full_response, citations=r.citations)

    # RETRIEVAL (default)
    if not retrieval_results:
        return _no_results(query, intent, queries)
    r = generate_response(query=query, context=context, citations=citations, chat_history=chat_history, api_key=api_key, metrics=metrics)
    return QueryResult(query=query, intent=intent, response=r.full_response, citations=r.citations, sub_queries=queries if len(queries) > 1 else None)


def process_query_stream(
    query: str, top_k: int = 10,
    api_key: Optional[str] = None, metrics=None,
    chat_history: list[dict] = None,
) -> tuple[StreamContext, Generator[str, None, None]]:
    """Process a query with streaming. Returns (context, text_generator).

    Usage:
        ctx, stream = process_query_stream(query, ...)
        for chunk in stream:
            print(chunk, end="")
    """
    intent, queries, concepts, retrieval_results, context, citations = _resolve(
        query, top_k, api_key, metrics, chat_history,
    )

    sctx = StreamContext(intent=intent, citations=citations, sub_queries=queries if len(queries) > 1 else None)

    if intent == QueryIntent.CLARIFY:
        def _clarify():
            yield generate_clarification_request(query)
        return sctx, _clarify()

    if intent == QueryIntent.DIRECT:
        return sctx, stream_direct_response(query, api_key=api_key, metrics=metrics)

    if intent == QueryIntent.SUMMARY:
        doc_meta = find_doc_for_query(query)
        results = retrieve_for_summary(doc_meta.doc_id, api_key=api_key) if doc_meta else retrieval_results
        if not results:
            return sctx, _yield_no_results()
        source_files = list(set(r.source_file for r in results))
        sctx.citations = [Citation(source_file=f, page_number=None, section=None, excerpt="Full document") for f in source_files]
        return sctx, stream_summary(results, doc_summary=getattr(doc_meta, 'summary', ''), query=query, api_key=api_key, metrics=metrics)

    if intent == QueryIntent.COMPARISON and len(concepts) >= 2:
        results_a, results_b = retrieve_for_comparison(concepts[0], concepts[1], api_key=api_key, metrics=metrics)
        ctx_a, cit_a = build_context(results_a)
        ctx_b, cit_b = build_context(results_b)
        sctx.citations = cit_a + cit_b
        return sctx, stream_comparison(ctx_a, ctx_b, concepts[0], concepts[1], query, api_key=api_key, metrics=metrics)

    if intent == QueryIntent.SYNTHESIS:
        return sctx, stream_synthesis(context, query, chat_history=chat_history, api_key=api_key, metrics=metrics)

    # RETRIEVAL
    if not retrieval_results:
        return sctx, _yield_no_results()
    return sctx, stream_response(query=query, context=context, citations=citations, chat_history=chat_history, api_key=api_key, metrics=metrics)


def _resolve(query, top_k, api_key, metrics, chat_history):
    """Shared: route + retrieve. Returns (intent, queries, concepts, results, context, citations)."""
    if metrics is not None:
        metrics.record_query()

    has_documents = bool(list_registered_docs())
    intent, queries, concepts = route_query(query, api_key=api_key, metrics=metrics, has_documents=has_documents)

    # Retrieve based on intent
    retrieval_results = []
    context = ""
    citations = []

    if intent in (QueryIntent.RETRIEVAL, QueryIntent.SYNTHESIS, QueryIntent.COMPARISON, QueryIntent.SUMMARY):
        if intent == QueryIntent.SYNTHESIS:
            retrieval_results = retrieve_broad(query, api_key=api_key, metrics=metrics)
        elif len(queries) > 1:
            retrieval_results = retrieve_for_multiple_queries(queries, top_k_per_query=5, api_key=api_key, metrics=metrics)
        else:
            retrieval_results = retrieve_with_neighbors(query, top_k=top_k, api_key=api_key, metrics=metrics)

        if retrieval_results:
            context, citations = build_context(retrieval_results)

    return intent, queries, concepts, retrieval_results, context, citations


def _yield_no_results():
    yield "I couldn't find relevant information in the documents. Please try rephrasing or check if the documents have been uploaded."


def _no_results(query, intent, queries=None):
    return QueryResult(
        query=query, intent=intent,
        response="I couldn't find relevant information in the documents. Please try rephrasing or check if the documents have been uploaded.",
        citations=[], sub_queries=queries,
    )


# ------------------------------------------------------------------
# Document Ingestion
# ------------------------------------------------------------------

def ingest_document(file_path: str, api_key: Optional[str] = None, metrics=None) -> dict:
    """Ingest a document: load -> chunk -> register -> embed -> store."""
    from ingestion.loader import load_document
    from ingestion.chunker import chunk_documents
    from vectordb.milvus_client import get_milvus_client
    import os

    doc_chunks = load_document(file_path)
    text_chunks = chunk_documents(doc_chunks)

    filename = os.path.basename(file_path)
    full_text = "\n\n".join(c.content for c in doc_chunks)

    doc_meta = register_document(
        filename=filename, full_text=full_text,
        chunk_count=len(text_chunks),
        api_key=api_key, metrics=metrics,
    )

    client = get_milvus_client(api_key=api_key)
    num_inserted = client.insert_chunks(text_chunks, doc_id=doc_meta.doc_id, metrics=metrics)

    if metrics is not None:
        metrics.record_upload()

    return {
        "file": file_path, "filename": filename,
        "doc_id": doc_meta.doc_id, "summary": doc_meta.summary,
        "topics": doc_meta.topics,
        "chunks_created": len(text_chunks), "chunks_inserted": num_inserted,
    }
