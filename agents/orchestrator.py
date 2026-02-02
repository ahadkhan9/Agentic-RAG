"""
Agent Orchestrator

Coordinates the flow between Router, Retriever, and Generator agents.
This is the main entry point for processing user queries.
"""
from dataclasses import dataclass
from typing import Optional

from agents.router import route_query, QueryIntent
from agents.retriever import (
    retrieve_documents,
    retrieve_for_multiple_queries,
    build_context,
    Citation,
)
from agents.generator import (
    generate_response,
    generate_direct_response,
    generate_clarification_request,
    GeneratedResponse,
)
from logger import get_logger

# Initialize logger
logger = get_logger("Orchestrator")


@dataclass
class QueryResult:
    """Complete result of processing a user query."""
    query: str
    intent: QueryIntent
    response: str
    citations: list[Citation]
    sub_queries: Optional[list[str]] = None


def process_query(query: str, top_k: int = 5) -> QueryResult:
    """
    Process a user query through the agent pipeline.
    
    Flow:
    1. Router: Classify intent and decompose if needed
    2. Retriever: Search documents based on intent
    3. Generator: Create response with citations
    
    Args:
        query: The user's question
        top_k: Number of documents to retrieve
    
    Returns:
        QueryResult with response and citations
    """
    logger.info(f"📥 Processing query: '{query}'")
    
    # Step 1: Route the query
    logger.debug("Step 1: Routing query...")
    intent, queries_to_process = route_query(query)
    logger.info(f"🎯 Intent: {intent.name}")
    
    if queries_to_process and len(queries_to_process) > 1:
        logger.info(f"📋 Decomposed into {len(queries_to_process)} sub-queries")
        for i, q in enumerate(queries_to_process, 1):
            logger.debug(f"  Sub-query {i}: {q}")
    
    # Step 2: Handle based on intent
    if intent == QueryIntent.CLARIFY:
        logger.info("❓ Generating clarification request...")
        clarification = generate_clarification_request(query)
        return QueryResult(
            query=query,
            intent=intent,
            response=clarification,
            citations=[]
        )
    
    if intent == QueryIntent.DIRECT:
        logger.info("💡 Generating direct response (no retrieval needed)...")
        result = generate_direct_response(query)
        return QueryResult(
            query=query,
            intent=intent,
            response=result.full_response,
            citations=[]
        )
    
    # Step 3: Retrieve documents
    logger.debug("Step 2: Retrieving documents...")
    if intent == QueryIntent.MULTI_PART:
        logger.info(f"🔍 Multi-part retrieval for {len(queries_to_process)} queries...")
        retrieval_results = retrieve_for_multiple_queries(
            queries_to_process,
            top_k_per_query=5
        )
    else:
        logger.info(f"🔍 Single query retrieval (top_k={top_k})...")
        retrieval_results = retrieve_documents(query, top_k=top_k)
    
    # AGENTIC BEHAVIOR: Check if results are mostly templates - if so, refine query
    if retrieval_results:
        template_count = sum(1 for r in retrieval_results if 'template' in r.source_file.lower())
        if template_count >= len(retrieval_results) * 0.5:  # More than half are templates
            logger.info("🔄 Agentic refinement: Found mostly templates, searching for data files...")
            # Try again with more specific query targeting actual data
            refined_query = f"{query} in database specifications inventory"
            refined_results = retrieve_documents(refined_query, top_k=top_k)
            
            # Merge results, prioritizing non-templates
            seen = set(r.content[:100] for r in retrieval_results)
            for r in refined_results:
                if r.content[:100] not in seen:
                    retrieval_results.append(r)
                    seen.add(r.content[:100])
            
            # Re-sort by score
            retrieval_results.sort(key=lambda x: x.score, reverse=True)
            logger.info(f"📚 After refinement: {len(retrieval_results)} documents")
    
    logger.info(f"📚 Retrieved {len(retrieval_results)} documents")
    
    # Handle no results
    if not retrieval_results:
        logger.warning("⚠️ No relevant documents found")
        return QueryResult(
            query=query,
            intent=intent,
            response="I couldn't find any relevant information in the documents. "
                     "Please try rephrasing your question or check if the relevant documents have been uploaded.",
            citations=[],
            sub_queries=queries_to_process if intent == QueryIntent.MULTI_PART else None
        )
    
    # Step 4: Build context and generate response
    logger.debug("Step 3: Building context...")
    context, citations = build_context(retrieval_results)
    logger.debug(f"Context length: {len(context)} chars, {len(citations)} citations")
    
    logger.info("🤖 Generating response with LLM...")
    generated = generate_response(
        query=query,
        context=context,
        citations=citations
    )
    
    logger.info(f"✅ Response generated ({len(generated.full_response)} chars, {len(generated.citations)} citations)")
    
    return QueryResult(
        query=query,
        intent=intent,
        response=generated.full_response,
        citations=generated.citations,
        sub_queries=queries_to_process if intent == QueryIntent.MULTI_PART else None
    )


def ingest_document(file_path: str) -> dict:
    """
    Ingest a document into the vector database.
    
    Args:
        file_path: Path to the document file
    
    Returns:
        Dict with ingestion stats
    """
    from ingestion.loader import load_document
    from ingestion.chunker import chunk_documents
    from vectordb.milvus_client import get_milvus_client
    
    logger.info(f"📄 Ingesting document: {file_path}")
    
    # Load and chunk document
    logger.debug("Loading document...")
    doc_chunks = load_document(file_path)
    logger.debug(f"Loaded {len(doc_chunks)} document sections")
    
    logger.debug("Chunking document...")
    text_chunks = chunk_documents(doc_chunks)
    logger.info(f"📝 Created {len(text_chunks)} chunks")
    
    # Insert into Milvus
    logger.debug("Inserting into vector database...")
    client = get_milvus_client()
    num_inserted = client.insert_chunks(text_chunks)
    logger.info(f"✅ Inserted {num_inserted} chunks into Milvus")
    
    return {
        "file": file_path,
        "chunks_created": len(text_chunks),
        "chunks_inserted": num_inserted
    }
