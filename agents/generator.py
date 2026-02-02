"""
Generator Agent

Generates responses using LLM with retrieved context.
Enforces citation of sources in responses.
"""
import time
from dataclasses import dataclass
from typing import Optional

from config import config
from agents.retriever import Citation, format_citations
from logger import get_logger

logger = get_logger("Generator")


def call_with_retry(func, max_retries=3, base_delay=1.0):
    """Call a function with exponential backoff retry for 503 errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if '503' in error_str or '429' in error_str or 'temporarily' in error_str or 'overloaded' in error_str:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"⚠️ API error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise last_error


@dataclass
class GeneratedResponse:
    """A generated response with citations."""
    answer: str
    citations: list[Citation]
    full_response: str  # Answer + formatted citations


def get_llm_client():
    """Get the LLM client based on configuration.
    
    Uses google.genai (the SDK pattern from Google ADK) instead of 
    the older google.generativeai module.
    """
    if config.llm_provider == "ollama":
        from ollama import Client
        return Client(), "ollama"
    else:
        # Use google.genai Client (same as ADK uses internally)
        from google import genai
        
        client = genai.Client(api_key=config.google_api_key)
        return client, "genai"


def generate_response(
    query: str,
    context: str,
    citations: list[Citation],
    system_prompt: Optional[str] = None
) -> GeneratedResponse:
    """
    Generate a response using LLM with retrieved context.
    
    Args:
        query: The user's question
        context: Retrieved document context
        citations: List of citations for the context
        system_prompt: Optional custom system prompt
    
    Returns:
        GeneratedResponse with answer and citations
    """
    if system_prompt is None:
        system_prompt = """You are a helpful manufacturing knowledge assistant. 
Your role is to answer questions accurately based on the provided context from company documents.

IMPORTANT RULES:
1. Only answer based on the information in the provided context
2. If the context doesn't contain the answer, say so clearly
3. Reference the source numbers [Source 1], [Source 2], etc. when citing information
4. Be concise but complete in your answers
5. For safety-related questions, always emphasize following proper procedures
6. For multi-part questions: Answer what you CAN find, then clearly state what information is NOT available"""

    full_prompt = f"""{system_prompt}

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {query}

Provide a helpful answer based on the context above. Reference sources when appropriate."""

    client, client_type = get_llm_client()
    
    if client_type == "ollama":
        response = client.chat(
            model=config.ollama_model,
            messages=[{"role": "user", "content": full_prompt}]
        )
        answer = response['message']['content']
    else:
        # Use google.genai Client API with retry for 503 errors
        def make_call():
            return client.models.generate_content(
                model=config.gemini_model,
                contents=full_prompt
            )
        response = call_with_retry(make_call)
        answer = response.text
    
    # Format full response with citations
    formatted_citations = format_citations(citations)
    full_response = answer + formatted_citations
    
    return GeneratedResponse(
        answer=answer,
        citations=citations,
        full_response=full_response
    )


def generate_direct_response(query: str) -> GeneratedResponse:
    """
    Generate a response without document context.
    Used for queries that don't need retrieval.
    """
    prompt = f"""You are a helpful assistant. Answer this question concisely.
If this is a question that requires specific knowledge from documents, 
indicate that you would need to search the document database.

Question: {query}"""

    client, client_type = get_llm_client()
    
    if client_type == "ollama":
        response = client.chat(
            model=config.ollama_model,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response['message']['content']
    else:
        # Use google.genai Client API with retry for 503 errors
        def make_call():
            return client.models.generate_content(
                model=config.gemini_model,
                contents=prompt
            )
        response = call_with_retry(make_call)
        answer = response.text
    
    return GeneratedResponse(
        answer=answer,
        citations=[],
        full_response=answer
    )


def generate_clarification_request(query: str) -> str:
    """Generate a request for clarification when query is unclear."""
    return f"""I'd be happy to help, but I need a bit more information to give you an accurate answer.

Could you please clarify:
- Which specific equipment or process are you asking about?
- What specific information are you looking for?

For example, you might ask:
- "What is the maintenance procedure for Pump A?"
- "What are the safety requirements for the hydraulic system?"
- "When is the next maintenance due for equipment ID PUMP-001?"
"""
