"""
Generator Agent

Generates responses using Gemini with retrieved context.
Enforces citation of sources in responses.
"""
from dataclasses import dataclass
from typing import Optional

from agents.retriever import Citation, format_citations
from logger import get_logger
from utils import call_gemini

logger = get_logger("Generator")


@dataclass
class GeneratedResponse:
    """A generated response with citations."""
    answer: str
    citations: list[Citation]
    full_response: str  # Answer + formatted citations


def generate_response(
    query: str,
    context: str,
    citations: list[Citation],
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
) -> GeneratedResponse:
    """Generate a response using Gemini with retrieved context.

    Args:
        query: The user's question
        context: Retrieved document context
        citations: List of citations for the context
        system_prompt: Optional custom system prompt
        api_key: Optional per-session Gemini API key

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

    answer = call_gemini(full_prompt, api_key=api_key)

    formatted_citations = format_citations(citations)
    full_response = answer + formatted_citations

    return GeneratedResponse(
        answer=answer,
        citations=citations,
        full_response=full_response,
    )


def generate_direct_response(
    query: str, api_key: Optional[str] = None
) -> GeneratedResponse:
    """Generate a response without document context.

    Used for queries that don't need retrieval (greetings, meta-questions).
    """
    prompt = f"""You are a helpful assistant. Answer this question concisely.
If this is a question that requires specific knowledge from documents,
indicate that you would need to search the document database.

Question: {query}"""

    answer = call_gemini(prompt, api_key=api_key)

    return GeneratedResponse(
        answer=answer,
        citations=[],
        full_response=answer,
    )


def generate_clarification_request(query: str) -> str:
    """Generate a request for clarification when query is unclear."""
    return """I'd be happy to help, but I need a bit more information to give you an accurate answer.

Could you please clarify:
- Which specific equipment or process are you asking about?
- What specific information are you looking for?

For example, you might ask:
- "What is the maintenance procedure for Pump A?"
- "What are the safety requirements for the hydraulic system?"
- "When is the next maintenance due for equipment ID PUMP-001?"
"""
