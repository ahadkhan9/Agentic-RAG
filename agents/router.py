"""
Router Agent

Classifies user intent and routes queries to appropriate handlers.
Supports query decomposition for complex multi-part questions.
"""
import json
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from config import config
from logger import get_logger

# Initialize logger
logger = get_logger("Router")


def call_with_retry(func, max_retries=3, base_delay=1.0):
    """Call a function with exponential backoff retry for 503 errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Retry on 503, 429 (rate limit), or temporary errors
            if '503' in error_str or '429' in error_str or 'temporarily' in error_str or 'overloaded' in error_str:
                delay = base_delay * (2 ** attempt)  # 1, 2, 4 seconds
                logger.warning(f"⚠️ API error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise  # Non-retryable error
    raise last_error  # All retries exhausted


class QueryIntent(str, Enum):
    """Types of user query intents."""
    RETRIEVAL = "retrieval"      # Needs document search
    DIRECT = "direct"            # Can answer without documents
    MULTI_PART = "multi_part"    # Complex query needing decomposition
    CLARIFY = "clarify"          # Query is unclear


@dataclass
class RoutingResult:
    """Result of query routing."""
    intent: QueryIntent
    sub_queries: list[str]
    reasoning: str


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


def classify_intent(query: str) -> RoutingResult:
    """
    Classify the user's query intent.
    
    Uses LLM to determine:
    - If retrieval is needed
    - If query should be decomposed
    - How to route the query
    """
    prompt = f"""Analyze this user query for a MANUFACTURING document Q&A system.

Query: "{query}"

IMPORTANT: This is a RAG system with uploaded manufacturing documents. Most queries should use "retrieval" to search documents.

Classification rules:
1. "retrieval" - Use for ANY question about equipment, procedures, specifications, safety, maintenance, SOPs, or manufacturing-related topics. THIS IS THE DEFAULT.
2. "direct" - ONLY for greetings, meta-questions about the system, or completely general knowledge unrelated to manufacturing.
3. "multi_part" - Complex questions with 2+ distinct sub-questions that need separate searches.
4. "clarify" - Query is too vague or unclear to process.

Respond in this exact JSON format:
{{
    "intent": "retrieval" | "direct" | "multi_part" | "clarify",
    "sub_queries": ["sub-question 1", "sub-question 2"],
    "reasoning": "Brief explanation of your classification"
}}

Only include sub_queries if intent is "multi_part". Otherwise, use empty list.
"""

    client, client_type = get_llm_client()
    
    if client_type == "ollama":
        response = client.chat(
            model=config.ollama_model,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response['message']['content']
    else:
        # Use google.genai Client API with retry for 503 errors
        def make_call():
            return client.models.generate_content(
                model=config.gemini_model,
                contents=prompt
            )
        response = call_with_retry(make_call)
        result_text = response.text
    
    # Parse JSON response
    try:
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result = json.loads(result_text.strip())
        
        return RoutingResult(
            intent=QueryIntent(result.get("intent", "retrieval")),
            sub_queries=result.get("sub_queries", []),
            reasoning=result.get("reasoning", "")
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        # Default to retrieval if parsing fails
        return RoutingResult(
            intent=QueryIntent.RETRIEVAL,
            sub_queries=[],
            reasoning="Defaulting to retrieval mode"
        )


def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex query into simpler sub-queries.
    
    Example:
        "What's the maintenance procedure for Pump A and when is it due?"
        -> ["What is the maintenance procedure for Pump A?",
            "When is the next maintenance due for Pump A?"]
    """
    prompt = f"""Break down this complex question into simpler, focused sub-questions.

Question: "{query}"

Rules:
- Each sub-question should focus on ONE thing
- Keep sub-questions clear and specific
- Preserve important context (equipment names, dates, etc.)
- Return 2-4 sub-questions

Respond with a JSON array of sub-questions only:
["sub-question 1", "sub-question 2", ...]
"""

    client, client_type = get_llm_client()
    
    if client_type == "ollama":
        response = client.chat(
            model=config.ollama_model,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response['message']['content']
    else:
        # Use google.genai Client API with retry for 503 errors
        def make_call():
            return client.models.generate_content(
                model=config.gemini_model,
                contents=prompt
            )
        response = call_with_retry(make_call)
        result_text = response.text
    
    try:
        # Extract JSON array
        if "[" in result_text:
            start = result_text.index("[")
            end = result_text.rindex("]") + 1
            result_text = result_text[start:end]
        
        sub_queries = json.loads(result_text)
        return sub_queries if sub_queries else [query]
    except (json.JSONDecodeError, ValueError):
        return [query]


def route_query(query: str) -> tuple[QueryIntent, list[str]]:
    """
    Route a query based on its intent.
    
    Returns:
        Tuple of (intent, queries_to_process)
        - For simple queries: ([original_query])
        - For multi-part: list of sub-queries
    """
    logger.debug(f"Routing query: '{query[:50]}...'")
    routing = classify_intent(query)
    logger.info(f"🎯 Classified as: {routing.intent.name} | Reason: {routing.reasoning}")
    
    if routing.intent == QueryIntent.MULTI_PART:
        # Use sub_queries from classification, or decompose
        queries = routing.sub_queries if routing.sub_queries else decompose_query(query)
        logger.info(f"📋 Decomposed into {len(queries)} sub-queries")
        for i, q in enumerate(queries, 1):
            logger.debug(f"  [{i}] {q}")
        return routing.intent, queries
    
    return routing.intent, [query]
