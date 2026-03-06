"""
Router Agent

Classifies user intent and routes queries to appropriate handlers.
Supports query decomposition for complex multi-part questions.
Gemini-only — uses shared utils for LLM calls.
"""
import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from config import config
from logger import get_logger
from utils import call_gemini

logger = get_logger("Router")


class QueryIntent(str, Enum):
    """Types of user query intents."""
    RETRIEVAL = "retrieval"
    DIRECT = "direct"
    MULTI_PART = "multi_part"
    CLARIFY = "clarify"


@dataclass
class RoutingResult:
    """Result of query routing."""
    intent: QueryIntent
    sub_queries: list[str]
    reasoning: str


def classify_intent(query: str, api_key: Optional[str] = None) -> RoutingResult:
    """Classify the user's query intent using Gemini.

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

    result_text = call_gemini(prompt, api_key=api_key)

    # Parse JSON response
    try:
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())

        return RoutingResult(
            intent=QueryIntent(result.get("intent", "retrieval")),
            sub_queries=result.get("sub_queries", []),
            reasoning=result.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return RoutingResult(
            intent=QueryIntent.RETRIEVAL,
            sub_queries=[],
            reasoning="Defaulting to retrieval mode",
        )


def decompose_query(query: str, api_key: Optional[str] = None) -> list[str]:
    """Decompose a complex query into simpler sub-queries.

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

    result_text = call_gemini(prompt, api_key=api_key)

    try:
        if "[" in result_text:
            start = result_text.index("[")
            end = result_text.rindex("]") + 1
            result_text = result_text[start:end]

        sub_queries = json.loads(result_text)
        return sub_queries if sub_queries else [query]
    except (json.JSONDecodeError, ValueError):
        return [query]


def route_query(
    query: str, api_key: Optional[str] = None
) -> tuple[QueryIntent, list[str]]:
    """Route a query based on its intent.

    Returns:
        Tuple of (intent, queries_to_process)
    """
    logger.debug(f"Routing query: '{query[:50]}...'")
    routing = classify_intent(query, api_key=api_key)
    logger.info(f"🎯 Classified as: {routing.intent.name} | Reason: {routing.reasoning}")

    if routing.intent == QueryIntent.MULTI_PART:
        queries = routing.sub_queries if routing.sub_queries else decompose_query(query, api_key=api_key)
        logger.info(f"📋 Decomposed into {len(queries)} sub-queries")
        for i, q in enumerate(queries, 1):
            logger.debug(f"  [{i}] {q}")
        return routing.intent, queries

    return routing.intent, [query]
