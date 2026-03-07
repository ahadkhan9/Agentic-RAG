"""
Router Agent

Classifies user intent and routes queries to appropriate handlers.
6 intents: retrieval, summary, comparison, synthesis, direct, clarify.
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
    SUMMARY = "summary"
    COMPARISON = "comparison"
    SYNTHESIS = "synthesis"
    DIRECT = "direct"
    CLARIFY = "clarify"


@dataclass
class RoutingResult:
    """Result of query routing."""
    intent: QueryIntent
    sub_queries: list[str]
    reasoning: str
    concepts: list[str]  # For comparison: [concept_a, concept_b]


def classify_intent(
    query: str,
    api_key: Optional[str] = None,
    metrics=None,
    has_documents: bool = True,
) -> RoutingResult:
    """Classify the user's query intent using Gemini.

    Args:
        query: The user's question
        api_key: Gemini API key
        metrics: Optional SessionMetrics
        has_documents: Whether any documents have been uploaded
    """
    doc_context = "Documents have been uploaded and are available for search." if has_documents else "No documents have been uploaded yet."

    prompt = f"""Analyze this user query for a document Q&A system.

Query: "{query}"

Context: {doc_context}

Classification rules:
1. "retrieval" — Specific questions that need to find particular information in documents. THIS IS THE DEFAULT for any factual question.
2. "summary" — User wants an overview, summary, or general description of a document or its contents. Triggers: "summarize", "overview", "what is this about", "main points", "give me a summary", "what does the document say".
3. "comparison" — User wants to compare two or more concepts, sections, or ideas found in documents. Triggers: "compare", "difference between", "X vs Y", "how does X differ from Y".
4. "synthesis" — User wants to understand relationships, connections, or how concepts relate to each other across documents. Triggers: "relate", "connect", "link", "how does X affect Y", "what's the relationship between".
5. "direct" — ONLY for greetings, meta-questions about the system itself, or trivial general knowledge clearly unrelated to documents.
6. "clarify" — Query is too vague or unclear to process.

Respond in this exact JSON format:
{{
    "intent": "retrieval" | "summary" | "comparison" | "synthesis" | "direct" | "clarify",
    "sub_queries": ["sub-question 1", "sub-question 2"],
    "concepts": ["concept_a", "concept_b"],
    "reasoning": "Brief explanation"
}}

Rules for fields:
- sub_queries: Only fill if there are 2+ distinct sub-questions. Otherwise empty list.
- concepts: Only fill for "comparison" intent. List the 2 things being compared. Otherwise empty list.
"""

    result_text = call_gemini(prompt, api_key=api_key, metrics=metrics)

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
            concepts=result.get("concepts", []),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return RoutingResult(
            intent=QueryIntent.RETRIEVAL,
            sub_queries=[],
            reasoning="Defaulting to retrieval mode",
            concepts=[],
        )


def decompose_query(
    query: str, api_key: Optional[str] = None, metrics=None
) -> list[str]:
    """Decompose a complex query into simpler sub-queries."""
    prompt = f"""Break down this complex question into simpler, focused sub-questions.

Question: "{query}"

Rules:
- Each sub-question should focus on ONE thing
- Keep sub-questions clear and specific
- Preserve important context
- Return 2-4 sub-questions

Respond with a JSON array of sub-questions only:
["sub-question 1", "sub-question 2", ...]
"""

    result_text = call_gemini(prompt, api_key=api_key, metrics=metrics)

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
    query: str,
    api_key: Optional[str] = None,
    metrics=None,
    has_documents: bool = True,
) -> tuple[QueryIntent, list[str], list[str]]:
    """Route a query based on its intent.

    Returns:
        Tuple of (intent, queries_to_process, concepts)
    """
    logger.debug(f"Routing query: '{query[:50]}...'")
    routing = classify_intent(query, api_key=api_key, metrics=metrics, has_documents=has_documents)
    logger.info(f"🎯 Classified as: {routing.intent.name} | Reason: {routing.reasoning}")

    queries = [query]
    if routing.sub_queries and len(routing.sub_queries) > 1:
        queries = routing.sub_queries
        logger.info(f"📋 Decomposed into {len(queries)} sub-queries")

    return routing.intent, queries, routing.concepts
