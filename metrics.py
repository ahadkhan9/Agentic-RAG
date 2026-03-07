"""
Session-scoped Token Usage Metrics

Tracks per-session: embedding tokens, LLM tokens (input/output),
API call counts, and cost estimates.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionMetrics:
    """Per-session usage tracking. Stored in Streamlit session_state."""

    embed_tokens: int = 0
    embed_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_calls: int = 0
    query_count: int = 0
    docs_uploaded: int = 0

    def record_llm_call(
        self,
        response,
        *,
        input_chars: int = 0,
    ) -> None:
        """Record a Gemini LLM call from the response object."""
        self.llm_calls += 1
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.llm_input_tokens += getattr(usage, "prompt_token_count", 0)
            self.llm_output_tokens += getattr(usage, "candidates_token_count", 0)
        elif input_chars:
            # Fallback: estimate ~4 chars per token
            self.llm_input_tokens += input_chars // 4

    def record_embedding_call(
        self, text_count: int, char_count: int
    ) -> None:
        """Record an embedding API call."""
        self.embed_calls += 1
        self.embed_tokens += char_count // 4  # ~4 chars per token

    def record_query(self) -> None:
        self.query_count += 1

    def record_upload(self) -> None:
        self.docs_uploaded += 1

    @property
    def total_tokens(self) -> int:
        return self.embed_tokens + self.llm_input_tokens + self.llm_output_tokens

    @property
    def total_api_calls(self) -> int:
        return self.embed_calls + self.llm_calls

    def get_cost_estimate(self) -> dict:
        """Rough cost estimate based on Gemini pricing (free tier generous).

        Flash-lite pricing (as of 2025):
          Input:  $0.075 / 1M tokens
          Output: $0.30 / 1M tokens
          Embedding: negligible
        """
        input_cost = (self.llm_input_tokens / 1_000_000) * 0.075
        output_cost = (self.llm_output_tokens / 1_000_000) * 0.30
        return {
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(input_cost + output_cost, 6),
        }

    def get_summary(self) -> dict:
        """Full metrics summary for display."""
        cost = self.get_cost_estimate()
        return {
            "queries": self.query_count,
            "docs_uploaded": self.docs_uploaded,
            "llm_calls": self.llm_calls,
            "embed_calls": self.embed_calls,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "embed_tokens": self.embed_tokens,
            "total_tokens": self.total_tokens,
            "est_cost_usd": cost["total_cost_usd"],
        }
