"""
Shared Utilities

Common functions used across agents — LLM client, retry logic, API key management.
"""
import time
import re
from typing import Optional

from google import genai

from config import config
from logger import get_logger

logger = get_logger("Utils")


# ---------------------------------------------------------------------------
# API Key Validation & Sanitization
# ---------------------------------------------------------------------------

_API_KEY_PATTERN = re.compile(r"^AIza[A-Za-z0-9_-]{35}$")


def validate_api_key(api_key: str) -> bool:
    """Validate that a string looks like a Gemini API key.

    Gemini API keys follow the pattern: AIza[35 alphanumeric/dash/underscore chars]
    """
    if not api_key or not isinstance(api_key, str):
        return False
    return bool(_API_KEY_PATTERN.match(api_key.strip()))


def sanitize_api_key(api_key: str) -> str:
    """Strip whitespace and validate format."""
    cleaned = api_key.strip()
    if not validate_api_key(cleaned):
        raise ValueError("Invalid Gemini API key format. Keys start with 'AIza' and are 39 characters.")
    return cleaned


# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """Create a Gemini client using the provided or default API key.

    Priority:
    1. Explicitly passed api_key (per-session, from user input)
    2. Server-level config.google_api_key (from .env)

    Raises ValueError if no key is available.
    """
    key = api_key or config.google_api_key
    if not key:
        raise ValueError(
            "No Gemini API key provided. Please enter your API key on the landing page."
        )
    return genai.Client(api_key=key)


# ---------------------------------------------------------------------------
# Retry Logic
# ---------------------------------------------------------------------------

def call_with_retry(func, max_retries: int = 3, base_delay: float = 1.0):
    """Call a function with exponential backoff retry for transient API errors.

    Retries on: 503 (overloaded), 429 (rate limit), temporary errors.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            retryable = any(
                code in error_str
                for code in ("503", "429", "temporarily", "overloaded", "deadline")
            )
            if retryable:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"⚠️ API error (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay}s: {str(e)[:100]}"
                )
                time.sleep(delay)
            else:
                raise
    raise last_error


# ---------------------------------------------------------------------------
# LLM Call Helper
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, api_key: Optional[str] = None) -> str:
    """Send a prompt to Gemini and return the response text.

    Handles client creation, retry, and error extraction in one place.
    """
    client = get_gemini_client(api_key)

    def make_call():
        return client.models.generate_content(
            model=config.gemini_model,
            contents=prompt,
        )

    response = call_with_retry(make_call)
    return response.text
