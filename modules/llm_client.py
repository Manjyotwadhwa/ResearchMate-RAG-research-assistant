"""Groq LLM construction and one shared, failure-tolerant call wrapper.

Every feature (Q&A, comparison, summary, gaps, LLM-judge evaluation) calls the
model through ``call_llm`` so API errors are translated into friendly messages
in exactly one place.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from config.settings import Settings


class LLMError(Exception):
    """Raised for any LLM problem. The message is safe to show to users."""


def get_llm(settings: Settings) -> BaseChatModel:
    """Create the Groq chat model from settings (key comes from the environment)."""
    if not settings.groq_api_key:
        raise LLMError(
            "GROQ_API_KEY is not set. Add it to your .env file (local) or Streamlit secrets (cloud)."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,  # per-request timeout
        max_retries=settings.llm_max_retries,  # SDK retries transient errors with backoff
    )


def _friendly_error(exc: Exception) -> str:
    """Map provider exceptions to short, actionable messages (matched by class name
    so we do not depend on provider-specific imports)."""
    name = type(exc).__name__
    text = str(exc).lower()
    if name == "AuthenticationError" or "invalid api key" in text or "401" in text:
        return "The Groq API key was rejected. Check GROQ_API_KEY."
    if name == "RateLimitError" or "rate limit" in text or "429" in text:
        return "Groq rate limit reached. Wait a minute and try again, or reduce the number of papers."
    if "Timeout" in name or "timed out" in text:
        return "The language model timed out. Please try again."
    if name == "APIConnectionError" or "connection" in text:
        return "Could not reach the Groq API. Check your internet connection."
    if name == "BadRequestError" or ("model" in text and "not" in text and "found" in text):
        return (
            "The request was rejected by Groq (the model name may be wrong or retired, or the "
            "input is too long). Check GROQ_MODEL or select fewer papers."
        )
    return f"The language model call failed ({name}). Please try again."


def call_llm(llm: BaseChatModel, prompt: ChatPromptTemplate, variables: dict) -> str:
    """Fill the prompt, call the model, return plain text. Raises LLMError on failure."""
    try:
        messages = prompt.format_messages(**variables)
        response = llm.invoke(messages)
    except Exception as exc:
        raise LLMError(_friendly_error(exc)) from exc

    content = response.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    content = (content or "").strip()
    if not content:
        raise LLMError("The language model returned an empty response. Please try again.")
    return content
