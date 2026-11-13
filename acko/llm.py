"""Gemini helper: one place that builds the chat model and cleans up what comes back."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from acko import config


class LLMNotConfigured(RuntimeError):
    """Raised when no Gemini API key can be found."""


@lru_cache(maxsize=8)
def get_llm(vision: bool = False, temperature: float = 0.2, max_tokens: int = 1500):
    """Return a LangChain Gemini chat model (cached, so it is not rebuilt for every question).

    vision=True uses GEMINI_VISION_MODEL (by default the same model as GEMINI_MODEL).
    """
    if not config.GEMINI_API_KEY:
        raise LLMNotConfigured(
            "GEMINI_API_KEY is missing. Set it as a Windows environment variable or in the .env file."
        )
    # Imported here so scripts that never call Gemini start faster.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.GEMINI_VISION_MODEL if vision else config.GEMINI_MODEL,
        api_key=config.GEMINI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=2,       # LangChain retries temporary errors (rate limit, 503) with a back-off
        timeout=60,
    )


def to_text(message) -> str:
    """Return the plain text of a model reply.

    Newer Gemini models can return a list of content blocks (text, thinking, ...) instead
    of one string, so we always go through this helper instead of using message.content.
    """
    text = getattr(message, "text", None)
    # Normally .text is a plain string. Very old langchain-core versions had text() as a method.
    if text is not None and not isinstance(text, str) and callable(text):
        text = text()
    if isinstance(text, str) and text.strip():
        return text.strip()
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return str(content).strip()


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences and chatter)."""
    cleaned = re.sub(r"```(?:json)?", "", text or "", flags=re.IGNORECASE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object found in model reply: {text[:200]!r}")
    return json.loads(cleaned[start:end + 1])


def friendly_error(exc: Exception) -> str:
    """Turn an API exception into a message a customer or student can act on."""
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, LLMNotConfigured):
        return msg
    if "429" in msg or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
        return "The free Gemini quota is used up for the moment. Wait a minute and try again."
    if "api key" in low or "api_key" in low or "401" in msg or "403" in msg or "permission_denied" in low:
        return "Gemini rejected the API key. Check the GEMINI_API_KEY value."
    if "404" in msg or "not found" in low or "is not supported" in low:
        return ("Gemini does not know the model name in GEMINI_MODEL. "
                "Run `python -m scripts.check_gemini` to see the names your key can use.")
    if "503" in msg or "unavailable" in low or "overloaded" in low or "timeout" in low or "timed out" in low:
        return "Gemini is busy right now. Try again in a few seconds."
    return f"Gemini error ({type(exc).__name__}): {msg[:200]}"
