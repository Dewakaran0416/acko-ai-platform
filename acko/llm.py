"""Gemini helper: one place that builds the chat model and cleans up what comes back."""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import FIRST_COMPLETED, Future, wait
from functools import lru_cache

from acko import config

TEXT_TIMEOUT = 20       # seconds one Gemini call may take; after that it is dropped and tried again
VISION_TIMEOUT = 45     # calls with a photo take longer
HEDGE_AFTER = 5.0       # if the first answer has not arrived after this many seconds, ask a second time (0 = never)
FENCE = chr(96) * 3     # three backtick characters (code fence); built here so this file has no backticks


class LLMNotConfigured(RuntimeError):
    """Raised when no Gemini API key can be found."""


def _start(call) -> Future:
    """Run call() on a background thread and return a Future that will hold its result (or its error)."""
    future: Future = Future()

    def run():
        try:
            future.set_result(call())
        except BaseException as exc:            # the caller gets it when it asks for the result
            future.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()   # daemon: a stuck call never keeps the program open
    return future


def _first_answer(futures):
    """Return the result of the first call that succeeds; if every call fails, raise the last error."""
    pending, error = set(futures), None
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            try:
                return future.result()
            except Exception as exc:            # this call failed, the other one may still answer
                error = exc
    raise error


class HedgedModel:
    """The Gemini chat model plus one safety net against stalled requests.

    Now and then a single Gemini request stalls for 20 seconds or more while the very next one answers in two.
    So when the first call has not answered after hedge_after seconds, the same question is sent a second time
    and whichever answer arrives first is used. A call that fails quickly (wrong key, quota used up) is NOT
    repeated. Everything except invoke() is passed straight through to the real model.
    """

    def __init__(self, model, hedge_after: float = 0):
        self._model = model
        self._hedge_after = hedge_after

    def invoke(self, messages, *args, **kwargs):
        def call():
            return self._model.invoke(messages, *args, **kwargs)

        if not self._hedge_after:
            return call()
        first = _start(call)
        done, _ = wait([first], timeout=self._hedge_after)
        if done:
            return first.result()               # answered in time, or failed quickly (raises here)
        return _first_answer([first, _start(call)])

    def __getattr__(self, name):                # only called for names this class does not have
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._model, name)


@lru_cache(maxsize=8)
def get_llm(vision: bool = False, temperature: float = 0.2, max_tokens: int = 1500):
    """Return the Gemini chat model (cached, so it is not rebuilt for every question).

    vision=True uses GEMINI_VISION_MODEL (by default the same model as GEMINI_MODEL).
    """
    if not config.GEMINI_API_KEY:
        raise LLMNotConfigured(
            "GEMINI_API_KEY is missing. Set it as a Windows environment variable or in the .env file."
        )
    # Imported here so scripts that never call Gemini start faster.
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(
        model=config.GEMINI_VISION_MODEL if vision else config.GEMINI_MODEL,
        api_key=config.GEMINI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=2,       # LangChain tries a temporary error (rate limit, 503, 504) one more time after a short wait
        timeout=VISION_TIMEOUT if vision else TEXT_TIMEOUT,
    )
    return HedgedModel(model, 0 if vision else HEDGE_AFTER)


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
    """Pull the first JSON object out of a model reply (handles code fences and chatter)."""
    cleaned = re.sub(FENCE + r"(?:json)?", "", text or "", flags=re.IGNORECASE).strip()
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
                "Run: python -m scripts.check_gemini to see the names your key can use.")
    if ("503" in msg or "504" in msg or "unavailable" in low or "overloaded" in low or "deadline" in low
            or "timeout" in low or "timed out" in low):
        return "Gemini is busy right now. Try again in a few seconds."
    return f"Gemini error ({type(exc).__name__}): {msg[:200]}"
