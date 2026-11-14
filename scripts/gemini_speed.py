"""Is Gemini slow right now, or is it something in the app?

Times real chatbot questions on Gemini with NO automatic retries, so a slow or failed call is visible.
Run from the project folder (the key is never printed; it takes 1 to 3 minutes):

    python -m scripts.gemini_speed                           your GEMINI_MODEL and gemini-3.5-flash-lite
    python -m scripts.gemini_speed gemini-3.5-flash-lite     only the model(s) you name
"""
from __future__ import annotations

import statistics
import sys
import time

from google import genai
from google.genai import types

from acko import config, rag

QUESTIONS = ["What is the waiting period for Acko health insurance?",
             "How do I file a claim for my car?",
             "What add-ons are available for bike insurance?"]
ROUNDS = 2              # every question is asked twice: 6 calls per model
LIMIT = 25              # seconds after which one call is given up
SLOW = 5                # the brief wants answers in under 5 seconds
OTHER_MODEL = "gemini-3.5-flash-lite"


def build_prompts() -> list[str]:
    """The same prompt the chatbot sends: the best passages plus the question."""
    prompts = []
    for question in QUESTIONS:
        chunks = rag.search(question)
        prompts.append(rag._prompt(question, chunks, [], rag.continuations(chunks)))
    return prompts


def is_missing(outcome: str) -> bool:
    """True when Gemini says this model name does not exist for the key."""
    return "NOT_FOUND" in outcome or outcome.startswith("ClientError: 404")


def time_one(client, model: str, prompt: str) -> tuple[float, str]:
    """Returns (seconds, outcome); the outcome is 'ok ...' or a short description of what went wrong."""
    started = time.perf_counter()
    try:
        reply = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=rag.SYSTEM_PROMPT, temperature=0.2,
                                               max_output_tokens=1500))
        if (reply.text or "").strip():
            usage = getattr(reply, "usage_metadata", None)
            thinking = getattr(usage, "thoughts_token_count", None) or 0
            written = getattr(usage, "candidates_token_count", None) or 0
            outcome = f"ok   (answer {written} tokens, thinking {thinking} tokens)"
        else:
            outcome = "empty answer"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {str(exc)[:110]}".replace("\n", " ")
    return time.perf_counter() - started, outcome


def test_model(client, model: str, prompts: list[str]) -> list[tuple[float, str]]:
    print(f"\n{model}   ({len(prompts) * ROUNDS} calls, no automatic retries, {LIMIT} s limit per call)")
    results = []
    for number in range(len(prompts) * ROUNDS):
        seconds, outcome = time_one(client, model, prompts[number % len(prompts)])
        results.append((seconds, outcome))
        print(f"  {number + 1}  {seconds:>5.1f} s   {outcome}")
        if is_missing(outcome):
            print("      This model name is not available for your key, so it is skipped.")
            break
        time.sleep(2)           # stay below the free rate limit
    return results


def summary_line(model: str, results: list[tuple[float, str]]) -> tuple[tuple, str]:
    answered = [s for s, outcome in results if outcome.startswith("ok")]
    slow = sum(1 for s, outcome in results if not outcome.startswith("ok") or s >= SLOW)
    median = statistics.median(answered) if answered else float("inf")
    slowest = max((s for s, _ in results), default=0.0)
    text = (f"  {model:<26} answered {len(answered)} of {len(results)}   slower than {SLOW} s or failed: {slow}   "
            f"median {median:.1f} s   slowest {slowest:.1f} s")
    return (len(results) - len(answered), slow, median), text


def main() -> int:
    if not config.GEMINI_API_KEY:
        print("FAIL  No Gemini key found. Set GEMINI_API_KEY (Windows environment variable or .env file).")
        return 1
    models = list(dict.fromkeys(sys.argv[1:] or [config.GEMINI_MODEL, OTHER_MODEL]))
    try:
        prompts = build_prompts()
    except rag.IndexMissing as exc:
        print("FAIL ", exc)
        return 1
    client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(timeout=LIMIT * 1000))
    ranked = []
    for model in models:
        results = test_model(client, model, prompts)
        if results and not is_missing(results[-1][1]):
            ranked.append((model, results))
    if not ranked:
        print("\nNo model could be tested.")
        return 1
    print("\nSummary")
    scored = []
    for model, results in ranked:
        score, line = summary_line(model, results)
        print(line)
        scored.append((score, model, results))
    best_score, best, best_results = min(scored, key=lambda item: item[0])
    if best == config.GEMINI_MODEL:
        print(f"\nBest: {best} (the one you already use).")
    else:
        print(f"\nBest: {best}. To use it, run both lines (the first is for this window, the second keeps it for later):")
        print(f'    $env:GEMINI_MODEL = "{best}"')
        print(f"    setx GEMINI_MODEL {best}")
    if best_score[1]:
        explain_problems(best_results)
    return 0


def explain_problems(results: list[tuple[float, str]]) -> None:
    """Says in plain words why the best model still had slow or failed calls."""
    outcomes = [outcome for _, outcome in results]
    refused = sum("429" in o or "RESOURCE_EXHAUSTED" in o for o in outcomes)
    busy = sum("503" in o or "UNAVAILABLE" in o for o in outcomes)
    stuck = sum("imeout" in o or "timed out" in o or "DEADLINE" in o or "504" in o for o in outcomes)
    slow = sum(1 for s, o in results if o.startswith("ok") and s >= SLOW)
    print("\nEven the best model had calls that were slow or failed:")
    if refused:
        print(f"  {refused} call(s) were refused with 'too many requests' (429). That is the free-tier limit; "
              "wait a minute and run the test again.")
    if busy:
        print(f"  {busy} call(s) got 'service unavailable' (503). Gemini was overloaded; that is not something in the app.")
    if stuck:
        print(f"  {stuck} call(s) did not finish within {LIMIT} s. Gemini was slow or busy; that is not something in the app.")
    if slow:
        print(f"  {slow} call(s) answered, but took {SLOW} s or more. The 'thinking' numbers above show how much of that "
              "was Gemini reasoning before it wrote the answer.")
    if not (refused or busy or stuck or slow):
        print("  Some calls came back empty or with an error; see the lines above.")


if __name__ == "__main__":
    raise SystemExit(main())
