"""Step 2 check: is your Gemini key found, and does the model answer?

Run from the project folder:   python -m scripts.check_gemini
The key itself is never printed.
"""
from __future__ import annotations

import time

from acko import config
from acko.llm import friendly_error, get_llm, to_text


def list_usable_models() -> None:
    """Ask Google which models this key may use (helps when GEMINI_MODEL has a wrong name)."""
    try:
        from google import genai

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        names = [m.name.removeprefix("models/") for m in client.models.list()
                 if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])]
        flash = sorted(n for n in names if n.startswith("gemini") and ("flash" in n or "lite" in n))
        print("      Models your key can use (put one in GEMINI_MODEL):")
        for name in flash[:12]:
            print("        ", name)
    except Exception as exc:  # the key itself may be the problem, nothing more to try here
        print("      Could not list models:", friendly_error(exc))


def main() -> int:
    print(f"Model name in use : {config.GEMINI_MODEL}")
    if not config.GEMINI_API_KEY:
        print("FAIL  No Gemini key found. Set GEMINI_API_KEY (Windows environment variable or .env file).")
        return 1
    print("OK    A Gemini key was found (it is not printed).")

    started = time.perf_counter()
    try:
        reply = to_text(get_llm(temperature=0, max_tokens=200).invoke("Reply with the single word: ready"))
    except Exception as exc:  # wrong key, wrong model name, quota, no internet ...
        print("FAIL  Gemini did not answer:", friendly_error(exc))
        list_usable_models()
        return 1
    print(f"OK    Gemini answered {reply[:30]!r} in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
