"""Ask the chatbot from the terminal (no browser needed).

    python -m scripts.ask "What is the waiting period for Acko health insurance?"

Several questions in one command share one conversation, so follow-ups can be tested too:
    python -m scripts.ask "What is the waiting period?" "And for pre-existing diseases?"
"""
from __future__ import annotations

import sys
import time
import uuid

from acko import rag

DEFAULT_QUESTIONS = ["What is the waiting period for Acko health insurance?"]


def main() -> int:
    questions = sys.argv[1:] or DEFAULT_QUESTIONS
    thread_id = str(uuid.uuid4())
    try:
        started = time.perf_counter()
        rag.warm_up()           # the web app does this once when it opens, so it is shown separately here
        print(f"Knowledge base and Gemini library loaded in {time.perf_counter() - started:.1f} s (one time)")
        for question in questions:
            result = rag.ask(question, thread_id)
            print(f"\nQuestion : {question}")
            note = f"   |   Gemini problem: {result['error']}" if result["error"] else ""
            print(f"Intent   : {result['intent']}   |   {result['seconds']:.1f} s{note}")
            print("Answer   :", result["answer"])
            for number, source in enumerate(result["sources"], 1):      # the passages Gemini was given
                print(f"   source {number}: {source['source']}, page {source['page']}")
    except rag.IndexMissing as exc:
        print("FAIL ", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
