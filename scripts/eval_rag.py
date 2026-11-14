"""Step 5: the retrieval test of the brief (20 questions), a larger extra check and an intent-routing test.

    python -m scripts.eval_rag             search test, no Gemini needed (about 1 minute)
    python -m scripts.eval_rag --answers   also asks Gemini the 20 questions and times the answers (about 2 minutes)

How a question is judged: a retrieved passage is RELEVANT when it contains the key facts of the answer.
data/eval/*.json lists, for every question, one or more fact patterns; a pattern is one or more phrases
that must all appear in the same passage. The result is also saved to reports/rag_eval.md (for your report).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import uuid

from acko import config, rag

EVAL_DIR = config.DATA_DIR / "eval"
REPORT = config.ROOT / "reports" / "rag_eval.md"
MODES = ["hybrid", "dense", "keyword"]
MODE_LABELS = {"hybrid": "hybrid (used by the chatbot)", "dense": "meaning search only", "keyword": "keyword search only"}
INTENT_EXAMPLES = [
    ("Is zero depreciation included in my base plan?", "policy_qa"),
    ("What is the waiting period for Acko health insurance?", "policy_qa"),
    ("What add-ons are available for bike insurance?", "policy_qa"),
    ("Does my bike insurance cover theft if I forgot to lock it?", "policy_qa"),
    ("How does the No Claim Bonus work?", "policy_qa"),
    ("What factors affect my premium?", "policy_qa"),
    ("How is the premium calculated?", "policy_qa"),
    ("How do I file a health insurance claim?", "policy_qa"),
    ("How much does bike insurance cost?", "policy_qa"),
    ("How much will insurance cost for my 2020 Honda City in Chennai?", "quotation"),
    ("I want a quote for my Hyundai i20", "quotation"),
    ("Can I get a premium estimate for my bike?", "quotation"),
    ("What would be the premium for my Royal Enfield Classic 350?", "quotation"),
    ("How do I file a claim for my car?", "claims"),
    ("I want to file a claim", "claims"),
    ("My car met with an accident yesterday", "claims"),
    ("My bike was damaged in the flood, I want to claim", "claims"),
    ("What is the status of my claim?", "claims"),
    ("Upload photos of my car damage", "claims"),
    ("How much will I get for my car damage claim?", "claims"),
]


def load(name: str) -> list[dict]:
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))


def is_relevant(item: dict, passage: str) -> bool:
    text = rag.normalize(passage)
    return any(all(phrase in text for phrase in pattern) for pattern in item["expected"])


def judge(item: dict, mode: str) -> dict:
    """Search the question and report at which place (1-5) the first relevant passage appears."""
    started = time.perf_counter()
    found = rag.search(item["question"], k=rag.TOP_K, mode=mode)
    seconds = time.perf_counter() - started
    for place, passage in enumerate(found, 1):
        if is_relevant(item, passage["text"]):
            return {"place": place, "passage": passage, "seconds": seconds, "top": found[0]}
    return {"place": None, "passage": None, "seconds": seconds, "top": found[0] if found else None}


def where(result: dict) -> str:
    passage = result["passage"] or result["top"]
    return f"{passage['source'].replace('Acko_', '').replace('.pdf', '')} p.{passage['page']}" if passage else "-"


def retrieval_test(items: list[dict], title: str, show_each: bool, lines: list[str]) -> dict:
    results = {mode: [judge(item, mode) for item in items] for mode in MODES}
    print(f"\n{title}: is a relevant passage among the top {rag.TOP_K} found?")
    lines += ["", f"## {title}", ""]
    if show_each:
        print("  #  found  place  question")
        lines += ["| # | Found | Place | Question | Passage |", "|---|---|---|---|---|"]
        for n, (item, result) in enumerate(zip(items, results["hybrid"]), 1):
            mark = "OK" if result["place"] else "MISS"
            print(f"{n:>3}  {mark:<5}  {result['place'] or '-':>3}    {item['question'][:75]}")
            lines.append(f"| {n} | {mark} | {result['place'] or '-'} | {item['question']} | {where(result)} |")
    lines += ["", "| Search method | Relevant passage in top 5 | Average search time |", "|---|---|---|"]
    summary = {}
    for mode in MODES:
        hits = sum(1 for r in results[mode] if r["place"])
        average = sum(r["seconds"] for r in results[mode]) / len(items)
        summary[mode] = hits
        text = f"{hits} / {len(items)}  ({100 * hits / len(items):.0f}%)"
        print(f"   {MODE_LABELS[mode]:<32} {text:<14} average {average * 1000:.0f} ms per search")
        lines.append(f"| {MODE_LABELS[mode]} | {text} | {average * 1000:.0f} ms |")
    misses = [(item, r) for item, r in zip(items, results["hybrid"]) if not r["place"]]
    if misses:
        lines += ["", "Questions where the chatbot's search did not find a relevant passage:", ""]
        print("   Not found:")
        for item, result in misses:
            top = result["top"]
            print(f"     - {item['question']}")
            print(f"       best passage it returned: {where(result)}")
            lines.append(f"- {item['question']} (best passage returned: {where(result)})")
    return summary


def intent_test(lines: list[str]) -> None:
    print("\nIntent routing test (policy_qa / quotation / claims):")
    wrong = [(q, want, rag.detect_intent(q)) for q, want in INTENT_EXAMPLES if rag.detect_intent(q) != want]
    right = len(INTENT_EXAMPLES) - len(wrong)
    print(f"   {right} / {len(INTENT_EXAMPLES)} questions routed correctly")
    lines += ["", "## Intent routing", "", f"{right} / {len(INTENT_EXAMPLES)} example questions were routed correctly.", ""]
    for question, want, got in wrong:
        print(f"     - {question!r}: expected {want}, got {got}")
        lines.append(f"- {question} (expected {want}, got {got})")


def answer_test(items: list[dict], lines: list[str]) -> None:
    print("\nAnswer test with Gemini (this takes about 2 minutes) ...")
    rag.warm_up()
    lines += ["", "## Answers from Gemini", "", "| # | Key fact in answer | Seconds | Question | Answer (first 160 characters) |",
              "|---|---|---|---|---|"]
    seconds_list, with_fact, failed = [], 0, 0          # seconds_list holds only the questions Gemini answered
    for n, item in enumerate(items, 1):
        result = rag.ask(item["question"], thread_id=str(uuid.uuid4()))
        if result["error"]:                              # Gemini gave no answer (the chat would show the closest passages)
            failed += 1
            print(f"{n:>3}  NONE   {result['seconds']:>4.1f} s   {item['question'][:70]}")
            print(f"       Gemini problem: {result['error'][:150]}")
            lines.append(f"| {n} | no answer from Gemini | {result['seconds']:.1f} | {item['question']} | "
                         f"{result['error'][:160].replace('|', '/')} |")
            if failed >= 3:
                print("\nGemini failed 3 times, so the test stops here. Wait a few minutes and run it again.")
                lines += ["", f"Stopped after question {n}: Gemini failed 3 times."]
                return
        else:
            answer = rag.normalize(result["answer"])
            has_fact = any(word.lower() in answer for word in item["answer_keywords"])
            with_fact += has_fact
            seconds_list.append(result["seconds"])
            print(f"{n:>3}  {'OK   ' if has_fact else 'CHECK'}  {result['seconds']:>4.1f} s   {item['question'][:70]}")
            lines.append(f"| {n} | {'yes' if has_fact else 'check'} | {result['seconds']:.1f} | {item['question']} | "
                         f"{result['answer'][:160].replace('|', '/').replace(chr(10), ' ')} |")
        time.sleep(4)               # stay well below the free Gemini rate limit (about 15 requests per minute)
    fast = sum(1 for s in seconds_list if s < 5)
    text = (f"Answers that contain the key fact: {with_fact} / {len(items)}.  Gemini gave no answer: {failed}.  "
            f"Answered in under 5 s: {fast} / {len(items)}")
    if seconds_list:
        text += (f" (median {statistics.median(seconds_list):.1f} s, average {sum(seconds_list) / len(seconds_list):.1f} s, "
                 f"slowest {max(seconds_list):.1f} s)")
    text += "."
    print("\n" + text + "\nRead the answers yourself too: the key-fact check is only a rough guide (rows marked CHECK).")
    lines += ["", text]


def main() -> int:
    lines = ["# Module 1 evaluation: policy chatbot", "", f"Created: {config.now_local():%Y-%m-%d %H:%M} ({config.APP_TIMEZONE})"]
    try:
        main_items = load("rag_questions.json")
        rag.warm_up()
        retrieval_test(main_items, f"MAIN TEST ({len(main_items)} questions from the brief and the policy PDFs)", True, lines)
        extra_items = load("rag_questions_extra.json")
        retrieval_test(extra_items, f"EXTRA CHECK ({len(extra_items)} more questions, written separately)", False, lines)
        intent_test(lines)
        if "--answers" in sys.argv:
            answer_test(main_items, lines)
    except rag.IndexMissing as exc:
        print("FAIL ", exc)
        return 1
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
