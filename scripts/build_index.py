"""Step 4: build the ChromaDB knowledge base from the policy PDFs.

Run from the project folder:   python -m scripts.build_index
The first run downloads the embedding model once (about 80 MB). Safe to run again: it rebuilds from scratch.
"""
from __future__ import annotations

import time
from collections import Counter

from acko import config, rag


def main() -> int:
    started = time.perf_counter()
    print("Reading the PDFs in", config.PDF_DIR)
    try:
        chunks, duplicates = rag.load_chunks()
    except FileNotFoundError as exc:
        print("FAIL ", exc)
        return 1
    per_file = Counter(c["source"] for c in chunks)
    for name, count in sorted(per_file.items()):
        print(f"   {count:>4} chunks   {name}")
    if duplicates:
        print(f"   ({duplicates} repeated passages were skipped: the FAQ PDF repeats some of its own pages)")
    if len(per_file) < 3:
        print("WARN  The brief asks for at least 3 policy PDFs; run: python -m scripts.import_data")
    print("Embedding the chunks into ChromaDB (the first run also downloads the model) ...")
    total, questions = rag.build_index(chunks)
    question = "How soon must I report a stolen car to the police?"
    top = rag.search(question, k=1)[0]
    print(f"OK    ChromaDB now holds {total} chunks from {len(per_file)} PDFs, plus {questions} FAQ questions "
          f"({time.perf_counter() - started:.0f} s)")
    print(f"      Test question : {question}")
    print(f"      Best match    : {top['source']}, page {top['page']}  ({top['heading'] or 'no chapter title'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
