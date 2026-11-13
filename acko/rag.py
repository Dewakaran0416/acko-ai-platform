"""Module 1 engine: policy PDFs -> small chunks -> ChromaDB -> Gemini answer, run by a LangGraph agent.

Read this file from top to bottom: it is also the journey of one customer question.
  1. load_chunks()     PDF pages are cleaned and cut into small pieces (chunks)
  2. build_index()     every chunk becomes a vector and is stored in ChromaDB
  3. search()          a question finds its best chunks (meaning search + keyword search, merged)
  4. detect_intent()   the question is sorted into policy_qa / quotation / claims
  5. the LangGraph     classify -> retrieve -> answer, and the conversation is remembered
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Annotated, TypedDict

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from acko import config
from acko.llm import friendly_error, get_llm, to_text

CHUNK_SIZE = 800          # characters per chunk (about 130 words)
CHUNK_OVERLAP = 100       # characters shared by neighbours, so a sentence is not cut in half
COLLECTION = "acko_policies"
TOP_K = 5                 # chunks sent to Gemini (the brief asks for the 3-5 best)
CANDIDATES = 20           # each search method proposes this many chunks before they are merged


class IndexMissing(RuntimeError):
    """The ChromaDB knowledge base has not been built yet."""


# ----------------------------------------------------------------------------- 1. PDFs -> chunks
# Typographic characters in the PDFs (non-breaking space, en dash, em dash, minus, plus-minus) -> plain ASCII.
_PLAIN = str.maketrans({0xA0: " ", 0x2013: "-", 0x2014: "-", 0x2212: "-", 0xB1: "+/-"})


def normalize(text: str) -> str:
    """Lower case, plain dashes, single spaces: lets us compare texts that differ only in layout."""
    return " ".join(text.translate(_PLAIN).lower().split())


def _clean_pages(pages: list) -> list:
    """Drop the header and footer lines that repeat on many pages, and bare page numbers."""
    seen: Counter = Counter()
    for page in pages:
        seen.update({line.strip() for line in page.page_content.splitlines() if line.strip()})
    limit = max(3, int(len(pages) * 0.25))
    repeated = {line for line, count in seen.items() if count >= limit}
    for page in pages:
        keep = []
        for line in page.page_content.splitlines():
            line = line.strip()
            if line and line not in repeated and not re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", line, re.I):
                keep.append(line)
        page.page_content = "\n".join(keep)
    return pages


HEADING = re.compile(r"^(?:CHAPTER|SECTION|PART|APPENDIX)\b[^\n]*$", re.M)
QUESTIONS = [re.compile(r"Q\s+[A-N]\d{1,2}\.\s+(.{10,200}?\?)"),
             re.compile(r"\bQ\s+((?:What|How|Can|Is|Does|Do|Why|When|Which|Who|Are|Will|Should)[^?]{8,200}\?)")]


def _headings(text: str) -> list[tuple[int, str]]:
    """Chapter and section titles on a page (they are written in capitals) with their position."""
    return [(m.start(), m.group(0).strip()) for m in HEADING.finditer(text) if m.group(0).strip().isupper()]


def questions_in(text: str) -> list[str]:
    """The FAQ questions written inside a chunk ('Q D1. How do I register a claim ...?')."""
    found: list[str] = []
    for pattern in QUESTIONS:
        found += pattern.findall(text)
    return found


def search_text(chunk: dict) -> str:
    """The text that is turned into a vector: the chunk, with the title of its chapter in front."""
    heading = chunk.get("heading", "")
    if heading and not chunk["text"].upper().startswith(heading.upper()[:25]):
        return f"{heading}. {chunk['text']}"
    return chunk["text"]


def load_chunks() -> tuple[list[dict], int]:
    """Read every PDF in data/policy_pdfs. Returns (chunks, number of duplicate passages that were skipped)."""
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {config.PDF_DIR}. Run: python -m scripts.import_data")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                              add_start_index=True)
    chunks: list[dict] = []
    seen: set[str] = set()
    duplicates = 0
    for pdf in pdfs:
        pages = _clean_pages(PyPDFLoader(str(pdf)).load())
        counter: Counter = Counter()
        heading = ""                                            # the chapter we are currently inside
        for page in pages:
            titles = _headings(page.page_content)
            for piece in splitter.split_documents([page]):
                text = " ".join(piece.page_content.split())
                if len(text) < 40:                              # skip crumbs such as a lone heading
                    continue
                if normalize(text) in seen:                     # the FAQ PDF repeats whole pages: keep each passage once
                    duplicates += 1
                    continue
                seen.add(normalize(text))
                start = piece.metadata.get("start_index", 0)
                chapter = heading
                for position, title in titles:                  # the last title above the chunk is its chapter
                    if position <= start:
                        chapter = title
                page_no = int(piece.metadata.get("page", 0)) + 1    # PyPDFLoader counts pages from 0
                counter[page_no] += 1
                chunks.append({"id": f"{pdf.stem}-p{page_no:03d}-{counter[page_no]}", "text": text,
                               "heading": chapter, "source": pdf.name, "page": page_no})
            if titles:
                heading = titles[-1][1]
    return chunks, duplicates


# ----------------------------------------------------------------------------- 2. ChromaDB
def _client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def build_index(chunks: list[dict] | None = None, progress=print) -> tuple[int, int]:
    """(Re)build the ChromaDB knowledge base. Returns (number of passages, number of FAQ questions stored)."""
    chunks = chunks if chunks is not None else load_chunks()[0]
    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:       # first run: there is nothing to delete
        pass
    # The embedding model is all-MiniLM-L6-v2 (the well-known sentence-transformers model) run through
    # ONNX; ChromaDB downloads it once (about 80 MB) into your user folder.
    embedder = ONNXMiniLM_L6_V2()
    collection = client.create_collection(COLLECTION, embedding_function=embedder, metadata={"hnsw:space": "cosine"})
    entries = []        # (id, text to turn into a vector, text to store, extra info)
    questions = 0
    for chunk in chunks:
        info = {"source": chunk["source"], "page": chunk["page"], "heading": chunk["heading"], "kind": "passage"}
        entries.append((chunk["id"], search_text(chunk), chunk["text"], info))
        # FAQ passages are also searchable by the question they answer: short questions match short customer
        # questions much better than a whole paragraph does.
        for number, question in enumerate(questions_in(chunk["text"])):
            entries.append((f"{chunk['id']}#q{number}", question, question,
                            {**info, "kind": "question", "chunk_id": chunk["id"]}))
            questions += 1
    for start in range(0, len(entries), 64):
        batch = entries[start:start + 64]
        collection.add(ids=[e[0] for e in batch], embeddings=embedder([e[1] for e in batch]),
                       documents=[e[2] for e in batch], metadatas=[e[3] for e in batch])
        progress(f"   embedded {min(start + 64, len(entries))} / {len(entries)} texts")
    reset_cache()
    return len(chunks), questions


# ----------------------------------------------------------------------------- 3. search
STOP_WORDS = set("""a an the of to in on at by for from with and or but if is are was were be been am do does did
i me my we our you your it its this that these those can could will would should may might have has had not no
what which who whom how when where why there their them they as so than then also any about into over under""".split())

# The PDFs use abbreviations (NCB, IDV, PED ...) while customers write the long form, and keyword search only
# matches exact words, so the keyword side also searches for the other spelling.
SYNONYMS = [("no claim bonus", "ncb"), ("own damage", "od"), ("third party", "tp"), ("insured declared value", "idv"),
            ("zero depreciation", "zero dep"), ("pre-existing disease", "ped"), ("pre-existing diseases", "ped"),
            ("roadside assistance", "rsa"), ("return to invoice", "rti"), ("personal accident", "pa cover"),
            ("non-traceable certificate", "ntc"), ("co-payment", "co-pay"), ("copay", "co-pay")]


def expand_for_keywords(question: str) -> str:
    low = question.lower()
    extra = []
    for long_form, short_form in SYNONYMS:
        if long_form in low and short_form not in low:
            extra.append(short_form)
        if re.search(r"\b" + re.escape(short_form) + r"\b", low) and long_form not in low:
            extra.append(long_form)
    return question + " " + " ".join(extra)


def _stem(word: str) -> str:
    """Fold a plural into its singular ('periods' -> 'period') so both spellings match."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _words(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9%]+", text.lower()) if w not in STOP_WORDS]


class KeywordIndex:
    """BM25: the classic keyword score. Exact words such as '24 hours', 'IDV' or 'maternity' count a lot."""

    def __init__(self, texts: list[str]):
        self.docs = [Counter(_words(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.average = sum(self.lengths) / max(len(self.docs), 1)
        in_docs: Counter = Counter()
        for d in self.docs:
            in_docs.update(d.keys())
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in in_docs.items()}

    def top(self, query: str, n: int) -> list[int]:
        terms = set(_words(query))
        scores = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.average))
            scores.append(score)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:n] if scores[i] > 0]


_cache: dict = {}


def reset_cache() -> None:
    _cache.clear()


def _store() -> dict:
    """Open the knowledge base once and keep it (the embedding model and keyword index are slow to build)."""
    if not _cache:
        try:
            collection = _client().get_collection(COLLECTION, embedding_function=ONNXMiniLM_L6_V2())
        except Exception as exc:
            raise IndexMissing("The policy knowledge base is not built yet. Run:  python -m scripts.build_index") from exc
        data = collection.get(include=["documents", "metadatas"])
        chunks = [{"id": i, "text": d, "source": m["source"], "page": m["page"], "heading": m.get("heading", "")}
                  for i, d, m in zip(data["ids"], data["documents"], data["metadatas"]) if m.get("kind") == "passage"]
        _cache.update(collection=collection, chunks=chunks, position={c["id"]: n for n, c in enumerate(chunks)},
                      keywords=KeywordIndex([search_text(c) for c in chunks]))
    return _cache


def warm_up() -> int:
    """Load everything and embed one test sentence, so the first real question is not slow."""
    store = _store()
    store["collection"].query(query_texts=["warm up"], n_results=1)
    try:
        get_llm(temperature=0.2, max_tokens=1500)       # loads the Gemini library now, not during the first question
    except Exception:
        pass                                            # no key yet: ask() shows a friendly message later
    return len(store["chunks"])


def continuations(chunks: list[dict], how_many: int = 3) -> list[dict]:
    """A long answer can be cut between two chunks (an FAQ answer that ends '(5) Bariatric' | 'Surgery - 36 months').
    So Gemini also receives the chunk that follows each of the best chunks. They are not shown as sources."""
    store = _store()
    known = {c["id"] for c in chunks}
    more: list[dict] = []
    for chunk in chunks[:how_many]:
        stem, _, number = chunk["id"].rpartition("-")           # 'Acko_Insurance_FAQs-p008' and '5'
        position = store["position"].get(f"{stem}-{int(number) + 1}")
        if position is not None and store["chunks"][position]["id"] not in known:
            known.add(store["chunks"][position]["id"])
            more.append({**store["chunks"][position], "after": chunk["id"]})
    return more


def search(question: str, k: int = TOP_K, mode: str = "hybrid") -> list[dict]:
    """Return the k best chunks. mode: 'dense' (meaning), 'keyword' (BM25) or 'hybrid' (both, merged)."""
    store = _store()
    ranked: dict[str, float] = {}
    if mode in ("dense", "hybrid"):
        found = store["collection"].query(query_texts=[question], n_results=CANDIDATES * 2)
        order: list[str] = []
        for hit_id, info in zip(found["ids"][0], found["metadatas"][0]):
            chunk_id = info.get("chunk_id") or hit_id          # a matched FAQ question points to its passage
            if chunk_id not in order:
                order.append(chunk_id)
        for rank, chunk_id in enumerate(order[:CANDIDATES]):
            ranked[chunk_id] = ranked.get(chunk_id, 0.0) + 1.0 / (60 + rank + 1)      # Reciprocal Rank Fusion
    if mode in ("keyword", "hybrid"):
        for rank, index in enumerate(store["keywords"].top(expand_for_keywords(question), CANDIDATES)):
            chunk_id = store["chunks"][index]["id"]
            ranked[chunk_id] = ranked.get(chunk_id, 0.0) + 1.0 / (60 + rank + 1)
    results = []
    for chunk_id, score in sorted(ranked.items(), key=lambda item: item[1], reverse=True)[:k]:
        chunk = dict(store["chunks"][store["position"][chunk_id]])
        chunk["score"] = round(score, 4)
        results.append(chunk)
    return results


# ----------------------------------------------------------------------------- 4. intent
HEALTH = re.compile(r"\b(health|hospital|hospitali[sz]ation|medical|cashless|surgery|maternity|opd)\b")
SMALL_TALK = re.compile(r"^\s*(hi|hello|hey|hii+|good (morning|afternoon|evening)|thanks|thank you|thx|bye|ok|okay)\b[\s.!?]*$")
MAKES = (r"(maruti|suzuki|honda|hyundai|tata|mahindra|toyota|kia|renault|nissan|skoda|volkswagen|ford|mg|jeep|bmw|audi|"
         r"mercedes|bajaj|hero|tvs|yamaha|royal enfield|ktm|ather|swift|city|creta|i20|activa|splendor|pulsar|classic)")
VEHICLE = r"(car|bike|scooter|scooty|motorcycle|two[- ]wheeler|four[- ]wheeler|vehicle|suv|sedan|hatchback|" + MAKES + r")"
CLAIM_WORD = re.compile(r"\bclaims?\b")
NOT_CLAIMS = re.compile(r"\b(ncb|bonus|claim[- ]free|claims? history)\b")      # No Claim Bonus questions are policy questions
UPLOAD_PHOTO = re.compile(r"\bupload\b.{0,25}\b(photo|photos|picture|pictures|image|images)\b")
ACCIDENT = re.compile(r"\bmy (new |old )?" + VEHICLE + r"\b.{0,60}\b(accident|crash|crashed|damaged|dent|dented|hit|collided|"
                      r"flood|flooded|stolen|burnt|fire|broke|broken)")
QUOTE_NOW = re.compile(
    r"\b(quote|quotation|quotations)\b"
    r"|\b(premium|price|cost|how much)\b.{0,40}\b(my|a|an|new|used|\d{4})\b.{0,30}\b" + VEHICLE + r"\b"
    r"|\b(insure|insuring)\b.{0,30}\b(my|a|an|new)\b")


def detect_intent(question: str) -> str:
    """Sort a question into policy_qa, quotation or claims with simple, explainable rules.

    No AI call is needed here, so routing is instant and free. Health questions always stay policy_qa,
    because the quote and claims modules are for motor (car and bike) insurance. Any question about a claim
    is answered from the policy documents AND points the customer to the Claims page.
    """
    text = question.lower()
    if HEALTH.search(text):
        return "policy_qa"
    if ACCIDENT.search(text) or UPLOAD_PHOTO.search(text) or (CLAIM_WORD.search(text) and not NOT_CLAIMS.search(text)):
        return "claims"
    if QUOTE_NOW.search(text):
        return "quotation"
    return "policy_qa"


def is_small_talk(question: str) -> bool:
    return bool(SMALL_TALK.match(question.lower()))


# ----------------------------------------------------------------------------- 5. answer + LangGraph
SYSTEM_PROMPT = """You are Acko's friendly insurance assistant for customers in India.
Answer ONLY from the policy extracts you are given.
- Use simple words and short sentences: at most 6 sentences or 5 short bullet points.
- Give exact numbers, limits, days and percentages whenever the extracts contain them.
- If the extracts do not contain the answer, say you could not find it in the policy documents and suggest
  contacting Acko support. Never guess or invent amounts, limits or waiting periods.
- An extract marked "the text that follows extract [n]" is the continuation of extract [n]: read the two together.
- The app shows the sources separately: do not add a source list and do not write extract numbers like [1]."""

QUOTE_REPLY = ("I can give you an instant premium estimate. It only needs your vehicle details "
               "(type, make, model, year, city, IDV and No Claim Bonus). Open **Get a Quote** in the menu on the left.")
SMALL_TALK_REPLY = ("Hello! I can answer questions about Acko's motor and health policies, give you a premium "
                    "estimate, or help you file a vehicle claim. What would you like to know?")


class ChatState(TypedDict, total=False):
    messages: Annotated[list, add_messages]    # the whole conversation, kept per thread by MemorySaver
    question: str
    intent: str
    small_talk: bool
    chunks: list[dict]
    more: list[dict]                           # passages that continue the best chunks (sent to Gemini only)
    answer: str
    error: str | None


def _search_text(question: str, history: list) -> str:
    """A short follow-up ('and for bikes?') is searched together with the previous question."""
    previous = [m.content for m in history if isinstance(m, HumanMessage)]
    return f"{previous[-1]} {question}" if previous and len(question.split()) < 5 else question


def _prompt(question: str, chunks: list[dict], history: list, more: list[dict] | None = None) -> str:
    parts = [f"[{i}] ({c['source']}, page {c['page']}; {c['heading'] or 'no chapter title'})\n{c['text']}"
             for i, c in enumerate(chunks, 1)]
    number = {c["id"]: i for i, c in enumerate(chunks, 1)}
    for c in more or []:
        parts.append(f"[{len(parts) + 1}] (the text that follows extract [{number[c['after']]}])\n{c['text']}")
    extracts = "\n\n".join(parts)
    talk = "\n".join(f"{'Customer' if isinstance(m, HumanMessage) else 'Assistant'}: {str(m.content)[:400]}"
                     for m in history[-6:])
    return f"Policy extracts:\n{extracts}\n\nConversation so far:\n{talk or '(none)'}\n\nCustomer question: {question}"


def _fallback_answer(chunks: list[dict], reason: str) -> str:
    """Gemini is unreachable: still show the customer the best passages instead of an empty chat."""
    lines = [f"I could not reach the AI service just now ({reason}). The closest passages in the policy documents are:"]
    for c in chunks[:3]:
        lines.append(f"- {c['text'][:350].rstrip()}... ({c['source']}, page {c['page']})")
    return "\n\n".join(lines)


def classify_node(state: ChatState) -> dict:
    question = state["question"]
    # Fields of the previous turn are reset here, otherwise the saved memory would show old sources.
    return {"intent": detect_intent(question), "small_talk": is_small_talk(question),
            "chunks": [], "more": [], "answer": "", "error": None}


def after_classify(state: ChatState) -> str:
    if state["small_talk"]:
        return "small_talk"
    return "quote_reply" if state["intent"] == "quotation" else "retrieve"


def small_talk_node(state: ChatState) -> dict:
    return {"answer": SMALL_TALK_REPLY, "messages": [HumanMessage(state["question"]), AIMessage(SMALL_TALK_REPLY)]}


def quote_node(state: ChatState) -> dict:
    return {"answer": QUOTE_REPLY, "messages": [HumanMessage(state["question"]), AIMessage(QUOTE_REPLY)]}


def retrieve_node(state: ChatState) -> dict:
    found = search(_search_text(state["question"], state.get("messages", [])))
    return {"chunks": found, "more": continuations(found)}


def answer_node(state: ChatState) -> dict:
    question, chunks, history = state["question"], state["chunks"], state.get("messages", [])
    error = None
    try:
        reply = get_llm(temperature=0.2, max_tokens=1500).invoke(
            [("system", SYSTEM_PROMPT), ("human", _prompt(question, chunks, history, state.get("more")))])
        answer = to_text(reply)
        if not answer:
            raise ValueError("Gemini returned an empty answer")
    except Exception as exc:
        error = friendly_error(exc)
        answer = _fallback_answer(chunks, error)
    return {"answer": answer, "error": error, "messages": [HumanMessage(question), AIMessage(answer)]}


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("classify", classify_node)
    graph.add_node("small_talk", small_talk_node)
    graph.add_node("quote_reply", quote_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", after_classify, ["small_talk", "quote_reply", "retrieve"])
    graph.add_edge("retrieve", "answer")
    for last in ("small_talk", "quote_reply", "answer"):
        graph.add_edge(last, END)
    return graph.compile(checkpointer=MemorySaver())      # MemorySaver = the conversation memory


_agent = None


def ask(question: str, thread_id: str = "default") -> dict:
    """Run one question through the agent. Questions with the same thread_id share one conversation."""
    global _agent
    if _agent is None:
        _agent = build_graph()
    started = time.perf_counter()
    state = _agent.invoke({"question": question.strip()}, config={"configurable": {"thread_id": thread_id}})
    return {"answer": state["answer"], "intent": state["intent"], "error": state.get("error"),
            "sources": state.get("chunks", []), "seconds": round(time.perf_counter() - started, 2)}
