"""Module 1 page: the policy chatbot (Streamlit chat screen on top of acko/rag.py)."""
from __future__ import annotations

import uuid

import streamlit as st

from acko import db, quote_ui, rag

EXAMPLES = [
    "Is zero depreciation included in my base plan?",
    "What is the waiting period for Acko health insurance?",
    "How do I file a claim for my car?",
    "What add-ons are available for bike insurance?",
]
NEXT_STEP = {   # intent -> (page file, button text, message while that module is not built yet)
    "quotation": ("views/quote.py", "Open Get a Quote", "The quote calculator is added in Module 2."),
    "claims": ("views/claims.py", "Open File a Claim", "The claims page is added in Module 3."),
}

user = st.session_state["user"]
st.title("Acko Policy Assistant")
st.caption("Ask about Acko motor and health policies. Answers come from Acko's policy documents.")


@st.cache_resource(show_spinner="Loading the policy knowledge base (only the first time) ...")
def load_knowledge_base() -> int:
    return rag.warm_up()


try:
    load_knowledge_base()
except rag.IndexMissing as exc:
    st.error(str(exc))
    st.stop()

if "chat" not in st.session_state:
    st.session_state["chat"] = []
    st.session_state["thread_id"] = str(uuid.uuid4())      # one thread = one conversation with memory


def show_sources(sources: list[dict]) -> None:
    if sources:
        with st.expander(f"Sources ({len(sources)} passages from the policy documents)"):
            for source in sources:
                title = f"**{source['source']}**, page {source['page']}"
                st.markdown(title + (f" - {source['heading']}" if source["heading"] else ""))
                st.caption(source["text"][:400] + ("..." if len(source["text"]) > 400 else ""))


def show_next_step(intent: str, key: str, prefill: dict | None = None) -> None:
    """Under a quotation or claims answer: a link to the page that does the job."""
    if intent in NEXT_STEP:
        page_file, label, not_ready = NEXT_STEP[intent]
        if page_file not in st.session_state.get("pages", []):
            st.info(not_ready)
        elif prefill:       # the question already named vehicle details: open the form with them filled in
            st.caption(f"I picked this up from your question: {prefill['text']}. The form opens with it filled in.")
            if st.button(label, key=key, icon=":material/arrow_forward:"):
                st.session_state["quote_prefill"] = prefill
                st.switch_page(page_file)
        else:
            st.page_link(page_file, label=label)


with st.sidebar:
    if st.button("New chat", icon=":material/add_comment:"):
        st.session_state["chat"] = []
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.rerun()

if not st.session_state["chat"]:
    st.write("Try one of these questions, or type your own:")
    for example in EXAMPLES:
        if st.button(example):
            st.session_state["queued"] = example
            st.rerun()

for number, message in enumerate(st.session_state["chat"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(f"Intent: {message['intent']}  |  answered in {message['seconds']:.1f} s")
            show_sources(message["sources"])
            show_next_step(message["intent"], f"next_step_{number}", message.get("prefill"))

question = st.chat_input("Ask about your policy ...") or st.session_state.pop("queued", None)
if question:
    st.session_state["chat"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the policy documents ..."):
            result = rag.ask(question, thread_id=st.session_state["thread_id"])
        st.markdown(result["answer"])
        if result["error"]:
            st.warning("The AI service had a problem, so the closest policy passages are shown instead.")
        st.caption(f"Intent: {result['intent']}  |  answered in {result['seconds']:.1f} s")
        show_sources(result["sources"])
        prefill = quote_ui.prefill_from_question(question) if result["intent"] == "quotation" else None
        show_next_step(result["intent"], f"next_step_{len(st.session_state['chat'])}", prefill)
    st.session_state["chat"].append({"role": "assistant", "content": result["answer"], "intent": result["intent"],
                                     "seconds": result["seconds"], "sources": result["sources"], "prefill": prefill})
    try:    # every question and answer is saved for the manager (chat_logs table)
        used = "; ".join(f"{s['source']} p.{s['page']}" for s in result["sources"][:3])
        db.save_chat_log(user["id"], result["intent"], question, result["answer"], used)
    except Exception as exc:
        st.warning(f"This chat could not be saved to the database ({type(exc).__name__}).")
