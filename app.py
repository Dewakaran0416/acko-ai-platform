"""Acko AI Platform: the one entry point of the web app.   Start it with:   streamlit run app.py

It shows the sign-in page first. After signing in, the menu on the left lists the pages that exist
(each module adds one file to the views folder) and that the signed-in role may use.
"""
from __future__ import annotations

import hmac
from pathlib import Path

import streamlit as st

from acko import config, db

st.set_page_config(page_title="Acko AI Platform", page_icon=":material/shield:", layout="wide")

ROOT = Path(__file__).parent
PAGES = [   # (file, menu title, icon, only for the manager?)
    ("views/chat.py", "Policy Chatbot", ":material/chat:", False),
    ("views/quote.py", "Get a Quote", ":material/calculate:", False),
    ("views/claims.py", "File a Claim", ":material/car_crash:", False),
    ("views/dashboard.py", "Dashboard", ":material/monitoring:", True),
    ("views/assistant.py", "Manager Assistant", ":material/smart_toy:", True),
]


@st.cache_resource
def prepare_database() -> bool:
    db.init_db()            # creates the four tables if they are missing (safe to repeat)
    return True


def sign_in(user, role: str) -> None:
    st.session_state["user"] = {"id": str(user.id), "name": user.name, "email": user.email, "role": role}
    st.rerun()


def login_page() -> None:
    st.title("Welcome to Acko")
    st.write("Sign in to chat with the policy assistant, get a quote or file a claim.")
    with st.form("customer_login"):
        name = st.text_input("Your name")
        email = st.text_input("Your e-mail")
        phone = st.text_input("Phone number (optional)")
        submitted = st.form_submit_button("Continue", type="primary")
    if submitted:
        if not name.strip() or "@" not in email:
            st.error("Please enter your name and a valid e-mail address.")
        else:
            user = db.get_or_create_customer(name, email, phone)
            if user.role != "customer":
                st.error("This e-mail belongs to the Acko team. Please use the manager sign-in below.")
            else:
                sign_in(user, "customer")
    if config.MANAGER_PASSWORD:         # the manager sign-in only exists when a password is set in .env
        with st.expander("Manager sign-in"):
            with st.form("manager_login"):
                password = st.text_input("Manager password", type="password")
                if st.form_submit_button("Sign in as manager"):
                    if hmac.compare_digest(password.encode(), config.MANAGER_PASSWORD.encode()):
                        sign_in(db.ensure_manager(), "manager")
                    else:
                        st.error("Wrong password.")


prepare_database()
user = st.session_state.get("user")
if user is None:
    page = st.navigation([st.Page(login_page, title="Sign in", icon=":material/login:")])
else:
    allowed = [(file, title, icon) for file, title, icon, manager_only in PAGES
               if (ROOT / file).exists() and (user["role"] == "manager" or not manager_only)]
    st.session_state["pages"] = [file for file, _, _ in allowed]
    page = st.navigation([st.Page(file, title=title, icon=icon) for file, title, icon in allowed])
    with st.sidebar:
        st.caption(f"Signed in as **{user['name']}** ({user['role']})")
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()
page.run()
