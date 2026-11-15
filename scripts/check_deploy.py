"""Deployment check: does everything the platform needs work on THIS computer or server?

    python -m scripts.check_deploy                          on your PC
    sudo docker exec acko python -m scripts.check_deploy    on the EC2 server (the app runs in the container "acko")

It checks the settings, the database (PostgreSQL on RDS or SQLite), the S3 bucket (writes a tiny test file, reads it
back and deletes it again), the model files on S3, the quote and claim models, the vector store and Gemini.
No password or key is ever printed. Exit code 0 = no FAIL line.
"""
from __future__ import annotations

import platform
import time
import uuid
from importlib import metadata

from acko import config, db, model_store, storage
from acko.fmt import inr

DB_HINTS = [
    ("password authentication failed", "The database password (DB_PASSWORD) is wrong."),
    ("timeout expired", "The database cannot be reached: check the firewall group acko-rds-sg and that the database status is 'available'."),
    ("timed out", "The database cannot be reached: check the firewall group acko-rds-sg and that the database status is 'available'."),
    ("could not translate host name", "DB_HOST is not a valid address."),
    ("connection refused", "Nothing answers at DB_HOST and DB_PORT: is the database still starting?"),
    ("no pg_hba.conf entry", "The database refuses this connection: set DB_SSLMODE=require."),
    ("does not exist", "The database name (DB_NAME) or a table does not exist on that server."),
]


class Warn(Exception):
    """This check cannot be done here. It is not an error."""


def explain(exc: Exception) -> str:
    if isinstance(exc, storage.StorageError):
        return str(exc)
    text = str(exc).strip().splitlines()[0][:230] if str(exc).strip() else ""
    hint = next((advice for needle, advice in DB_HINTS if needle in str(exc).lower()), "")
    return f"{type(exc).__name__}: {text}" + (f"  ->  {hint}" if hint else "")


# ---------------------------------------------------------------------------------------------- the checks
def check_settings() -> str:
    parts = ["Gemini key " + ("set" if config.GEMINI_API_KEY else "not set"),
             "manager password " + ("set" if config.MANAGER_PASSWORD else "not set"),
             (f"S3 bucket {config.S3_BUCKET} ({config.AWS_REGION})" if config.S3_BUCKET else "S3 off (files go to the folder uploads/)"),
             ("database: PostgreSQL" if db.get_engine().dialect.name == "postgresql" else "database: SQLite (this computer only)")]
    return "; ".join(parts)


def check_database() -> str:
    from sqlalchemy import text

    from scripts import init_db as init_db_script

    description = db.describe_database()
    db.init_db()
    init_db_script.write_test()
    engine = db.get_engine()
    extra = ""
    if engine.dialect.name == "postgresql":
        with engine.connect() as connection:
            version = " ".join(str(connection.execute(text("select version()")).scalar()).split()[:2])
            encrypted = connection.execute(text("select ssl from pg_stat_ssl where pid = pg_backend_pid()")).scalar()
        if not encrypted:
            raise Warn(f"{description} ({version}) works, but the connection is NOT encrypted. Set DB_SSLMODE=require.")
        extra = f" ({version}, encrypted connection)"
    return f"{description}{extra}; the 4 tables exist and a write test passed"


def check_s3() -> str:
    if storage.backend() != "s3":
        raise Warn("S3_BUCKET is not set, so claim files go to the folder uploads/ on this computer")
    line = storage.check()
    who = storage.identity()
    key = f"healthcheck/probe-{uuid.uuid4().hex[:12]}.txt"
    payload = b"acko deployment check"
    storage.put(key, payload, "text/plain")
    try:
        if storage.get(key) != payload:
            raise RuntimeError("The test file came back with different content.")
        encryption = storage.details(key)["encryption"]
    finally:
        storage.remove(key)
    if storage.exists(key):
        raise RuntimeError("The test file could not be deleted again.")
    return f"{line}; as {who or 'an unknown identity'}: wrote, read back and deleted a test file (stored encrypted: {encryption})"


def check_models_on_s3() -> str:
    if storage.backend() != "s3":
        raise Warn("S3 is not set up, so the models are only checked as local files (see the next checks)")
    items = model_store.status()
    same = [item for item in items if item.result == "same"]
    if len(same) != len(items):
        odd = ", ".join(f"{item.name} ({item.result})" for item in items if item.result != "same")
        raise Warn(f"{len(same)} of {len(items)} model files are identical on S3. Different: {odd}")
    return f"all {len(items)} model files in use are identical to the copies in {model_store.where()}"


def check_quote_models() -> str:
    from acko import quote

    started = time.perf_counter()
    engine = quote.QuoteEngine.load()
    said = []
    for kind in quote.KINDS:
        options = engine.options(kind)
        make = options["makes"][0]
        model = next(iter(options["models"][make]))
        fuel = options["models"][make][model][0]
        result = engine.quote(kind, make, model, fuel, 2021, options["cities"][0], 400_000, options["ncb_steps"][0])
        if not (result.premium > 0 and result.factors):
            raise RuntimeError(f"The {kind} quote came back empty.")
        said.append(f"{kind} {make} {model} {inr(result.premium)}")
    return f"{'; '.join(said)}; SHAP explanation built ({time.perf_counter() - started:.0f} s)"


def check_claim_models() -> str:
    from acko import claims

    engine = claims.ClaimEngine.load()
    said = []
    for kind, idv in (("car", 600_000), ("bike", 90_000)):
        estimate = engine.estimate(kind, "collision damage", 6, 2, idv)
        if not (0 <= estimate.probability <= 1 and estimate.amount > 0):
            raise RuntimeError(f"The {kind} claim estimate is not valid.")
        said.append(f"{kind} {inr(estimate.amount)}, approval {estimate.probability:.0%} ({estimate.status})")
    return "; ".join(said)


def check_vector_store() -> str:
    from acko import rag

    try:
        top = rag.search("How soon must I report a stolen car to the police?", k=1)[0]
    except Exception as exc:
        raise RuntimeError(f"The vector store could not be used ({type(exc).__name__}). Build it with: python -m scripts.build_index") from exc
    return f"ChromaDB answered from {top['source']}, page {top['page']} ({config.CHROMA_DIR.name})"


def check_gemini() -> str:
    from acko.llm import friendly_error, get_llm, to_text

    if not config.GEMINI_API_KEY:
        raise Warn("No Gemini key: the chatbot and the photo check will not work")
    started = time.perf_counter()
    try:
        reply = to_text(get_llm(temperature=0, max_tokens=200).invoke("Reply with the single word: ready"))
    except Exception as exc:
        raise RuntimeError(friendly_error(exc)) from exc
    return f"Gemini ({config.GEMINI_MODEL}) answered {reply[:20]!r} in {time.perf_counter() - started:.1f} s"


CHECKS = [("settings", check_settings), ("database", check_database), ("S3 bucket", check_s3),
          ("models on S3", check_models_on_s3), ("quote models", check_quote_models), ("claim models", check_claim_models),
          ("vector store", check_vector_store), ("Gemini", check_gemini)]


VERSION_LINES = (("scikit-learn", "pandas", "numpy", "scipy", "joblib", "shap"),
                 ("streamlit", "chromadb", "sqlalchemy", "boto3", "pillow"),
                 ("langchain-core", "langchain-community", "langchain-google-genai", "langgraph", "google-genai"))


def versions() -> str:
    """The versions that matter when the models trained on one computer run on another (compare PC and server)."""
    lines = [f"Python {platform.python_version()} on {platform.system()}"]
    for group in VERSION_LINES:
        found = []
        for name in group:
            try:
                found.append(f"{name} {metadata.version(name)}")
            except metadata.PackageNotFoundError:
                found.append(f"{name} MISSING")
        lines.append(" | ".join(found))
    return "\n".join(lines)


def main() -> int:
    print("Acko deployment check")
    print(versions())
    ok = warn = fail = 0
    for label, function in CHECKS:
        try:
            detail, mark = function(), "[ ok ]"
            ok += 1
        except Warn as exc:
            detail, mark = str(exc), "[warn]"
            warn += 1
        except Exception as exc:
            detail, mark = explain(exc), "[FAIL]"
            fail += 1
        print(f"{mark} {label:<13} {detail}")
    total = ok + warn + fail
    if fail:
        print(f"Result: {fail} FAIL, {warn} warning(s), {ok} ok (of {total} checks)")
    elif warn:
        print(f"Result: ALL OK, with {warn} warning(s) ({ok} of {total} checks fully passed)")
    else:
        print(f"Result: ALL OK ({ok} of {total} checks)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
