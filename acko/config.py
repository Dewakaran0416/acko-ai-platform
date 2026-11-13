"""Central settings for the whole project.

Everything that can differ between your laptop and the AWS server comes from
environment variables: a local .env file, or real environment variables on EC2.
No password or API key is ever written in the code.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url

ROOT = Path(__file__).resolve().parents[1]
# Real environment variables win over the .env file (load_dotenv does not override them).
load_dotenv(ROOT / ".env")

# A Windows console can use an old code page that cannot print the rupee sign.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # not a normal text stream (for example inside Streamlit)
        pass


def env(name: str, default: str = "") -> str:
    """Read an environment variable; strips spaces and stray quotes."""
    return os.getenv(name, default).strip().strip('"').strip("'")


# --------------------------------------------------------------------------- folders
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # the 4 big CSV files (not committed to Git)
PDF_DIR = DATA_DIR / "policy_pdfs"    # the Acko policy PDFs for the chatbot
MODEL_DIR = ROOT / "models"           # trained .pkl files, metrics, SHAP pictures
CHROMA_DIR = Path(env("CHROMA_DIR") or ROOT / "chroma_db")   # ChromaDB vector store

RAW_FILES = {
    "car_quote": "acko_car_quotation.csv",
    "bike_quote": "acko_bike_quotation.csv",
    "car_claims": "acko_car_claims.csv",
    "bike_claims": "acko_bike_claims.csv",
}

# --------------------------------------------------------------------------- Gemini
GEMINI_API_KEY = env("GEMINI_API_KEY") or env("GOOGLE_API_KEY")
# Model names change over time. If Gemini answers "404 model not found",
# run `python -m scripts.check_gemini`: it lists the names your key can use.
GEMINI_MODEL = env("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_VISION_MODEL = env("GEMINI_VISION_MODEL") or GEMINI_MODEL

# --------------------------------------------------------------------------- database
APP_TIMEZONE = env("APP_TIMEZONE", "Asia/Kolkata")


def now_local() -> datetime:
    """Current time in the app time zone, stored without a zone (same on laptop and EC2)."""
    return datetime.now(ZoneInfo(APP_TIMEZONE)).replace(tzinfo=None)


def database_url() -> URL:
    """SQLite on your laptop; PostgreSQL (AWS RDS) as soon as DB_HOST is set."""
    if env("DATABASE_URL"):
        return make_url(env("DATABASE_URL"))
    if env("DB_HOST"):
        # URL.create escapes special characters in the password for us.
        return URL.create(
            "postgresql+psycopg2",
            username=env("DB_USER", "postgres"),
            password=env("DB_PASSWORD"),
            host=env("DB_HOST"),
            port=int(env("DB_PORT", "5432")),
            database=env("DB_NAME", "postgres"),
            query={"sslmode": env("DB_SSLMODE")} if env("DB_SSLMODE") else {},
        )
    return URL.create("sqlite", database=str(ROOT / "acko.db"))


# --------------------------------------------------------------------------- business rules
# A claim whose approval probability is below this goes to human review (brief: default 0.5).
APPROVAL_THRESHOLD = float(env("APPROVAL_THRESHOLD", "0.5"))
MANAGER_PASSWORD = env("MANAGER_PASSWORD")
MANAGER_EMAIL = env("MANAGER_EMAIL", "manager@acko.local")

# --------------------------------------------------------------------------- AWS
S3_BUCKET = env("S3_BUCKET")
AWS_REGION = env("AWS_REGION") or env("AWS_DEFAULT_REGION") or "ap-south-1"
