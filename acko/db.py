"""Database layer (SQLAlchemy ORM) - the single source of truth for all modules.

The four core tables come from the project brief: users, quotations, claims, chat_logs.
A few extra columns (city, vehicle make/model, fuel type, ...) are added because the
brief's own example questions need them, for example "average payout for bike claims
in Chennai", "quotations for electric cars" or "which vehicle model has the most claims".

SQLite is used on your laptop; PostgreSQL on AWS RDS as soon as DB_HOST is set.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import date, datetime
from typing import Iterator

from sqlalchemy import (Date, DateTime, Float, ForeignKey, Integer, String, Text, Uuid,
                        create_engine, event, inspect, select, text)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from acko import config
from acko.config import now_local


class Base(DeclarativeBase):
    """Every table class below inherits from this one."""


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(150), unique=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(15), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="customer")        # customer / manager
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local)


class Quotation(Base):
    __tablename__ = "quotations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), index=True)
    vehicle_type: Mapped[str] = mapped_column(String(10))                    # car / bike
    vehicle_make: Mapped[str] = mapped_column(String(50))
    vehicle_model: Mapped[str] = mapped_column(String(80))
    manufacturing_year: Mapped[int] = mapped_column(Integer)
    city: Mapped[str] = mapped_column(String(80))
    idv: Mapped[float] = mapped_column(Float)
    ncb_percent: Mapped[int] = mapped_column(Integer)
    predicted_premium: Mapped[float] = mapped_column(Float)
    # extra columns (the brief's manager questions need them, e.g. "electric car quotations"):
    fuel_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    policy_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local, index=True)


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), index=True)
    vehicle_type: Mapped[str] = mapped_column(String(10))                    # car / bike: picks the model pair
    policy_number: Mapped[str | None] = mapped_column(String(50), nullable=True)
    incident_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    damage_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    affected_parts: Mapped[str | None] = mapped_column(String(200), nullable=True)
    damage_severity: Mapped[str | None] = mapped_column(String(20), nullable=True)   # minor / moderate / major
    image_s3_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    form_s3_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    predicted_amount: Mapped[float] = mapped_column(Float)
    approval_probability: Mapped[float] = mapped_column(Float)               # 0 to 1
    status: Mapped[str] = mapped_column(String(20), default="pending")       # pending / approved / rejected / review
    # extra columns (manager questions and dashboard charts need them):
    city: Mapped[str | None] = mapped_column(String(80), nullable=True)
    vehicle_make: Mapped[str | None] = mapped_column(String(50), nullable=True)
    vehicle_model: Mapped[str | None] = mapped_column(String(80), nullable=True)
    idv: Mapped[float | None] = mapped_column(Float, nullable=True)
    # the two details of the claim form that the brief asks Gemini to read (added in Module 3, see NEW_COLUMNS below):
    vehicle_registration: Mapped[str | None] = mapped_column(String(20), nullable=True)
    damage_description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local, index=True)


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), index=True)
    intent: Mapped[str] = mapped_column(String(30))                          # policy_qa / quotation / claims
    question: Mapped[str] = mapped_column(Text)
    retrieved_source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    response: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local, index=True)


# --------------------------------------------------------------------------- engine and sessions
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Create the database connection once and reuse it."""
    global _engine
    if _engine is None:
        url = config.database_url()
        if url.get_backend_name() == "sqlite":
            # Streamlit runs every browser session in its own thread, so SQLite must allow that.
            _engine = create_engine(url, connect_args={"check_same_thread": False})

            @event.listens_for(_engine, "connect")
            def _turn_on_foreign_keys(dbapi_connection, connection_record):
                # SQLite ignores foreign keys unless asked; PostgreSQL always enforces them.
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            connect_args = {"connect_timeout": 10}
            if config.env("DB_SSLMODE"):
                connect_args["sslmode"] = config.env("DB_SSLMODE")     # "require" = always encrypt (RDS)
            _engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Open a session; commit when the block succeeds, roll back when it fails."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Columns that were added to a table after the first version of this project. create_all() never changes a table that
# already exists, so init_db() adds them with ALTER TABLE (the same statement works on SQLite and on PostgreSQL).
# Names and types come from this list only, never from user input.
NEW_COLUMNS = {"claims": {"vehicle_registration": "VARCHAR(20)", "damage_description": "VARCHAR(500)"}}


def ensure_columns() -> list[str]:
    """Add the columns an older database is missing. Returns what was added, for example ['claims.vehicle_registration']."""
    engine = get_engine()
    added = []
    for table, columns in NEW_COLUMNS.items():
        for name, sql_type in columns.items():
            if name in {column["name"] for column in inspect(engine).get_columns(table)}:
                continue
            try:
                with engine.begin() as connection:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))
            except (OperationalError, ProgrammingError):
                # another process may have added it a moment ago: look again, and only fail when it is really missing
                if name not in {column["name"] for column in inspect(engine).get_columns(table)}:
                    raise
            else:
                added.append(f"{table}.{name}")
    return added


def init_db() -> None:
    """Create all four tables, and add the newer columns to tables that already exist (safe to run again)."""
    Base.metadata.create_all(get_engine())
    ensure_columns()


def describe_database() -> str:
    url = get_engine().url
    return f"{url.get_backend_name()} @ {url.host or url.database}"


# --------------------------------------------------------------------------- helpers used by the pages
def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def get_or_create_customer(name: str, email: str, phone: str | None = None) -> User:
    """Find the customer by e-mail or create a new one (role = customer)."""
    email = email.strip().lower()
    with session_scope() as session:
        user = session.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(name=name.strip(), email=email, phone=(phone or "").strip() or None, role="customer")
            session.add(user)
            session.flush()
        return user


def ensure_manager() -> User:
    """The one manager account used after the manager password login."""
    email = config.MANAGER_EMAIL.lower()
    with session_scope() as session:
        user = session.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(name="Acko Manager", email=email, role="manager")
            session.add(user)
            session.flush()
        return user


def save_chat_log(user_id: str | uuid.UUID, intent: str, question: str, response: str,
                  retrieved_source: str | None = None) -> None:
    with session_scope() as session:
        session.add(ChatLog(user_id=_as_uuid(user_id), intent=intent, question=question, response=response,
                            retrieved_source=(retrieved_source or "")[:255] or None))


def save_quotation(user_id: str | uuid.UUID, **fields) -> Quotation:
    with session_scope() as session:
        row = Quotation(user_id=_as_uuid(user_id), **fields)
        session.add(row)
        session.flush()
        return row


def save_claim(user_id: str | uuid.UUID, **fields) -> Claim:
    with session_scope() as session:
        row = Claim(user_id=_as_uuid(user_id), **fields)
        session.add(row)
        session.flush()
        return row


def recent_quotations(user_id: str | uuid.UUID, limit: int = 5) -> list[Quotation]:
    """The newest quotations of one customer (shown under the quote form)."""
    with session_scope() as session:
        query = (select(Quotation).where(Quotation.user_id == _as_uuid(user_id))
                 .order_by(Quotation.created_at.desc()).limit(limit))
        return list(session.scalars(query))


def recent_claims(user_id: str | uuid.UUID, limit: int = 5) -> list[Claim]:
    """The newest claims of one customer (shown under the claim form)."""
    with session_scope() as session:
        query = (select(Claim).where(Claim.user_id == _as_uuid(user_id))
                 .order_by(Claim.created_at.desc()).limit(limit))
        return list(session.scalars(query))
