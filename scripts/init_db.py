"""Step 3: create the 4 database tables and prove that writing works.

Run from the project folder:   python -m scripts.init_db
Safe to run again: existing tables and rows are kept. Later, the same command checks
the PostgreSQL database on AWS RDS (it uses whatever database config.py points to).
"""
from __future__ import annotations

from sqlalchemy import inspect

from acko import db
from acko.db import ChatLog, Claim, Quotation, User

EXPECTED_TABLES = ["users", "quotations", "claims", "chat_logs"]


def write_test() -> None:
    """Write one row into every table inside a transaction that is rolled back at the end."""
    with db.get_session_factory()() as session:
        user = User(name="Write Test", email="write-test@example.com", role="customer")
        session.add(user)
        session.flush()                      # gives the user its id
        session.add_all([
            Quotation(user_id=user.id, vehicle_type="car", vehicle_make="Maruti", vehicle_model="Swift",
                      manufacturing_year=2020, city="Chennai", idv=500000, ncb_percent=25,
                      predicted_premium=7850.0, fuel_type="Petrol"),
            Claim(user_id=user.id, vehicle_type="car", predicted_amount=28500.0, approval_probability=0.82),
            ChatLog(user_id=user.id, intent="policy_qa", question="test question", response="test answer"),
        ])
        session.flush()                      # the database checks every column and foreign key here
        session.rollback()                   # undo everything: no test rows stay in your tables


def main() -> int:
    print("Database :", db.describe_database())
    db.init_db()
    inspector = inspect(db.get_engine())
    found = {name: len(inspector.get_columns(name)) for name in inspector.get_table_names()}
    missing = [name for name in EXPECTED_TABLES if name not in found]
    if missing:
        print("FAIL  Missing tables:", ", ".join(missing))
        return 1
    print("Tables   :", ", ".join(f"{name} ({found[name]} columns)" for name in EXPECTED_TABLES))
    try:
        write_test()
    except Exception as exc:
        print("FAIL  Write test failed:", type(exc).__name__, str(exc)[:300])
        return 1
    print("OK    Write test: one row written to each table, then rolled back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
