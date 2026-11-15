"""Write a few real rows of every table to docs/sample_records.sql (the brief asks for the schema plus sample records).

Run from the project folder:   python -m scripts.export_sample_records
It reads the database the app uses (SQLite on a laptop, PostgreSQL on AWS RDS when DB_HOST is set) and changes no row in it.
For quotations and claims the newest 3 rows of each vehicle type are taken, for the chat log the newest 3 of each topic, and
every customer they belong to, so that all the links between the tables hold.
The file is meant to be public, so people are made anonymous: a name becomes Customer 1, Customer 2 ..., an e-mail address
user1@example.com, user2@example.com ..., and a phone number, a policy number and a vehicle registration keep only their last
two characters. Everything else is as it was saved. The demo customers of `python -m scripts.seed_demo_data` ("Demo Customer N") and
their rows are left out: the file shows what real people did in the app. After writing, the script prints the lines that can hold
free text, so that you can read them before the file is published.

Load it into an empty PostgreSQL database after the schema:
    psql -f docs/schema_postgresql.sql -f docs/sample_records.sql
"""
from __future__ import annotations

import re
import uuid
from datetime import date, datetime

from sqlalchemy import func, select

from acko import config, db

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<![\d.])(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)")   # an Indian mobile number: 10 digits starting with 6 to 9
UUID_TEXT = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
OUTPUT = config.ROOT / "docs" / "sample_records.sql"
PER_GROUP = 3                                  # rows per vehicle type (quotations, claims) and per topic (chat log)
GROUPS = [("quotations", db.Quotation, "vehicle_type"), ("claims", db.Claim, "vehicle_type"), ("chat_logs", db.ChatLog, "intent")]
DEMO_NAME, DEMO_EMAIL = "Demo Customer %", "demo.%@example.com"     # how scripts/seed_demo_data.py names its customers


def demo_ids():
    """The ids of everyone who looks like a demo customer (a query that is used inside the other queries)."""
    return select(db.User.id).where(db.User.name.like(DEMO_NAME), db.User.email.like(DEMO_EMAIL))


def literal(value) -> str:
    """One value as SQL text (the same text is valid in PostgreSQL and in SQLite)."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, uuid.UUID):
        return f"'{value}'"
    if isinstance(value, datetime):
        return "'" + value.isoformat(sep=" ") + "'"
    if isinstance(value, date):
        return "'" + value.isoformat() + "'"
    return "'" + str(value).replace("'", "''") + "'"


def mask_tail(value: str | None) -> str | None:
    """Keep only the last two characters (a phone number, a policy number or a vehicle registration)."""
    return value if not value else "*" * max(len(value) - 2, 0) + value[-2:]


def shorten(text, width: int = 72) -> str:
    """One line of plain ASCII (a Windows console may not show every character), cut at `width`."""
    text = " ".join(str(text or "").split()).encode("ascii", "replace").decode("ascii")
    return text if len(text) <= width else text[: width - 3] + "..."


def look_alikes(text: str) -> list[str]:
    """Anything in the file that looks like a real e-mail address or mobile number (the masked ones and example.com are fine)."""
    text = UUID_TEXT.sub(" ", text)
    found = [mail for mail in EMAIL.findall(text) if not mail.lower().endswith("@example.com")] + [number.strip() for number in PHONE.findall(text)]
    return sorted(set(found))


def pick_rows(session) -> dict:
    """The newest rows of every group, as ORM objects, grouped by table name (rows of demo customers are not taken)."""
    picked = {name: [] for name, _, _ in GROUPS}
    for name, model, group_column in GROUPS:
        column = getattr(model, group_column)
        real = model.user_id.not_in(demo_ids())
        for value in session.scalars(select(column).where(real).distinct().order_by(column)).all():
            picked[name] += session.scalars(select(model).where(column == value, real).order_by(model.created_at.desc()).limit(PER_GROUP)).all()
        picked[name].sort(key=lambda row: row.created_at)
    return picked


def anonymous_users(session, picked: dict) -> tuple[list, dict]:
    """The people the picked rows belong to (oldest first) and, for each, the anonymous values that replace the real ones."""
    owners = {row.user_id for rows in picked.values() for row in rows}
    users = sorted(session.scalars(select(db.User).where(db.User.id.in_(owners))).all(), key=lambda user: (user.created_at, str(user.id))) if owners else []
    masked, customers, managers = {}, 0, 0
    for user in users:
        if user.role == "manager":
            managers += 1
            masked[user.id] = {"name": "Acko Manager", "email": "manager@example.com" if managers == 1 else f"manager{managers}@example.com",
                               "phone": mask_tail(user.phone)}
        else:
            customers += 1
            masked[user.id] = {"name": f"Customer {customers}", "email": f"user{customers}@example.com", "phone": mask_tail(user.phone)}
    return users, masked


def claim_overrides(row) -> dict:
    return {"policy_number": mask_tail(row.policy_number), "vehicle_registration": mask_tail(row.vehicle_registration)}


def insert_line(table, row, overrides: dict | None = None) -> str:
    overrides = overrides or {}
    names = [column.name for column in table.columns]
    values = [literal(overrides[name] if name in overrides else getattr(row, name)) for name in names]
    return f"INSERT INTO {table.name} ({', '.join(names)}) VALUES ({', '.join(values)});"


def build(session) -> tuple[str, dict]:
    tables = db.Base.metadata.tables
    picked = pick_rows(session)
    users, masked = anonymous_users(session, picked)
    totals = {name: int(session.scalar(select(func.count()).select_from(model).where(model.user_id.not_in(demo_ids()))) or 0)
              for name, model, _ in GROUPS}
    totals["users"] = int(session.scalar(select(func.count()).select_from(db.User).where(db.User.id.not_in(demo_ids()))) or 0)

    lines = ["-- Sample records of the Acko AI Platform: a few real rows of every table, written by `python -m scripts.export_sample_records`.",
             f"-- Taken from a {db.get_engine().url.get_backend_name()} database on {config.now_local():%d %b %Y}.",     # no file path or host name in a public file
             f"-- Rows in the whole database (demo rows not counted): users {totals['users']}, quotations {totals['quotations']}, "
             f"claims {totals['claims']}, chat_logs {totals['chat_logs']}.",
             "-- People are anonymous: names become Customer 1, Customer 2 ..., e-mail addresses user1@example.com ...; a phone number,",
             "-- a policy number and a vehicle registration keep only their last two characters.",
             "-- Load it after the schema:  psql -f docs/schema_postgresql.sql -f docs/sample_records.sql",
             "BEGIN;", ""]
    counts = {"users": len(users)}
    lines.append(f"-- users ({len(users)} rows: the people of the rows below)")
    lines += [insert_line(tables["users"], user, masked[user.id]) for user in users]
    for name, _, _ in GROUPS:
        rows = picked[name]
        counts[name] = len(rows)
        lines += ["", f"-- {name} ({len(rows)} of {totals[name]} rows)"]
        lines += [insert_line(tables[name], row, claim_overrides(row) if name == "claims" else None) for row in rows]
    lines += ["", "COMMIT;", ""]
    return "\n".join(lines), counts


def review(session) -> list[str]:
    """The exported values that can hold free text or something personal, one short line each (to be read before publishing)."""
    picked = pick_rows(session)
    users, masked = anonymous_users(session, picked)
    lines = [f"  user   {masked[user.id]['name']} | {masked[user.id]['email']} | {masked[user.id]['phone'] or '-'}" for user in users]
    lines += [f"  claim  {mask_tail(row.vehicle_registration) or '-'} | {mask_tail(row.policy_number) or '-'} | {shorten(row.damage_description) or '-'}"
              for row in picked["claims"]]
    lines += [f"  chat   {row.intent} | {shorten(row.question)}" for row in picked["chat_logs"]]
    return lines


def main() -> int:
    db.init_db()
    with db.session_scope() as session:
        text, counts = build(session)
        shown = review(session)
    if not counts["users"]:
        print("STOP  The database has no rows of real customers yet (demo rows are left out). Use the app (sign in, ask the chatbot, "
              "get a quote) and run this again.")
        return 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"OK    Wrote {OUTPUT.relative_to(config.ROOT)}: " + ", ".join(f"{count} {name}" for name, count in counts.items())
          + f"   (database: {db.describe_database()})")
    print("Read these lines. They are the only places where something personal could still be, and the file will be public:")
    print("\n".join(shown))
    found = look_alikes(text)
    if found:
        print("CHECK Something in the file looks like an e-mail address or a phone number: " + ", ".join(shorten(item, 40) for item in found[:6])
              + ("" if len(found) <= 6 else f" (and {len(found) - 6} more)") + ". Send me a screenshot before you commit.")
    else:
        print("OK    No e-mail address or phone number is left in the file (the masked ones do not count).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
