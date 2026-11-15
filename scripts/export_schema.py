"""Write the PostgreSQL schema of the four tables to docs/schema_postgresql.sql (the brief asks for the schema as a file).

Run from the project folder:   python -m scripts.export_schema
The file is made from the models in acko/db.py, so it always matches what  python -m scripts.init_db  creates.
It does not need a database: nothing is connected, nothing is changed.
"""
from __future__ import annotations

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from acko import config, db

OUTPUT = config.ROOT / "docs" / "schema_postgresql.sql"
ORDER = ["users", "quotations", "claims", "chat_logs"]      # the order of the brief; users first because the others point to it


def tidy(statement) -> str:
    """The statement as text: four spaces instead of tabs and no spaces at the end of a line."""
    return "\n".join(line.rstrip().replace("\t", "    ") for line in str(statement).strip().splitlines())


def build() -> str:
    dialect = postgresql.dialect()
    tables = {table.name: table for table in db.Base.metadata.sorted_tables}
    if set(tables) != set(ORDER):
        raise SystemExit(f"The tables in acko/db.py ({', '.join(sorted(tables))}) are not the four expected ones.")
    lines = ["-- PostgreSQL schema of the Acko AI Platform: 4 tables.",
             "-- Made by `python -m scripts.export_schema` from the models in acko/db.py.",
             "-- `python -m scripts.init_db` creates the same tables (SQLite on a laptop, PostgreSQL on AWS RDS).",
             ""]
    for name in ORDER:
        table = tables[name]
        lines.append(tidy(CreateTable(table).compile(dialect=dialect)) + ";")
        lines.append("")
        for index in sorted(table.indexes, key=lambda item: item.name):
            lines.append(tidy(CreateIndex(index).compile(dialect=dialect)) + ";")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="ascii", newline="\n") as handle:
        handle.write(build())
    print(f"OK    Wrote {OUTPUT.relative_to(config.ROOT)} ({len(ORDER)} tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
