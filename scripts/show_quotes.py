"""Show the newest quotations in the database: when, who asked, for which vehicle, and the premium.

Run from the project folder:   python -m scripts.show_quotes
It is the proof that every quote from the Get a Quote page is saved with its time and the customer's id.
"""
from __future__ import annotations

from sqlalchemy import func, select

from acko import db
from acko.fmt import inr


def main() -> int:
    print("Database :", db.describe_database())
    db.init_db()
    with db.session_scope() as session:
        total = session.scalar(select(func.count()).select_from(db.Quotation))
        rows = session.execute(select(db.Quotation, db.User.name).join(db.User, db.User.id == db.Quotation.user_id)
                               .order_by(db.Quotation.created_at.desc()).limit(10)).all()
    print(f"Quotations saved so far: {total}   (newest first, at most 10 shown)")
    for row, name in rows:
        print(f"{row.created_at:%Y-%m-%d %H:%M}  user {str(row.user_id)[:8]} ({name})  {row.vehicle_type} "
              f"{row.vehicle_make} {row.vehicle_model} {row.manufacturing_year} ({row.fuel_type}), {row.city}, "
              f"IDV {inr(row.idv, 'Rs ')}, NCB {row.ncb_percent}%, {row.policy_type}  ->  premium {inr(row.predicted_premium, 'Rs ')}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
