"""Show the newest claims in the database and check that their files are really stored.

    python -m scripts.show_claims              the 10 newest claims, and whether their photo and form are in S3 (or in uploads/)
    python -m scripts.show_claims --no-files   only the database, without asking S3

It is the proof for the claims part of the brief: every claim has its amount, approval probability and status, and the keys of
its two files (claims/USER-ID/CLAIM-ID/damage.jpg and form.pdf), and those files exist.
"""
from __future__ import annotations

import argparse

from sqlalchemy import func, select

from acko import db, storage
from acko.claim_flow import probability_text
from acko.fmt import inr


def stored(key: str | None) -> str:
    """'stored', 'MISSING', 'none' (no such file was uploaded) or why it could not be checked."""
    if not key:
        return "none"
    try:
        return "stored" if storage.exists(key) else "MISSING"
    except storage.StorageError as error:
        return f"could not be checked: {error}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.show_claims", description="Show the newest claims.")
    parser.add_argument("--no-files", action="store_true", help="do not check whether the files exist in S3 / uploads")
    args = parser.parse_args(argv)
    print("Database :", db.describe_database())
    print("Files    :", storage.describe())
    db.init_db()
    with db.session_scope() as session:
        total = session.scalar(select(func.count()).select_from(db.Claim))
        rows = session.execute(select(db.Claim, db.User.name).join(db.User, db.User.id == db.Claim.user_id)
                               .order_by(db.Claim.created_at.desc()).limit(10)).all()
    print(f"Claims saved so far: {total}   (newest first, at most 10 shown)")
    for claim, name in rows:
        print()
        print(f"{claim.created_at:%Y-%m-%d %H:%M}  claim {str(claim.id)[:8].upper()}  user {str(claim.user_id)[:8]} ({name})")
        vehicle = " ".join(part for part in (claim.vehicle_type, claim.vehicle_make, claim.vehicle_model, claim.vehicle_registration) if part)
        print(f"  vehicle : {vehicle}   policy {claim.policy_number}   incident {claim.incident_date}   city {claim.city}")
        print(f"  damage  : {claim.damage_type}, {claim.damage_severity}; parts: {claim.affected_parts}")
        print(f"  result  : amount {inr(claim.predicted_amount, 'Rs ')} (IDV {inr(claim.idv, 'Rs ') if claim.idv else '-'}), "
              f"approval probability {probability_text(claim.approval_probability)}, status {claim.status}")
        photo = "" if args.no_files else f"   [{stored(claim.image_s3_key)}]"
        form = "" if args.no_files else f"   [{stored(claim.form_s3_key)}]"
        print(f"  photo   : {claim.image_s3_key}{photo}")
        print(f"  form    : {claim.form_s3_key or '-'}{form}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
