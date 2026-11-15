"""Fill the database with clearly labelled demo rows, so that the manager dashboard has something to show (optional).

Run from the project folder:
    python -m scripts.seed_demo_data             add the demo rows
    python -m scripts.seed_demo_data --remove    delete them again (only them)
    python -m scripts.seed_demo_data --refresh   delete them and add them again, so that the dates move up to today

Why: a new database, for example the one on AWS RDS, is empty, and an empty dashboard has nothing to show. This script adds 24 demo
customers ("Demo Customer 1" ... with the e-mail address demo.1@example.com ...) and for them 160 quotations, 70 claims and
200 chatbot questions, spread over the last 45 days (the questions over the last 21 days).

Nothing is typed in by hand: every premium comes from your trained quote model, and every claim amount, approval probability and
status ("approved" or "review") from your trained claims models, the same ones the app uses. The vehicles, cities, damage types and
questions are typical ones, so the charts look like those of a real company. The answers of the chatbot are not real (the response
column says so), and a demo claim has no photo and no form (its image and form keys are empty).

The rows are the same every time (a fixed random seed); only the dates follow today's date. The script refuses to add a second set.
Real customers are never touched: --remove deletes only users whose name is "Demo Customer N" AND whose e-mail address is
demo.N@example.com, together with their rows. The same test keeps these rows out of docs/sample_records.sql.
"""
from __future__ import annotations

import argparse
import random
import re
import uuid
from datetime import datetime, timedelta

from sqlalchemy import delete, select

from acko import config, db
from acko.fmt import inr

SEED = "acko-demo-2026"                 # the same rows every time
DAYS = 45                               # quotations and claims: the last 45 days (the dashboard starts with the last 30)
CHAT_DAYS = 21                          # chatbot questions: the last 21 days (the dashboard's question list looks at the last 7)
CUSTOMERS, QUOTATIONS, CLAIMS, CHATS = 24, 160, 70, 200
SHARE_OF_CARS = {"quotations": 0.55, "claims": 0.55}
NAME, EMAIL = "Demo Customer {}", "demo.{}@example.com"
NAME_LIKE, EMAIL_LIKE = "Demo Customer %", "demo.%@example.com"          # the database search ...
NAME_PATTERN, EMAIL_PATTERN = re.compile(r"Demo Customer (\d+)"), re.compile(r"demo\.(\d+)@example\.com")   # ... and the exact test
ANSWER = "Demo data: this answer was not written by the chatbot."

# How often each make, city, year ... is chosen (the ones that are not listed get the small default weight).
MAKE_WEIGHTS = {
    "car": {"Maruti Suzuki": 10, "Hyundai": 7, "Tata": 7, "Mahindra": 5, "Toyota": 4, "Honda": 3, "Kia": 3, "MG": 1.5, "Volkswagen": 1.2,
            "Skoda": 1.2, "Renault": 1.2, "Ford": 0.8, "BMW": 0.8, "Mercedes-Benz": 0.8, "Jeep": 0.6, "Audi": 0.6, "Nissan": 0.5, "BYD": 0.4},
    "bike": {"Hero MotoCorp": 8, "Honda": 7, "TVS": 6, "Bajaj": 6, "Royal Enfield": 5, "Suzuki": 2, "Yamaha": 2, "KTM": 1.5, "Ola Electric": 1.5,
             "Ather Energy": 1.2, "Kawasaki": 0.6, "Ampere": 0.5, "Bounce": 0.3}}
FUEL_WEIGHTS = {"Petrol": 6, "Diesel": 3, "Electric": 1, "CNG": 0.7, "Hybrid": 0.5, "LPG": 0.05}
YEAR_WEIGHTS = {2025: 6, 2024: 8, 2023: 8, 2022: 7, 2021: 6, 2020: 5, 2019: 5, 2018: 4, 2017: 3, 2016: 3, 2015: 2, 2014: 2}
CITY_WEIGHTS = {"Mumbai": 12, "Delhi": 11, "Bengaluru": 11, "Chennai": 8, "Hyderabad": 8, "Pune": 7, "Kolkata": 5, "Ahmedabad": 4, "Jaipur": 3,
                "Lucknow": 2.5, "Surat": 2.5, "Kochi": 2, "Coimbatore": 2, "Chandigarh": 2, "Gurgaon": 2, "Noida": 2, "Nagpur": 1.5,
                "Indore": 1.5, "Thane": 1.5, "Vadodara": 1.2, "Visakhapatnam": 1.2, "Patna": 1, "Nashik": 1, "Bhopal": 1}
SMALL_CITY = 0.08
POLICY_TYPES = {"Comprehensive": 70, "Third Party": 20, "Own Damage": 10}
NCB_WEIGHTS = [2.5, 2, 2, 2, 1.5, 1]                           # for the steps 0, 20, 25, 35, 45, 50 (the steps are read from the model)
USAGE = {"Personal": 7, "Delivery": 2, "Commercial": 1}
TIMES_OF_DAY = {"Morning": 3, "Afternoon": 3, "Evening": 3, "Night": 2}
PREVIOUS_CLAIMS = [62, 22, 10, 4, 2]                            # for 0, 1, 2, 3, 4 earlier claims
# damage type: (weight for a car, weight for a bike, lowest and highest damage severity score, 1 to 10)
DAMAGE = {"scratch": (6, 5, 1, 3), "dent": (5, 4, 2, 5), "bumper damage": (4, 2, 3, 6), "collision damage": (4, 4, 5, 9),
          "windshield damage": (3, 0.3, 3, 6), "panel damage": (2, 1, 4, 7), "crack": (2, 1.5, 2, 5), "partial loss": (1.6, 1.6, 6, 9),
          "flood damage": (0.8, 0.4, 6, 10), "fire damage": (1, 0.8, 7, 10), "rollover": (0.5, 1, 7, 10), "theft": (1.2, 1.5, 8, 10),
          "total loss": (0.8, 0.8, 9, 10)}

# What customers ask the chatbot: (question, topic, how often). The topic is the one the chatbot itself gives such a question.
CHAT = [
    ("How do I file a claim?", "claims", 14),
    ("What does the Acko motor insurance policy cover?", "policy_qa", 12),
    ("How much will insurance cost for my Maruti Swift?", "quotation", 10),
    ("What is zero depreciation cover?", "policy_qa", 9),
    ("How long does claim settlement take?", "claims", 8),
    ("What documents do I need to file a claim?", "claims", 8),
    ("What is IDV?", "policy_qa", 7),
    ("How does the no claim bonus work?", "policy_qa", 7),
    ("I want a quote for my bike", "quotation", 6),
    ("Does the policy cover flood damage?", "policy_qa", 6),
    ("What is not covered under my car insurance?", "policy_qa", 5),
    ("What is the premium for a Hyundai Creta in Chennai?", "quotation", 5),
    ("How do I file a claim for a bike accident?", "claims", 4),
    ("How do I renew my policy?", "policy_qa", 4),
    ("Is theft of my vehicle covered?", "policy_qa", 4),
    ("What photos should I upload for a claim?", "claims", 3),
    ("Can I transfer my NCB to a new car?", "policy_qa", 3),
    ("Get me a quote for a 2020 Honda City petrol", "quotation", 3),
    ("Will my claim be approved automatically?", "claims", 3),
    ("What is the waiting period for Acko health insurance?", "policy_qa", 3),
    ("Can I claim for a stolen vehicle?", "claims", 2),
    ("What happens after I submit a claim?", "claims", 2),
    ("Does the policy cover tyre damage?", "policy_qa", 2),
    ("Can I cancel my policy and get a refund?", "policy_qa", 2),
    ("Quote for Royal Enfield Classic 350 in Pune", "quotation", 2),
    ("How much is third party insurance for a bike?", "quotation", 2),
    ("Is roadside assistance included?", "policy_qa", 2),
    ("My car was damaged in a flood, can I claim?", "claims", 2),
    ("Can you give me a premium for my Tata Nexon diesel?", "quotation", 1),
    ("Is a pre-existing disease covered in the health policy?", "policy_qa", 1),
    ("Can I file a claim without a police report?", "claims", 1),
    ("Does the policy cover personal accident?", "policy_qa", 1),
    ("What is the premium for an electric scooter?", "quotation", 1),
    ("What is the deductible on my policy?", "policy_qa", 1),
]


# --------------------------------------------------------------------------- which rows are demo rows
def demo_users(session) -> list:
    """The demo customers of this database: name "Demo Customer N" and e-mail demo.N@example.com with the same N."""
    candidates = session.scalars(select(db.User).where(db.User.role == "customer", db.User.name.like(NAME_LIKE),
                                                       db.User.email.like(EMAIL_LIKE))).all()
    found = []
    for user in candidates:
        by_name, by_email = NAME_PATTERN.fullmatch(user.name or ""), EMAIL_PATTERN.fullmatch(user.email or "")
        if by_name and by_email and by_name.group(1) == by_email.group(1):
            found.append(user)
    return sorted(found, key=lambda user: int(NAME_PATTERN.fullmatch(user.name).group(1)))


def remove_demo() -> dict:
    """Delete the demo customers and everything that belongs to them. Returns how many rows went from each table."""
    removed = {"users": 0, "quotations": 0, "claims": 0, "chat_logs": 0}
    with db.session_scope() as session:
        ids = [user.id for user in demo_users(session)]
        if ids:
            for name, model in (("chat_logs", db.ChatLog), ("claims", db.Claim), ("quotations", db.Quotation)):
                removed[name] = int(session.execute(delete(model).where(model.user_id.in_(ids))).rowcount or 0)
            removed["users"] = int(session.execute(delete(db.User).where(db.User.id.in_(ids))).rowcount or 0)
    return removed


# --------------------------------------------------------------------------- small helpers
def stream(name: str) -> random.Random:
    """One random stream for each table, so that the rows of one table do not change when another one does."""
    return random.Random(f"{SEED}/{name}")


def new_id(rng: random.Random) -> uuid.UUID:
    return uuid.UUID(int=rng.getrandbits(128), version=4)


def weighted(rng: random.Random, options: list, table: dict, default: float):
    """One of `options`; an option that is not in `table` gets the weight `default`."""
    return rng.choices(options, weights=[table.get(option, default) for option in options])[0]


def moment(rng: random.Random, now: datetime, days: int, bias: float) -> datetime:
    """A time in the last `days` days: more in the recent days, mostly between 07:00 and 23:00, never later than now."""
    day = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=int(rng.random() ** bias * days))
    stamp = day + timedelta(minutes=int(rng.triangular(7 * 60, 23 * 60, 14 * 60)), seconds=rng.randint(0, 59))
    earlier = timedelta(minutes=rng.randint(1, 240), seconds=rng.randint(0, 59))        # always drawn, so that the rows do not depend on the hour
    return stamp if stamp < now else now - earlier


def kind_of(rng: random.Random, table: str) -> str:
    return "car" if rng.random() < SHARE_OF_CARS[table] else "bike"


def pick_vehicle(rng, engine, kind: str, options: dict) -> tuple[str, str, str, int, int]:
    """make, model, fuel, year of manufacture and an insured value (IDV) that is usual for them, within the limits of the quote form."""
    make = weighted(rng, options["makes"], MAKE_WEIGHTS[kind], 0.5)
    model = rng.choice(list(options["models"][make]))
    fuel = weighted(rng, list(options["models"][make][model]), FUEL_WEIGHTS, 0.2)
    year = weighted(rng, list(YEAR_WEIGHTS), YEAR_WEIGHTS, 1)
    low, high = options["idv_form"]["min"], options["idv_form"]["max"]
    usual = engine.typical_idv(kind, make, model, fuel, year) or (500_000 if kind == "car" else 60_000)
    idv = int(min(max(round(usual * rng.uniform(0.85, 1.15), -3), low), high))
    return make, model, fuel, year, idv


def pick_ncb(rng, steps: list) -> int:
    return int(rng.choices(steps, weights=NCB_WEIGHTS[: len(steps)] + [1] * max(len(steps) - len(NCB_WEIGHTS), 0))[0])


def small_count(rng: random.Random, highest: int, weights: list) -> int:
    """A small whole number from 0 to `highest` (0 is the most usual)."""
    return int(rng.choices(range(highest + 1), weights=weights[: highest + 1])[0])


def pick_customer(rng, users: list):
    return rng.choices(users, weights=[1 / (number + 1) ** 0.4 for number in range(len(users))])[0]


# --------------------------------------------------------------------------- the rows
def make_users(now: datetime) -> list:
    """The demo customers. Their ids come from their e-mail addresses, so they are the same every time."""
    return [db.User(id=uuid.uuid5(uuid.NAMESPACE_DNS, EMAIL.format(number)), name=NAME.format(number), email=EMAIL.format(number),
                    phone=None, role="customer", created_at=now - timedelta(days=DAYS + 5))
            for number in range(1, CUSTOMERS + 1)]


def make_quotations(quote_engine, users: list, now: datetime) -> list:
    rng, rows = stream("quotations"), []
    for _ in range(QUOTATIONS):
        kind = kind_of(rng, "quotations")
        options = quote_engine.options(kind)
        make, model, fuel, year, idv = pick_vehicle(rng, quote_engine, kind, options)
        city = weighted(rng, options["cities"], CITY_WEIGHTS, SMALL_CITY)
        ncb = pick_ncb(rng, options["ncb_steps"])
        policy_type = weighted(rng, options["policy_types"], POLICY_TYPES, 1)
        add_ons = small_count(rng, options["ranges"]["num_addons"][1], [40, 30, 18, 8, 4])
        past_claims = small_count(rng, options["ranges"]["claim_history_count"][1], [70, 20, 7, 3, 1])
        low, high = options["ranges"]["customer_age"]
        driver_age = int(min(max(rng.gauss(36, 9), low), high))
        usage = weighted(rng, options["usage_types"], USAGE, 1) if kind == "bike" else None
        customer = pick_customer(rng, users)
        when = moment(rng, now, DAYS, 1.4)
        row_id = new_id(rng)
        result = quote_engine.quote(kind, make, model, fuel, year, city, idv, ncb, policy_type=policy_type, num_addons=add_ons,
                                    past_claims=past_claims, driver_age=driver_age, usage_type=usage)
        rows.append(db.Quotation(id=row_id, user_id=customer.id, vehicle_type=kind, vehicle_make=make, vehicle_model=model,
                                 manufacturing_year=year, city=city, idv=float(idv), ncb_percent=ncb,
                                 predicted_premium=round(float(result.premium), 2), fuel_type=fuel, policy_type=policy_type, created_at=when))
    return rows


def make_claims(claim_engine, quote_engine, users: list, now: datetime) -> list:
    rng, rows = stream("claims"), []
    for number in range(1, CLAIMS + 1):
        kind = kind_of(rng, "claims")
        options, quote_options = claim_engine.options(kind), quote_engine.options(kind)
        make, model, fuel, year, idv = pick_vehicle(rng, quote_engine, kind, quote_options)
        city = weighted(rng, options["cities"], CITY_WEIGHTS, SMALL_CITY)
        types = [name for name in DAMAGE if name in options["damage_types"]]
        damage = rng.choices(types, weights=[DAMAGE[name][0 if kind == "car" else 1] for name in types])[0]
        severity = rng.randint(DAMAGE[damage][2], DAMAGE[damage][3])
        most = options["max_parts"]
        how_many = int(min(most, max(1, round(rng.triangular(1, most, 1 + severity / 10 * (most - 1))))))
        chosen = set(rng.sample(options["parts"], how_many))
        parts = [part for part in options["parts"] if part in chosen]
        zero_dep = 1 if rng.random() < 0.3 else 0
        previous = small_count(rng, options["ranges"]["previous_claims_count"][1], PREVIOUS_CLAIMS)
        policy_age = rng.randint(*options["ranges"]["policy_age_months"])
        time_of_day = weighted(rng, options["times_of_day"], TIMES_OF_DAY, 1)
        ncb = pick_ncb(rng, options["ncb_steps"])
        usage = weighted(rng, options["usage_types"], USAGE, 1) if kind == "bike" else None
        helmet = 1 if rng.random() < 0.88 else 0
        experience = int(min(max(rng.gauss(8, 6), options["ranges"]["rider_experience_years"][0]),
                             options["ranges"]["rider_experience_years"][1])) if kind == "bike" else None
        customer = pick_customer(rng, users)
        when = moment(rng, now, DAYS, 1.4)
        incident = (when - timedelta(days=rng.randint(0, 4))).date()
        row_id = new_id(rng)
        estimate = claim_engine.estimate(kind, damage, severity, len(parts), float(idv), zero_dep=zero_dep, previous_claims=previous,
                                         policy_age_months=policy_age, time_of_day=time_of_day, ncb=ncb, city=city, usage_type=usage,
                                         helmet_worn=helmet if kind == "bike" else None, rider_experience=experience)
        rows.append(db.Claim(
            id=row_id, user_id=customer.id, vehicle_type=kind, policy_number=f"DEMO-{kind.upper()}-{number:05d}", incident_date=incident,
            damage_type=damage, affected_parts=", ".join(parts), damage_severity=estimate.severity, image_s3_key=None, form_s3_key=None,
            predicted_amount=round(estimate.amount, 2), approval_probability=round(estimate.probability, 4), status=estimate.status,
            city=city, vehicle_make=make, vehicle_model=model, idv=float(idv), vehicle_registration=None,
            damage_description=f"Demo claim: {damage} ({estimate.severity}), parts: {', '.join(parts)}."[:500], created_at=when))
    return rows


def make_chat_logs(users: list, now: datetime) -> list:
    rng, rows = stream("chat_logs"), []
    for _ in range(CHATS):
        question, intent, _ = rng.choices(CHAT, weights=[weight for _, _, weight in CHAT])[0]
        customer = pick_customer(rng, users)
        when = moment(rng, now, CHAT_DAYS, 1.2)
        rows.append(db.ChatLog(id=new_id(rng), user_id=customer.id, intent=intent, question=question, retrieved_source=None,
                               response=ANSWER, created_at=when))
    return rows


def build(claim_engine, quote_engine, now: datetime) -> dict:
    users = make_users(now)
    rows = {"quotations": make_quotations(quote_engine, users, now), "claims": make_claims(claim_engine, quote_engine, users, now),
            "chat_logs": make_chat_logs(users, now)}
    first = {}
    for table in rows.values():
        for row in table:
            first[row.user_id] = min(first.get(row.user_id, row.created_at), row.created_at)
    rng = stream("users")
    for user in users:        # a customer exists a little before the first thing he or she did
        user.created_at = first.get(user.id, now - timedelta(days=DAYS)) - timedelta(minutes=rng.randint(2, 40))
    rows["users"] = users
    return rows


# --------------------------------------------------------------------------- the commands
def summary(rows: dict) -> list[str]:
    claims, quotes = rows["claims"], rows["quotations"]
    review = sum(1 for claim in claims if claim.status == "review")
    average = {kind: [quote.predicted_premium for quote in quotes if quote.vehicle_type == kind] for kind in ("car", "bike")}
    stamps = [row.created_at for table in ("quotations", "claims", "chat_logs") for row in rows[table]]
    return [f"      Claims: {len(claims) - review} approved, {review} sent for review ({round(100 * review / max(len(claims), 1))} %).   "
            f"Average premium: car {inr(sum(average['car']) / max(len(average['car']), 1), 'Rs ')}, "
            f"bike {inr(sum(average['bike']) / max(len(average['bike']), 1), 'Rs ')}.",
            f"      Dates: {min(stamps):%d %b %Y} to {max(stamps):%d %b %Y}."]


def add_demo() -> int:
    with db.session_scope() as session:
        already = demo_users(session)
        emails = [EMAIL.format(number) for number in range(1, CUSTOMERS + 1)]
        taken = session.scalars(select(db.User.email).where(db.User.email.in_(emails))).all()
    if already:
        print(f"STOP  The demo rows are already in this database ({len(already)} demo customers). Nothing was added.")
        print("      To start again with fresh dates:   python -m scripts.seed_demo_data --refresh")
        return 1
    if taken:
        print(f"STOP  {len(taken)} of the e-mail addresses demo.1@example.com ... are used by other customers. Nothing was added.")
        return 1
    try:
        from acko import claims, quote
        claim_engine, quote_engine = claims.ClaimEngine.load(), quote.QuoteEngine.load()
    except RuntimeError as error:
        print(f"STOP  The trained models could not be loaded. {error}")
        return 1
    print("Working: the trained models price the demo quotations and decide the demo claims (it can take up to a minute) ...")
    rows = build(claim_engine, quote_engine, config.now_local())
    with db.session_scope() as session:          # one transaction: either all rows are saved or none
        session.add_all(rows["users"])
        session.flush()
        for name in ("quotations", "claims", "chat_logs"):
            session.add_all(rows[name])
    print(f"OK    Added {len(rows['users'])} demo customers, {len(rows['quotations'])} quotations, {len(rows['claims'])} claims and "
          f"{len(rows['chat_logs'])} chatbot questions.")
    print("\n".join(summary(rows)))
    print("      Open the app as manager and look at the Dashboard.   Check its numbers:   python -m scripts.check_dashboard")
    print("      Take the demo rows away again:   python -m scripts.seed_demo_data --remove")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.seed_demo_data", description="Add or remove demo rows for the manager dashboard.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--remove", action="store_true", help="delete the demo rows (and only them)")
    group.add_argument("--refresh", action="store_true", help="delete the demo rows and add them again, with dates up to today")
    args = parser.parse_args(argv)
    print("Database :", db.describe_database())
    db.init_db()
    if args.remove or args.refresh:
        removed = remove_demo()
        if removed["users"]:
            print(f"OK    Removed {removed['users']} demo customers, {removed['quotations']} quotations, {removed['claims']} claims and "
                  f"{removed['chat_logs']} chatbot questions. Rows of real customers were not touched.")
        else:
            print("OK    There are no demo rows in this database, so nothing was removed.")
        if args.remove:
            return 0
    return add_demo()


if __name__ == "__main__":
    raise SystemExit(main())
