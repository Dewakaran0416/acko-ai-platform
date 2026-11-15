"""Module 4: the numbers behind the manager dashboard (the page itself is views/dashboard.py).

Every function asks the database again each time it is called and keeps nothing in memory. So a claim or a quotation that
was saved a moment ago is in the next answer, and the Refresh button of the page really re-reads the data.

Times in the database are Asia/Kolkata wall-clock times without a zone (acko/config.py, now_local), so "today" is today in India,
on a laptop and on the AWS server alike. The queries use only plain SQL (count, sum, avg, group by), so SQLite on a laptop and
PostgreSQL on AWS RDS give the same numbers. scripts/check_dashboard.py compares them with separate SQL.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pandas as pd
from sqlalchemy import case, func, select

from acko import db
from acko.config import now_local
from acko.db import ChatLog, Claim, Quotation, User, session_scope

VEHICLES = ("car", "bike")                       # the two vehicle types, always in this order
VEHICLE_NAMES = {"car": "Car", "bike": "Bike"}
PRESETS = ("Today", "Last 7 days", "Last 30 days", "This month", "All time", "Custom range")
TOP_CITIES = 10                                  # the brief: top cities by claim volume
TOP_QUESTIONS = 10                               # the brief: top 10 chatbot questions of the week
REVIEW_ROWS = 50                                 # the review table shows the 50 most doubtful claims
MAX_QUESTION_ROWS = 20000                        # safety limit when the chat log is read
LONG_PERIOD_DAYS = 92                            # a longer trend is drawn per week instead of per day
# The same greetings as acko/rag.py (copied, so that the dashboard does not have to load ChromaDB just for this list).
SMALL_TALK = re.compile(r"^\s*(hi|hello|hey|hii+|good (morning|afternoon|evening)|thanks|thank you|thx|bye|ok|okay)\b[\s.!?]*$")


def day_start(day: date) -> datetime:
    """Midnight at the start of a day."""
    return datetime.combine(day, time.min)


@dataclass(frozen=True)
class Period:
    """Whole days from `start` to `end`, both days included. `start` is None for 'all time'."""
    start: date | None
    end: date

    @property
    def since(self) -> datetime | None:
        """The first moment inside the period (None = from the very beginning)."""
        return None if self.start is None else day_start(self.start)

    @property
    def until(self) -> datetime:
        """The first moment after the period (midnight at the end of the last day)."""
        return day_start(self.end + timedelta(days=1))

    def conditions(self, column) -> list:
        """The SQL conditions that keep only the rows of `column` that fall inside the period."""
        found = [column < self.until]
        if self.since is not None:
            found.append(column >= self.since)
        return found

    def text(self) -> str:
        if self.start is None:
            return f"All time (up to {self.end:%d %b %Y})"
        if self.start == self.end:
            return f"{self.end:%d %b %Y}"
        if self.start.year == self.end.year:
            return f"{self.start:%d %b} - {self.end:%d %b %Y}"
        return f"{self.start:%d %b %Y} - {self.end:%d %b %Y}"


def make_period(name: str, today: date, custom: tuple[date, date] | None = None) -> Period:
    """The period for one entry of PRESETS. 'Custom range' needs the two dates the manager picked."""
    if name == "Today":
        return Period(today, today)
    if name == "Last 7 days":
        return Period(today - timedelta(days=6), today)
    if name == "Last 30 days":
        return Period(today - timedelta(days=29), today)
    if name == "This month":
        return Period(today.replace(day=1), today)
    if name == "All time":
        return Period(None, today)
    if name == "Custom range" and custom is not None:
        first, last = sorted(custom)
        return Period(first, last)
    raise ValueError(f"Unknown period: {name!r}")


def backend_name() -> str:
    """'PostgreSQL' on AWS RDS, 'SQLite' on a laptop."""
    return {"postgresql": "PostgreSQL", "sqlite": "SQLite"}.get(db.get_engine().url.get_backend_name(), "database")


# --------------------------------------------------------------------------- the numbers
@dataclass(frozen=True)
class ClaimWindows:
    """Claims today, this week and this month. They do not depend on the date filter."""
    today: int
    week: int                  # the last 7 days, today included
    month: int                 # since the 1st of this month
    week_start: date
    month_start: date


def _count_if(condition):
    return func.count(case((condition, 1)))


def claim_windows(now: datetime | None = None) -> ClaimWindows:
    now = now or now_local()
    today = now.date()
    week_start = today - timedelta(days=6)
    month_start = today.replace(day=1)
    with session_scope() as session:
        in_today, in_week, in_month = session.execute(
            select(_count_if(Claim.created_at >= day_start(today)),
                   _count_if(Claim.created_at >= day_start(week_start)),
                   _count_if(Claim.created_at >= day_start(month_start)))
            .where(Claim.created_at < day_start(today + timedelta(days=1)))).one()
    return ClaimWindows(int(in_today), int(in_week), int(in_month), week_start, month_start)


@dataclass(frozen=True)
class Summary:
    """The big numbers of one period."""
    claims: int
    approved: int                                  # status 'approved': the approval probability was high enough
    review: int                                    # status 'review': sent to a human
    claims_by_type: dict
    average_payout: dict                           # average predicted claim amount per vehicle type (None = no claims)
    quotations: int
    quotations_by_type: dict
    average_premium: float | None                  # average predicted premium of all quotations in the period
    average_premium_by_type: dict

    @property
    def approval_rate(self) -> float | None:
        """Share of the claims that were approved (0 to 1), None when there are no claims."""
        return self.approved / self.claims if self.claims else None


def summary(period: Period) -> Summary:
    with session_scope() as session:
        claim_rows = session.execute(
            select(Claim.vehicle_type, func.count(), func.avg(Claim.predicted_amount),
                   func.sum(case((Claim.status == "approved", 1), else_=0)),
                   func.sum(case((Claim.status == "review", 1), else_=0)))
            .where(*period.conditions(Claim.created_at)).group_by(Claim.vehicle_type)).all()
        quote_rows = session.execute(
            select(Quotation.vehicle_type, func.count(), func.sum(Quotation.predicted_premium))
            .where(*period.conditions(Quotation.created_at)).group_by(Quotation.vehicle_type)).all()
    claims_by_type = {kind: 0 for kind in VEHICLES}
    payout = {kind: None for kind in VEHICLES}
    for kind, count, average, _, _ in claim_rows:
        if kind in VEHICLES:
            claims_by_type[kind] = int(count)
            payout[kind] = None if average is None else float(average)
    quotations_by_type = {kind: 0 for kind in VEHICLES}
    premium = {kind: None for kind in VEHICLES}
    for kind, count, total in quote_rows:
        if kind in VEHICLES:
            quotations_by_type[kind] = int(count)
            premium[kind] = None if not count else float(total) / int(count)
    quotations = sum(int(count) for _, count, _ in quote_rows)
    premium_total = sum(float(total or 0) for _, _, total in quote_rows)
    return Summary(
        claims=sum(int(row[1]) for row in claim_rows),
        approved=sum(int(row[3] or 0) for row in claim_rows),
        review=sum(int(row[4] or 0) for row in claim_rows),
        claims_by_type=claims_by_type, average_payout=payout,
        quotations=quotations, quotations_by_type=quotations_by_type,
        average_premium=premium_total / quotations if quotations else None, average_premium_by_type=premium)


def claims_by_city(period: Period, limit: int = TOP_CITIES) -> pd.DataFrame:
    """The cities with the most claims, biggest first. Columns: city, car, bike, total."""
    with session_scope() as session:
        rows = session.execute(
            select(Claim.city, Claim.vehicle_type, func.count())
            .where(Claim.city.is_not(None), Claim.city != "", Claim.vehicle_type.in_(VEHICLES),
                   *period.conditions(Claim.created_at))
            .group_by(Claim.city, Claim.vehicle_type)).all()
    columns = ["city", *VEHICLES, "total"]
    if not rows:
        return pd.DataFrame({name: pd.Series(dtype="object" if name == "city" else "int64") for name in columns})
    counts = {}
    for city, kind, count in rows:
        counts.setdefault(city, dict.fromkeys(VEHICLES, 0))[kind] = int(count)
    table = pd.DataFrame([{"city": city, **per_type, "total": sum(per_type.values())} for city, per_type in counts.items()],
                         columns=columns)
    return table.sort_values(["total", "city"], ascending=[False, True], kind="stable").head(limit).reset_index(drop=True)


@dataclass(frozen=True)
class Trend:
    frame: pd.DataFrame        # columns: day (the first day of the bucket), vehicle ('Car' or 'Bike'), claims
    unit: str                  # 'day' or 'week'


def claims_trend(period: Period) -> Trend:
    """Claims per day (per week when the period is long), car and bike separately. Days without claims count as 0."""
    with session_scope() as session:
        rows = session.execute(
            select(Claim.created_at, Claim.vehicle_type)
            .where(Claim.vehicle_type.in_(VEHICLES), *period.conditions(Claim.created_at))).all()
    data = pd.DataFrame(rows, columns=["created_at", "vehicle_type"])
    data["created_at"] = pd.to_datetime(data["created_at"])
    days = data["created_at"].dt.normalize()
    if period.start is not None:
        first = period.start
    else:
        first = days.min().date() if len(days) else period.end
    unit = "week" if (period.end - first).days + 1 > LONG_PERIOD_DAYS else "day"
    if unit == "week":
        buckets = days - pd.to_timedelta(days.dt.day_of_week, unit="D")           # the Monday of each claim's week
        index = pd.date_range(first - timedelta(days=first.weekday()), period.end - timedelta(days=period.end.weekday()), freq="7D")
    else:
        buckets = days
        index = pd.date_range(first, period.end, freq="D")
    names = [VEHICLE_NAMES[kind] for kind in VEHICLES]
    full = pd.MultiIndex.from_product([index, names], names=["day", "vehicle"])
    if len(data):
        found = pd.DataFrame({"day": buckets, "vehicle": data["vehicle_type"].map(VEHICLE_NAMES)}).groupby(["day", "vehicle"]).size()
        counts = found.reindex(full, fill_value=0)
    else:
        counts = pd.Series(0, index=full)
    return Trend(counts.rename("claims").reset_index(), unit)


def normalize_question(text: str) -> str:
    """Questions that differ only in capital letters, spaces or a final '?' count as the same question."""
    return re.sub(r"\s+", " ", str(text)).strip().lower().strip(" ?!.,;:\"'")


def top_questions(period: Period, limit: int = TOP_QUESTIONS) -> pd.DataFrame:
    """The questions customers asked the chatbot most often. Columns: question, times_asked, last_asked, intent."""
    with session_scope() as session:
        rows = session.execute(
            select(ChatLog.question, ChatLog.intent, ChatLog.created_at)
            .where(*period.conditions(ChatLog.created_at)).order_by(ChatLog.created_at.desc()).limit(MAX_QUESTION_ROWS)).all()
    asked: dict[str, dict] = {}
    for question, intent, when in rows:                  # newest first, so the first wording seen is the latest one
        key = normalize_question(question)
        if not key or SMALL_TALK.match(key):             # a greeting or a thank-you is not a question about the policy
            continue
        entry = asked.setdefault(key, {"question": re.sub(r"\s+", " ", str(question)).strip(), "times_asked": 0,
                                       "last_asked": when, "intents": Counter()})
        entry["times_asked"] += 1
        entry["intents"][intent] += 1
    ranked = sorted(asked.values(), key=lambda entry: entry["last_asked"], reverse=True)
    ranked = sorted(ranked, key=lambda entry: entry["times_asked"], reverse=True)[:limit]       # a stable sort: ties stay newest first
    return pd.DataFrame([{"question": entry["question"], "times_asked": entry["times_asked"], "last_asked": entry["last_asked"],
                          "intent": entry["intents"].most_common(1)[0][0]} for entry in ranked],
                        columns=["question", "times_asked", "last_asked", "intent"])


@dataclass(frozen=True)
class Review:
    frame: pd.DataFrame        # at most REVIEW_ROWS claims, the least likely to be approved first
    total: int                 # all claims waiting for review in the period


def claims_for_review(period: Period, limit: int = REVIEW_ROWS) -> Review:
    """Claims the model did not approve on its own. A low approval probability means 'a person should look', not 'fraud'."""
    waiting = [Claim.status == "review", *period.conditions(Claim.created_at)]
    with session_scope() as session:
        total = int(session.scalar(select(func.count()).select_from(Claim).where(*waiting)) or 0)
        rows = session.execute(
            select(Claim, User.name).join(User, User.id == Claim.user_id).where(*waiting)
            .order_by(Claim.approval_probability.asc(), Claim.created_at.desc()).limit(limit)).all()
    table = []
    for claim, customer in rows:
        vehicle = VEHICLE_NAMES.get(claim.vehicle_type, str(claim.vehicle_type).title())
        make_and_model = " ".join(part for part in (claim.vehicle_make, claim.vehicle_model) if part)
        if make_and_model:
            vehicle += f" ({make_and_model})"
        damage = "-" if not claim.damage_type else claim.damage_type + (f" ({claim.damage_severity})" if claim.damage_severity else "")
        table.append({"reference": str(claim.id)[:8].upper(), "when": claim.created_at, "customer": customer,
                      "policy_number": claim.policy_number or "-", "vehicle": vehicle, "city": claim.city or "-",
                      "damage": damage, "amount": float(claim.predicted_amount), "probability": float(claim.approval_probability)})
    columns = ["reference", "when", "customer", "policy_number", "vehicle", "city", "damage", "amount", "probability"]
    return Review(pd.DataFrame(table, columns=columns), total)


@dataclass(frozen=True)
class Snapshot:
    """Everything the dashboard shows, read from the database in one go."""
    read_at: datetime
    period: Period
    windows: ClaimWindows
    summary: Summary
    cities: pd.DataFrame
    trend: Trend
    questions_period: Period                       # always the last 7 days
    questions: pd.DataFrame
    review: Review


def snapshot(period: Period, now: datetime | None = None) -> Snapshot:
    now = now or now_local()
    windows = claim_windows(now)
    week = Period(windows.week_start, now.date())
    return Snapshot(read_at=now, period=period, windows=windows, summary=summary(period), cities=claims_by_city(period),
                    trend=claims_trend(period), questions_period=week, questions=top_questions(week),
                    review=claims_for_review(period))
