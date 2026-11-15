"""Check that the dashboard shows what is really in the database (the brief: "Dashboard numbers match the actual records").

Run from the project folder:   python -m scripts.check_dashboard

It reads the same database as the app (SQLite on your laptop, PostgreSQL on AWS RDS when DB_HOST is set) two times: once the way the
dashboard does (acko/dashboard_data.py) and once with plain SQL written in this file. For Today, Last 7 days, Last 30 days, This month
and All time it prints both answers side by side and says OK or DIFFERENT. Nothing in the database is changed.
It ends with exit code 0 when every number matches, 1 when one does not.
"""
from __future__ import annotations

import sys
from datetime import datetime, time, timedelta

from sqlalchemy import DateTime, bindparam, text

from acko import config, dashboard_data as dd, db
from acko.fmt import inr

RS = "Rs "        # a Windows console cannot always show the rupee sign
BEGINNING = datetime(1900, 1, 1)


def money(value) -> str:
    return "-" if value is None else inr(value, RS)


class Checker:
    def __init__(self) -> None:
        self.total = 0
        self.different = 0

    def row(self, name: str, dashboard, database, show=str, close: float = 0.0) -> None:
        if dashboard is None or database is None:
            same = dashboard is None and database is None
        else:
            same = abs(float(dashboard) - float(database)) <= close
        self.total += 1
        self.different += (not same)
        print(f"  {name:<34}{show(dashboard) if dashboard is not None else '-':>14}{show(database) if database is not None else '-':>14}   "
              f"{'OK' if same else 'DIFFERENT'}")


def run_sql(connection, sql: str, start: datetime, end: datetime):
    """One plain SQL query for the rows from `start` up to (not including) `end`."""
    statement = text(sql).bindparams(bindparam("start", start, type_=DateTime), bindparam("end", end, type_=DateTime))
    return connection.execute(statement).all()


WINDOW = "created_at >= :start AND created_at < :end"
SQL = {
    "claims": f"SELECT COUNT(*), SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END), SUM(CASE WHEN status = 'review' THEN 1 ELSE 0 END) "
              f"FROM claims WHERE {WINDOW}",
    "car_payout": f"SELECT AVG(predicted_amount) FROM claims WHERE vehicle_type = 'car' AND {WINDOW}",
    "bike_payout": f"SELECT AVG(predicted_amount) FROM claims WHERE vehicle_type = 'bike' AND {WINDOW}",
    "quotes": f"SELECT COUNT(*), AVG(predicted_premium) FROM quotations WHERE {WINDOW}",
    "cities": f"SELECT city, COUNT(*) AS n FROM claims WHERE city IS NOT NULL AND city <> '' AND vehicle_type IN ('car', 'bike') AND {WINDOW} "
              "GROUP BY city",
    "trend": f"SELECT COUNT(*) FROM claims WHERE vehicle_type IN ('car', 'bike') AND {WINDOW}",
}


def midnight(day) -> datetime:
    return datetime.combine(day, time.min)


def main() -> int:
    today = config.now_local().date()
    periods = [("Today", today, today), ("Last 7 days", today - timedelta(days=6), today),
               ("Last 30 days", today - timedelta(days=29), today), ("This month", today.replace(day=1), today), ("All time", None, today)]
    print(f"Database: {db.describe_database()}    Today in India: {today:%d %b %Y}\n")
    checker = Checker()
    with db.get_engine().connect() as connection:
        print("Claims at a glance (claims filed today, in the last 7 days, this month)")
        print(f"  {'':<34}{'dashboard':>14}{'database':>14}")
        windows = dd.claim_windows()
        for name, first, got in (("Claims today", today, windows.today), ("Claims this week (last 7 days)", today - timedelta(days=6), windows.week),
                                 ("Claims this month", today.replace(day=1), windows.month)):
            checker.row(name, got, run_sql(connection, SQL["claims"], midnight(first), midnight(today + timedelta(days=1)))[0][0])
        for name, first, last in periods:
            period = dd.Period(first, last)
            start, end = (BEGINNING if first is None else midnight(first)), midnight(last + timedelta(days=1))
            print(f"\n{name}: " + (period.text() if first is not None else f"everything up to {today:%d %b %Y}"))
            print(f"  {'':<34}{'dashboard':>14}{'database':>14}")
            numbers = dd.summary(period)
            claims, approved, review = run_sql(connection, SQL["claims"], start, end)[0]
            approved, review = int(approved or 0), int(review or 0)
            checker.row("Claims", numbers.claims, claims)
            checker.row("Approved by the model", numbers.approved, approved)
            checker.row("Sent for human review", numbers.review, review)
            checker.row("Approval rate (%)", None if numbers.approval_rate is None else numbers.approval_rate * 100,
                        None if not claims else approved / claims * 100, show=lambda value: f"{float(value):.1f}", close=1e-9)
            for kind in dd.VEHICLES:
                expected = run_sql(connection, SQL[f"{kind}_payout"], start, end)[0][0]
                checker.row(f"Average payout, {kind}", numbers.average_payout[kind], expected, show=money, close=0.005)
            quotes, premium = run_sql(connection, SQL["quotes"], start, end)[0]
            checker.row("Quotations", numbers.quotations, quotes)
            checker.row("Average premium", numbers.average_premium, premium, show=money, close=0.005)
            by_city = {city: int(count) for city, count in run_sql(connection, SQL["cities"], start, end)}
            shown = [(row.city, int(row.total)) for row in dd.claims_by_city(period).itertuples()]
            best = sorted(by_city.values(), reverse=True)[:dd.TOP_CITIES]            # compared by count, so a tie between two cities cannot matter
            checker.row("Top cities: the counts", int([count for _, count in shown] == best), 1, show=lambda value: "same" if int(value) else "different")
            checker.row("Top cities: each city's count", int(all(by_city.get(city) == count for city, count in shown)), 1,
                        show=lambda value: "same" if int(value) else "different")
            checker.row("Claims in the trend chart (total)", int(dd.claims_trend(period).frame["claims"].sum()), run_sql(connection, SQL["trend"], start, end)[0][0])
            checker.row("Claims in the review table", dd.claims_for_review(period).total, review)
    print()
    if checker.different:
        print(f"{checker.different} of {checker.total} numbers are DIFFERENT. Please send me this output.")
        return 1
    print(f"All {checker.total} numbers match the database.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
