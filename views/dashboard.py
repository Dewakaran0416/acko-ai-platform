"""Module 4 page: the manager dashboard.

Live numbers read from the database (acko/dashboard_data.py): claims today / this week / this month, the approval rate, the average
payout and premium, the cities with the most claims, the claim trend, the questions customers ask most and the claims that wait
for a person. The manager picks a period (the date filter) and can press Refresh to read the database again. Only a manager sees it.
"""
from __future__ import annotations

from datetime import timedelta

import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from acko import config, dashboard_data as dd
from acko.fmt import inr

user = st.session_state["user"]
if user["role"] != "manager":          # the menu already hides this page from customers; this is the second lock
    st.error("The dashboard is only for Acko managers.")
    st.stop()

# One colour per vehicle type, the same in every chart (the first two colours of a checked palette, one set for each theme).
COLORS = {"light": {"Car": "#2a78d6", "Bike": "#eb6834", "gap": "#ffffff"},
          "dark": {"Car": "#3987e5", "Bike": "#d95926", "gap": "#0e1117"}}
TOPICS = {"policy_qa": "Policy question", "quotation": "Quote", "claims": "Claim"}


def theme_colors() -> dict:
    try:
        return COLORS["dark" if st.context.theme.type == "dark" else "light"]
    except Exception:
        return COLORS["light"]


def money(value) -> str:
    return "-" if value is None else inr(value)


def counted(number: int, word: str) -> str:
    """counted(1, 'claim') -> '1 claim', counted(3, 'claim') -> '3 claims'."""
    return f"{number} {word}" + ("" if number == 1 else "s")


# --------------------------------------------------------------------------- title, period and refresh
today = config.now_local().date()
title_box, refresh_box = st.columns([6, 1.3], vertical_alignment="center")
title_box.title("Management Dashboard")
refresh = refresh_box.button("Refresh", icon=":material/refresh:", width="stretch", help="Read the numbers from the database again.")
st.caption(f"Live numbers from the {dd.backend_name()} database. Times are India time (Asia/Kolkata).")

choice = st.segmented_control("Period", dd.PRESETS, default="Last 30 days", required=True, width="stretch", key="dash_period") or "Last 30 days"
custom = None
if choice == "Custom range":
    picked = st.date_input("First and last day", value=(today - timedelta(days=29), today), max_value=today, width=300, key="dash_dates")
    if len(picked) != 2:               # the manager has clicked the first day only
        st.info("Now pick the last day of the range.")
        st.stop()
    custom = tuple(picked)
period = dd.make_period(choice, today, custom)

try:
    data = dd.snapshot(period)
except Exception as exc:
    st.error(f"The numbers could not be read from the database ({type(exc).__name__}). Press Refresh to try again.")
    st.stop()
if refresh:
    st.toast("The numbers were read from the database again.", icon=":material/check:")
st.caption(f"Selected period: **{period.text()}**. The database was read at {data.read_at:%H:%M:%S}.")

# --------------------------------------------------------------------------- big numbers
st.subheader("Claims at a glance")
windows = data.windows
today_box, week_box, month_box = st.columns(3)
today_box.metric("Claims today", windows.today, border=True, help="Claims filed since midnight.")
week_box.metric("Claims this week", windows.week, border=True, help=f"The last 7 days, from {windows.week_start:%a %d %b}, today included.")
month_box.metric("Claims this month", windows.month, border=True, help=f"Claims filed since {windows.month_start:%d %b}.")
st.caption("These three numbers always show the present; the period above does not change them.")

st.subheader("In the selected period")
numbers = data.summary
quiet = {"delta_color": "off", "delta_arrow": "off", "border": True}          # a small note under the number, no arrow
row = st.columns(4)
row[0].metric("Claims", numbers.claims, delta=f"{numbers.review} sent for review", help="Claims filed in the selected period.", **quiet)
row[1].metric("Approval rate", "-" if numbers.approval_rate is None else f"{numbers.approval_rate * 100:.0f} %",
              delta=f"{numbers.approved} of {counted(numbers.claims, 'claim')}",
              help=f"Share of the claims whose approval probability was high enough (at least {config.APPROVAL_THRESHOLD * 100:.0f} %) "
                   "for the model to approve them. The others go to a person.", **quiet)
row[2].metric("Average payout, car", money(numbers.average_payout["car"]), delta=counted(numbers.claims_by_type["car"], "car claim"),
              help="Average of the predicted claim amounts.", **quiet)
row[3].metric("Average payout, bike", money(numbers.average_payout["bike"]), delta=counted(numbers.claims_by_type["bike"], "bike claim"),
              help="Average of the predicted claim amounts.", **quiet)
row = st.columns(4)
row[0].metric("Quotations", numbers.quotations, delta=f"{numbers.quotations_by_type['car']} car, {numbers.quotations_by_type['bike']} bike",
              help="Quotations given in the selected period.", **quiet)
row[1].metric("Average premium", money(numbers.average_premium), delta=f"of {counted(numbers.quotations, 'quotation')}",
              help="Average of the premiums quoted.", **quiet)
row[2].metric("Average premium, car", money(numbers.average_premium_by_type["car"]),
              delta=counted(numbers.quotations_by_type["car"], "car quotation"), **quiet)
row[3].metric("Average premium, bike", money(numbers.average_premium_by_type["bike"]),
              delta=counted(numbers.quotations_by_type["bike"], "bike quotation"), **quiet)


# --------------------------------------------------------------------------- charts
def chart_height(rows: int) -> int:
    """Both charts get the same height, so that the two cards next to each other are equally tall."""
    return max(320, 36 * rows + 90)


def city_figure(cities, colors) -> go.Figure:
    """Horizontal stacked bars: the cities with the most claims on top, car and bike in their own colour, the total at the end."""
    long = cities.melt(id_vars=["city", "total"], value_vars=list(dd.VEHICLES), var_name="kind", value_name="claims")
    long["Vehicle"] = long["kind"].map(dd.VEHICLE_NAMES)
    long = long[long["claims"] > 0]                                  # a bar of length 0 would only draw a thin line
    figure = px.bar(long, x="claims", y="city", color="Vehicle", orientation="h", color_discrete_map={"Car": colors["Car"], "Bike": colors["Bike"]},
                    category_orders={"city": list(cities["city"]), "Vehicle": ["Car", "Bike"]})
    figure.update_traces(marker_line_color=colors["gap"], marker_line_width=2,
                         hovertemplate="%{y}: %{x} %{fullData.name} claims<extra></extra>")
    figure.add_trace(go.Scatter(x=cities["total"], y=cities["city"], mode="text", text=cities["total"].astype(str), textposition="middle right",
                                showlegend=False, hoverinfo="skip", cliponaxis=False))
    biggest = int(cities["total"].max())
    figure.update_xaxes(title=None, range=[0, biggest * 1.15 + 0.5], showgrid=False, showticklabels=False)     # the totals are written at the bar ends
    figure.update_yaxes(title=None)                 # Plotly Express already puts the first city on top
    figure.update_layout(bargap=0.35, height=chart_height(len(cities)), margin=dict(l=0, r=8, t=8, b=0),
                         legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None))
    return figure


def trend_chart(trend, colors, height: int) -> alt.Chart:
    """One line per vehicle type: claims per day (per week for a long period). Days without a claim are drawn as 0."""
    peak = int(trend.frame["claims"].max()) if len(trend.frame) else 0
    return (alt.Chart(trend.frame)
            .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True))
            .encode(x=alt.X("day:T", title=None, axis=alt.Axis(format="%d %b", labelOverlap="greedy")),
                    y=alt.Y("claims:Q", title=f"Claims per {trend.unit}", scale=alt.Scale(domain=[0, max(peak, 1)], nice=True),
                            axis=alt.Axis(tickMinStep=1, format="d")),
                    color=alt.Color("vehicle:N", title=None, scale=alt.Scale(domain=["Car", "Bike"], range=[colors["Car"], colors["Bike"]]),
                                    legend=alt.Legend(orient="top", direction="horizontal")),
                    tooltip=[alt.Tooltip("day:T", title="Week starting" if trend.unit == "week" else "Day", format="%a %d %b %Y"),
                             alt.Tooltip("vehicle:N", title="Vehicle"), alt.Tooltip("claims:Q", title="Claims")])
            .properties(height=height))                     # Streamlit fits the legend and the axis labels inside this height


colors = theme_colors()
city_box, trend_box = st.columns(2)
with city_box.container(border=True):
    st.markdown("**Top cities by claim volume**")
    st.caption("The cities with the most claims in the selected period.")
    if data.cities.empty:
        st.info("No claims with a city in this period.")
    else:
        best = data.cities.iloc[0]
        st.plotly_chart(city_figure(data.cities, colors), width="stretch", config={"displayModeBar": False},
                        alt=f"Bar chart of claims per city. {best['city']} has the most: {best['total']} claims.")
with trend_box.container(border=True):
    st.markdown("**Claim volume over time**")
    st.caption(f"Claims per {data.trend.unit}. A {data.trend.unit} without a claim shows as 0.")
    if numbers.claims == 0:
        st.info("No claims in this period.")
    else:
        st.altair_chart(trend_chart(data.trend, colors, chart_height(len(data.cities))), width="stretch")

# --------------------------------------------------------------------------- what customers ask
st.subheader("Top 10 chatbot questions this week")
asked_since = data.questions_period.start
st.caption(f"Questions asked in the last 7 days (from {asked_since:%a %d %b}). Questions that differ only in capital letters, spaces "
           "or a final '?' are counted together. Greetings are left out. This list does not change with the period above.")
if data.questions.empty:
    st.info("Nobody has asked the chatbot a question in the last 7 days.")
else:
    questions = data.questions.assign(intent=data.questions["intent"].map(lambda name: TOPICS.get(name, name)))
    st.dataframe(questions, hide_index=True, width="stretch", column_order=["question", "times_asked", "intent", "last_asked"],
                 column_config={"question": st.column_config.TextColumn("Question", width="large"),
                                "times_asked": st.column_config.ProgressColumn("Times asked", min_value=0,
                                                                                max_value=int(questions["times_asked"].max()), format="%d"),
                                "intent": st.column_config.TextColumn("Topic"),
                                "last_asked": st.column_config.DatetimeColumn("Last asked", format="DD MMM, HH:mm")})

# --------------------------------------------------------------------------- claims that wait for a person
review = data.review
st.subheader(f"Claims waiting for human review ({review.total})")
st.caption(f"The model gave each of these claims an approval probability below {config.APPROVAL_THRESHOLD * 100:.0f} %, so a person should look "
           "at it. A low probability means 'please check this claim', it is not proof of fraud. The claims least likely to be approved come first.")
if review.total == 0:
    st.success("No claim is waiting for review in this period.")
else:
    if review.total > len(review.frame):
        st.caption(f"The {len(review.frame)} claims least likely to be approved are shown, out of {review.total}.")
    waiting = review.frame.assign(amount=review.frame["amount"].map(inr), probability=review.frame["probability"] * 100)
    st.dataframe(waiting, hide_index=True, width="stretch",
                 column_order=["reference", "when", "probability", "amount", "customer", "vehicle", "damage", "city", "policy_number"],
                 column_config={"when": st.column_config.DatetimeColumn("When", format="DD MMM, HH:mm"),
                                "reference": st.column_config.TextColumn("Reference", help="The same reference the customer sees."),
                                "customer": "Customer", "policy_number": "Policy number", "vehicle": "Vehicle", "city": "City",
                                "damage": "Damage", "amount": st.column_config.TextColumn("Predicted amount"),
                                "probability": st.column_config.ProgressColumn("Approval probability", min_value=0, max_value=100,
                                                                                format="%.0f %%")})

# --------------------------------------------------------------------------- the same numbers as tables
with st.expander("Show the numbers behind the charts"):
    st.markdown("**Claims per city**")
    st.dataframe(data.cities.rename(columns={"city": "City", "car": "Car", "bike": "Bike", "total": "Total"}), hide_index=True, width="stretch")
    st.markdown(f"**Claims per {data.trend.unit}**")
    by_day = data.trend.frame.pivot(index="day", columns="vehicle", values="claims")[["Car", "Bike"]]
    by_day.index = by_day.index.strftime("%a %d %b %Y")
    by_day.index.name = "Week starting" if data.trend.unit == "week" else "Day"
    st.dataframe(by_day.reset_index(), hide_index=True, width="stretch")
