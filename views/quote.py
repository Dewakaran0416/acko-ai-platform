"""Module 2 page: Get a Quote - the form, the premium estimate, the SHAP chart, and saving to the quotations table."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from acko import config, db, quote, quote_ui
from acko.fmt import RUPEE, inr

RED, GREEN, GREY = "#c0392b", "#1e8449", "#566573"       # more rupees / fewer rupees / totals
user = st.session_state["user"]
state = st.session_state

st.title("Get a Quote")
st.caption("Tell us about your vehicle. The estimate comes from a model trained on Acko's quotation data, "
           "and the chart shows which factors moved the price.")

try:
    engine = quote_ui.get_engine()      # loaded once, and again whenever scripts.train_quote writes a new model file
    quote_ui.warm_up()
except quote.ModelMissing as exc:
    st.error(str(exc))
    st.stop()


def waterfall(result: quote.Quote) -> go.Figure:
    names = ["Typical policy"] + [f.label for f in result.factors] + ["Your quote"]
    values = [result.typical] + [f.rupees for f in result.factors] + [result.premium]
    texts = [inr(result.typical)] + [("+" if f.rupees >= 0 else "-") + inr(abs(f.rupees)) for f in result.factors] + [inr(result.premium)]
    hover = ["The typical premium of all policies in the training data"] + [f.detail for f in result.factors] + \
            ["The typical premium plus every change above"]
    figure = go.Figure(go.Waterfall(
        x=names, y=values, measure=["absolute"] + ["relative"] * len(result.factors) + ["total"],
        text=texts, textposition="outside", textfont={"size": 12}, cliponaxis=False, customdata=hover,
        hovertemplate="<b>%{x}</b><br>%{customdata}<extra></extra>",
        increasing={"marker": {"color": RED}}, decreasing={"marker": {"color": GREEN}}, totals={"marker": {"color": GREY}},
        connector={"line": {"color": "#8a8f98", "width": 1}}))
    figure.update_layout(height=440, margin={"l": 10, "r": 10, "t": 20, "b": 10}, showlegend=False,
                         yaxis_title=f"Annual premium ({RUPEE})", xaxis={"tickangle": -30}, waterfallgap=0.3)
    return figure


def show_result(last: dict) -> None:
    result: quote.Quote = last["quote"]
    st.divider()
    st.subheader(last["title"])
    left, right = st.columns([1, 2])
    with left:
        st.metric("Estimated annual premium", inr(result.premium))
        error = engine.metrics(result.kind)["form_range"]
        st.caption(f"It is an estimate. On policies the model had never seen, its typical error was about "
                   f"{error['median_pct_error']:.0f} % (RMSE {inr(error['rmse'])}).")
        st.caption(f"The typical premium of all {result.kind} policies is {inr(result.typical)}.")
        saved = last["saved"]
        if "id" in saved:
            st.success(f"Saved as quotation {saved['id'][:8].upper()} on {saved['at']:%d %b %Y at %H:%M}.")
        else:
            st.warning(f"The quote could not be saved to the database ({saved['error']}).")
    with right:
        st.plotly_chart(waterfall(result), config={"displayModeBar": False})
        st.caption("Read it from left to right: start at the typical premium. Red bars add rupees, green bars take rupees off. "
                   "The last bar is your quote.")
    for note in result.notes:
        st.warning(note)
    by_key = {f.key: f for f in result.factors}
    st.write("**The four main factors for this quote**")
    for column, key in zip(st.columns(4), ["idv", "city", "age", "ncb"]):
        factor = by_key[key]
        sign = "+" if factor.rupees >= 0 else "-"
        column.metric(factor.label, factor.detail.split(" (")[0], f"{sign}{inr(abs(factor.rupees))}", delta_color="inverse")
    with st.expander("See the numbers behind the chart"):
        st.dataframe([{"Factor": f.label, "Your value": f.detail,
                       f"Effect on the premium ({RUPEE})": ("+" if f.rupees >= 0 else "-") + inr(abs(f.rupees))}
                      for f in result.factors], hide_index=True)


# ------------------------------------------------------------------ the form
def start_with(key: str, value, allowed=None) -> None:
    """Give a widget its first value through session state. The widget is then created without index= or value=,
    so details placed there by the chat (fill_from_chat) never clash with the widget's own default."""
    if key not in state or (allowed is not None and state[key] not in allowed):
        state[key] = value


def fill_from_chat(details: dict) -> None:
    """Put the details found in a chat question into the widgets' state. Must run before the widgets are created."""
    kinds = [details["kind"]] if details.get("kind") in quote.KINDS else list(quote.KINDS)
    if len(kinds) == 1:
        state["quote_kind"] = kinds[0].title()
        kind, make, model = kinds[0], details.get("make"), details.get("model")
        if make:
            state[f"quote_make_{kind}"] = make
            if model:
                state[f"quote_model_{kind}_{make}"] = model
                if details.get("fuel"):
                    state[f"quote_fuel_{kind}_{make}_{model}"] = details["fuel"]
    for kind in kinds:
        for field, name in (("year", "year"), ("city", "city"), ("ncb", "ncb"), ("policy_type", "policy")):
            if field in details:
                state[f"quote_{name}_{kind}"] = details[field]
    state.pop("last_quote", None)           # the old result belongs to the old question


# The make / model / fuel lists depend on each other, so the form is not wrapped in st.form (that would freeze them).
prefill, chat_idv = state.pop("quote_prefill", None), None
if prefill:
    fill_from_chat(prefill["details"])
    chat_idv = prefill["details"].get("idv")
    st.info(f"Filled in from your chat question: {prefill['text']}. Please check every field, then press Get my quote.")

start_with("quote_kind", "Car", ["Car", "Bike"])
kind = st.radio("Vehicle type", ["Car", "Bike"], horizontal=True, key="quote_kind").lower()
options = engine.options(kind)
defaults = options["defaults"]

st.subheader("Your vehicle")
default_make, default_model, default_fuel = options.get("default_vehicle") or (options["makes"][0], None, None)
column_a, column_b, column_c = st.columns(3)
start_with(f"quote_make_{kind}", default_make, options["makes"])
make = column_a.selectbox("Make", options["makes"], key=f"quote_make_{kind}")
models = sorted(options["models"][make])
start_with(f"quote_model_{kind}_{make}", default_model if make == default_make and default_model in models else models[0], models)
model = column_b.selectbox("Model", models, key=f"quote_model_{kind}_{make}")
fuels = options["models"][make][model]
start_with(f"quote_fuel_{kind}_{make}_{model}",
           default_fuel if (make, model) == (default_make, default_model) else "Petrol" if "Petrol" in fuels else fuels[0], fuels)
fuel = column_c.selectbox("Fuel type", fuels, key=f"quote_fuel_{kind}_{make}_{model}")
column_a, column_b = st.columns(2)
newest = max(config.now_local().year, quote.PRICING_YEAR)
years = list(range(newest, options["year_min"] - 1, -1))
start_with(f"quote_year_{kind}", quote.PRICING_YEAR - 5, years)
year = column_a.selectbox("Year of manufacture", years, key=f"quote_year_{kind}",
                          help=f"Vehicle age is counted as {quote.PRICING_YEAR} minus this year, the same way as in the training data.")
cities = options["cities"]
start_with(f"quote_city_{kind}", "Mumbai" if "Mumbai" in cities else cities[0], cities)
city = column_b.selectbox("City", cities, key=f"quote_city_{kind}")
column_b.caption(f"City risk score: {options['city_risk'][city]:.2f} (higher = more claims and thefts)")

st.subheader("Your cover")
column_a, column_b, column_c = st.columns(3)
low, high = options["idv_form"]["min"], options["idv_form"]["max"]
typical = engine.typical_idv(kind, make, model, fuel, year)
idv_key = f"quote_idv_{kind}_{make}_{model}_{fuel}_{year}"
first_idv = int(min(max(chat_idv or typical or 500_000, low), high))
if chat_idv:
    state[idv_key] = first_idv               # the amount named in the chat question
else:
    start_with(idv_key, first_idv)           # otherwise the usual value for this vehicle and age
idv = column_a.number_input(f"IDV, the insured declared value ({RUPEE})", min_value=low, max_value=high, step=5_000,
                            format="%d", key=idv_key,
                            help=f"Between {inr(low)} and {inr(high)}. It starts at the usual value for this vehicle and age.")
column_a.caption(f"= {inr(idv)}" + (f"   |   usual for this vehicle: {inr(typical)}" if typical else ""))
start_with(f"quote_ncb_{kind}", options["ncb_steps"][0], options["ncb_steps"])
ncb = column_b.select_slider("No-claim bonus (NCB), %", options=options["ncb_steps"], key=f"quote_ncb_{kind}",
                             help="The discount for every claim-free year. 0 if you have no bonus yet.")
policy_types = options["policy_types"]
start_with(f"quote_policy_{kind}", defaults["policy_type"], policy_types)
policy_type = column_c.selectbox("Cover type", policy_types, key=f"quote_policy_{kind}", help="Comprehensive covers your own "
                                 "vehicle and others. Third Party covers only damage to others. Own Damage covers only your own vehicle.")

with st.expander("More details (optional: typical values are used if you skip this)"):
    ranges = options["ranges"]
    column_a, column_b, column_c, column_d = st.columns(4)
    start_with(f"quote_addons_{kind}", defaults["num_addons"])
    add_ons = column_a.slider("Add-ons", ranges["num_addons"][0], ranges["num_addons"][1], key=f"quote_addons_{kind}")
    start_with(f"quote_claims_{kind}", defaults["claim_history_count"])
    past_claims = column_b.slider("Claims in the past", ranges["claim_history_count"][0], ranges["claim_history_count"][1],
                                  key=f"quote_claims_{kind}")
    start_with(f"quote_driver_{kind}", defaults["customer_age"])
    driver_age = column_c.slider("Driver age", ranges["customer_age"][0], ranges["customer_age"][1], key=f"quote_driver_{kind}")
    usage = None
    if kind == "bike":
        usage_types = options["usage_types"]
        start_with("quote_usage", defaults["usage_type"], usage_types)
        usage = column_d.selectbox("Bike use", usage_types, key="quote_usage")

if st.button("Get my quote", type="primary"):
    try:
        result = engine.quote(kind, make, model, fuel, year, city, idv, ncb, policy_type=policy_type, num_addons=add_ons,
                              past_claims=past_claims, driver_age=driver_age, usage_type=usage)
    except Exception as exc:
        st.error(f"The quote could not be calculated ({type(exc).__name__}). Please try again.")
        st.stop()
    try:    # every quotation is saved with the time and the signed-in customer (quotations table)
        row = db.save_quotation(user["id"], vehicle_type=kind, vehicle_make=make, vehicle_model=model,
                                manufacturing_year=int(year), city=city, idv=float(idv), ncb_percent=int(ncb),
                                predicted_premium=round(result.premium, 2), fuel_type=fuel, policy_type=policy_type)
        saved = {"id": str(row.id), "at": row.created_at}
    except Exception as exc:
        saved = {"error": type(exc).__name__}
    st.session_state["last_quote"] = {"quote": result, "saved": saved,
                                      "title": f"{make} {model} ({fuel}, {year}) in {city}, IDV {inr(idv)}, NCB {ncb} %"}

if "last_quote" in st.session_state:
    show_result(st.session_state["last_quote"])

try:
    recent = db.recent_quotations(user["id"], 5)
except Exception:
    recent = []
if recent:
    st.divider()
    st.subheader("Your recent quotations")
    st.dataframe([{"When": f"{r.created_at:%d %b %Y %H:%M}", "Vehicle": f"{r.vehicle_make} {r.vehicle_model} ({r.manufacturing_year})",
                   "City": r.city, "IDV": inr(r.idv), "NCB": f"{r.ncb_percent} %", "Premium": inr(r.predicted_premium)}
                  for r in recent], hide_index=True)
