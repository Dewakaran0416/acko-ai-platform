"""Module 3 page: File a Claim.

The customer uploads the damage photo and the claim form. Gemini reads both (the photo as JSON: damage type, severity,
parts; the form: policy number, date, registration, description, and whether it is a car or a bike). The customer checks
the details, the claims models give the amount and the approval probability, and the claim is saved: the two files go to
S3 (or the local folder uploads/) and one row goes to the claims table. The logic is in acko/claim_flow.py.
"""
from __future__ import annotations

import hashlib
import uuid

import pandas as pd
import streamlit as st

from acko import claim_flow, claims, config, db, storage, vision
from acko.claim_flow import plain, probability_text
from acko.fmt import RUPEE, inr

user = st.session_state["user"]
state = st.session_state
today = config.now_local().date()
LABELS = {**claims.COLUMN_LABELS, "city": "City"}
WHICH = {"photo": "The damage photo", "form": "The claim form"}

st.title("File a Claim")
st.caption("Upload the damage photo and the claim form. Gemini reads both, you check the details, and the claims models "
           "give the claim amount and the approval probability.")


def model_stamp() -> float:
    """Changes whenever train_claims.py writes a new model file, so a running app picks it up without a restart."""
    return max((path.stat().st_mtime for path in claims.MODEL_FILES.values() if path.exists()), default=0.0)


@st.cache_resource(show_spinner="Loading the claims models (only the first time) ...")
def load_engine(stamp: float) -> claims.ClaimEngine:
    return claims.ClaimEngine.load()


@st.cache_data(ttl=60, show_spinner=False)
def storage_status() -> tuple[bool, str]:
    """Can the files be saved? Checked at most once a minute (for S3 it is one small request)."""
    try:
        return True, storage.check()
    except storage.StorageError as error:
        return False, str(error)


try:
    engine = load_engine(model_stamp())
except claims.ModelMissing as exc:
    st.error(str(exc))
    st.stop()

storage_ok, storage_text = storage_status()
if storage_ok:
    st.caption(f"Your files are kept in: {storage_text}.")
else:
    st.error(f"Files cannot be saved right now, so a claim cannot be filed. {storage_text}")
state.setdefault("claim_round", 0)


def table(rows: list[dict]) -> None:
    """A small static table. The first column is the row title. Text that comes from a file or from Gemini is shown
    exactly as it is (a table would otherwise turn [text](address) into a link)."""
    if rows:
        title = list(rows[0])[0]
        shown = [{name: value if name == title or not isinstance(value, str) else plain(value) for name, value in row.items()}
                 for row in rows]
        st.table(pd.DataFrame(shown).set_index(title))


def show_recent() -> None:
    try:
        recent = db.recent_claims(user["id"], 5)
    except Exception:
        recent = []
    if recent:
        st.divider()
        st.subheader("Your recent claims")
        st.dataframe([{"When": f"{r.created_at:%d %b, %H:%M}", "Reference": str(r.id)[:8].upper(),
                       "Vehicle": r.vehicle_type.title() + (f" ({r.vehicle_make} {r.vehicle_model})" if r.vehicle_make else ""),
                       "Damage": f"{r.damage_type} ({r.damage_severity})", "Amount": inr(r.predicted_amount),
                       "Approval": probability_text(r.approval_probability),
                       "Status": claim_flow.STATUS_TEXT.get(r.status, r.status)} for r in recent], hide_index=True)


def show_filed(done: dict) -> None:
    filed: claim_flow.Filed = done["filed"]
    estimate = filed.estimate
    values = done["values"]
    st.success(f"Your claim is filed. Reference: {filed.claim_id[:8].upper()}  ({filed.created_at:%d %b %Y at %H:%M})")
    column_a, column_b, column_c = st.columns(3)
    column_a.metric("Estimated claim amount", inr(estimate.amount))
    column_b.metric("Approval probability", probability_text(estimate.probability))
    column_c.metric("Status", claim_flow.STATUS_TEXT.get(estimate.status, estimate.status))
    threshold = f"{estimate.threshold * 100:.0f} %"
    if estimate.status == "approved":
        st.success(f"The approval probability ({probability_text(estimate.probability)}) is at or above the approval threshold "
                   f"({threshold}), so the claim is approved automatically.")
    else:
        st.warning(f"The approval probability ({probability_text(estimate.probability)}) is below the approval threshold "
                   f"({threshold}), so the claim goes to a person at Acko for review.")
    for text in estimate.notes:
        st.warning(text)
    amount_error = engine.metrics(estimate.kind, "amount")["all"]
    approval = engine.metrics(estimate.kind, "approval")["at_threshold"]
    st.caption(f"The amount is an estimate. On claims the model had never seen, its typical error was about "
               f"{amount_error['median_pct_error']:.0f} % of the claim. At the standard threshold of {approval['threshold'] * 100:.0f} %, "
               f"the approval decision matched the real one for {approval['accuracy'] * 100:.0f} % of those claims.")
    st.write("**Where your files are kept**")
    st.caption(storage.describe())
    st.code(filed.image_key + (f"\n{filed.form_key}" if filed.form_key else ""), language=None)
    if not filed.form_key:
        st.caption("No claim form was uploaded with this claim.")
    with st.expander("The details this estimate is based on"):
        summary = [("Vehicle type", estimate.kind.title()), ("Policy number", values["policy_number"].strip()),
                   ("Date of incident", f"{values['incident_date']:%d %b %Y}"),
                   ("Registration number", filed.registration or "-"), ("Damaged parts", ", ".join(values["parts"]))]
        for key, value in estimate.answers.items():
            if value is None:
                continue
            text = inr(value) if key == "idv" else ("Yes" if value else "No") if key in ("zero_dep_addon", "helmet_worn") \
                else f"{value} %" if key == "ncb_at_claim_percent" else str(value)
            summary.append((LABELS.get(key, key), text))
        table([{"Detail": label, "Value": text} for label, text in summary])


# ------------------------------------------------------------------ after a claim is filed: the result, until "File another claim"
if state.get("claim_filed"):
    show_filed(state["claim_filed"])
    if st.button("File another claim", type="primary", icon=":material/add:"):
        state.pop("claim_filed", None)
        state.pop("claim_docs", None)
        state["claim_round"] += 1
        st.rerun()
    show_recent()
    st.stop()

# ------------------------------------------------------------------ 1. the two files
st.subheader("1. Your documents")
round_no = state["claim_round"]
column_a, column_b = st.columns(2)
photo_file = column_a.file_uploader("Damage photo (JPG, PNG or WEBP)", type=["jpg", "jpeg", "png", "webp"],
                                    max_upload_size=10, key=f"claim_photo_{round_no}",
                                    help="A clear photo of the damaged vehicle, up to 10 MB. Location data inside the photo is removed.")
form_file = column_b.file_uploader("Claim form (PDF, optional)", type=["pdf"], max_upload_size=5, key=f"claim_form_{round_no}",
                                   help="Up to 5 MB. With the form, the policy number, the date of the incident and the vehicle "
                                        "type are read for you.")
photo_bytes = photo_file.getvalue() if photo_file else None
form_bytes = form_file.getvalue() if form_file else None
hashes = (hashlib.sha1(photo_bytes).hexdigest() if photo_bytes else None, hashlib.sha1(form_bytes).hexdigest() if form_bytes else None)

if photo_bytes is None:
    st.info("Upload the damage photo to start.")
if st.button("Read my documents", type="primary", disabled=photo_bytes is None, icon=":material/document_scanner:"):
    with st.spinner("Gemini is reading your photo and your form. This takes about 10 to 30 seconds ..."):
        try:
            docs = claim_flow.read_documents(photo_bytes, form_bytes)
        except Exception as exc:
            st.error(f"The documents could not be read ({type(exc).__name__}). Please try again.")
            st.stop()
    state["claim_docs"] = {"rid": uuid.uuid4().hex[:8], "hashes": hashes, "docs": docs}

reading = state.get("claim_docs")
if reading is None:
    st.stop()
if reading["hashes"] != hashes:
    st.warning("The files were changed after they were read. Press Read my documents again.")
    st.stop()
docs: claim_flow.Documents = reading["docs"]
rid = reading["rid"]

# ------------------------------------------------------------------ 2. what Gemini found
st.subheader("2. What Gemini found")
st.caption(f"Read in {docs.seconds:.0f} s. Pressing Read my documents again reads the files again and replaces what you typed below.")
for problem in docs.problems:
    st.warning(f"{WHICH.get(problem.which, 'A file')} could not be read: {problem.message}")
if docs.photo_jpeg is None:
    st.error("The claim needs a usable damage photo. Please upload another photo.")
    st.stop()
if docs.problems:
    st.caption("You can still fill in the details below by hand.")
if form_bytes and docs.form_pdf is None:
    st.caption("This claim form cannot be used, so it will not be filed with the claim.")

left, right = st.columns(2)
with left:
    st.write("**Damage photo**")
    st.image(docs.photo_jpeg, width=360, caption="The photo as it will be stored (upright, location data removed)")
    if docs.photo is not None:
        table(claim_flow.photo_rows(docs.photo))
        if not docs.photo.damage_visible:
            st.warning("Gemini could not see damage on a vehicle in this photo. Please check the photo, or fill in the damage by hand.")
        if docs.photo.summary:
            st.caption(plain(docs.photo.summary))
        for text in docs.photo.notes:
            st.caption(plain(text))
        with st.expander("Gemini's answer for the photo, as JSON"):
            st.json(docs.photo.as_json())
with right:
    st.write("**Claim form**")
    if docs.form is not None:
        table(claim_flow.form_rows(docs.form))
        if docs.form.missing:
            st.warning("Not found on the form: " + ", ".join(vision.FIELD_LABELS[name] for name in docs.form.missing) +
                       ". Please fill these in below.")
        for text in docs.form.notes:
            st.caption(plain(text))
        with st.expander("Gemini's answer for the form, as JSON"):
            st.json(claim_flow.form_json(docs.form))
    elif not form_bytes:
        st.caption("No claim form was uploaded. With a form, the policy number, the date and the vehicle type are read for you.")
    else:
        st.caption("Nothing could be taken from the form (see above).")


def start_with(key: str, value, allowed=None) -> None:
    """Give a widget its first value through session state, so the widget itself is created without index= or value=."""
    if key not in state or (allowed is not None and state[key] not in allowed):
        state[key] = value


# ------------------------------------------------------------------ 3. check the details
st.subheader("3. Check the details")
routing = docs.routing
kind_key = f"claim_kind_{rid}"
if routing is not None and routing.kind:
    start_with(kind_key, routing.kind.title(), ["Car", "Bike"])
choice = st.radio("Vehicle type (this decides which pair of claims models is used)", ["Car", "Bike"], index=None, horizontal=True,
                  key=kind_key)
if routing is not None and routing.reason:
    (st.warning if routing.needs_check else st.info)(plain(routing.reason))
if choice is None:
    st.stop()
kind = choice.lower()
if routing is not None and routing.kind and routing.kind != kind:
    st.caption(f"You chose {kind} yourself, so the {kind} models are used.")

options = engine.options(kind)
ranges, cities = options["ranges"], options["cities"]
limit = claims.MAX_PARTS[kind]
first_key = f"claim_first_{rid}_{kind}"
if first_key not in state:
    try:
        latest = next((q for q in db.recent_quotations(user["id"], 10) if q.vehicle_type == kind), None)
    except Exception:
        latest = None
    state[first_key] = claim_flow.suggest(docs, kind, options, latest, today)
first, source = state[first_key]


def field(name: str) -> str:
    return f"claim_{name}_{rid}_{kind}"


def note(where, name: str) -> None:
    text = claim_flow.SOURCE_TEXT.get(source.get(name, ""), "")
    if text:
        where.caption(text[0].upper() + text[1:])


st.write("**The policy and the vehicle**")
column_a, column_b, column_c = st.columns(3)
start_with(field("policy"), first["policy_number"])
policy_number = column_a.text_input("Policy number", key=field("policy"), max_chars=50)
note(column_a, "policy_number")
if first["incident_date"]:
    start_with(field("date"), first["incident_date"])
incident_date = column_b.date_input("Date of incident", value=None, min_value=claim_flow.EARLIEST_DATE, max_value=today,
                                    key=field("date"), format="DD/MM/YYYY")
note(column_b, "incident_date")
start_with(field("reg"), first["registration"])
registration = column_c.text_input("Vehicle registration number", key=field("reg"), max_chars=20, placeholder="for example TN09AB1234")
note(column_c, "registration")
column_a, column_b, column_c = st.columns(3)
start_with(field("make"), first["make"])
make = column_a.text_input("Vehicle make (optional)", key=field("make"), max_chars=50)
note(column_a, "make")
start_with(field("model"), first["model"])
model = column_b.text_input("Vehicle model (optional)", key=field("model"), max_chars=80)
note(column_b, "model")
start_with(field("city"), first["city"], cities)
city = column_c.selectbox("City", cities, key=field("city"))
note(column_c, "city")

st.write("**The damage**")
column_a, column_b = st.columns(2)
if first["damage_type"]:
    start_with(field("type"), first["damage_type"], claims.DAMAGE_TYPES)
damage_type = column_a.selectbox("Damage type", claims.DAMAGE_TYPES, index=None, placeholder="Choose the damage type", key=field("type"))
note(column_a, "damage_type")
start_with(field("severity"), first["severity"], list(range(1, 11)))
severity = column_b.slider("Damage severity (1 = very small, 10 = vehicle destroyed)", 1, 10, key=field("severity"))
column_b.caption(f"That is {claims.severity_label(severity)} damage." + (" Read from the photo." if source["severity"] == "photo" else
                 " Gemini could not judge the severity, so this is a middle value: please set it yourself."))
start_with(field("parts"), first["parts"])
parts = st.multiselect(f"Damaged parts (at most {limit})", vision.PARTS[kind], max_selections=limit, select_all=False,
                       key=field("parts"))
note(st, "parts")
start_with(field("desc"), first["description"])
description = st.text_area("Damage description", key=field("desc"), max_chars=claim_flow.DESCRIPTION_MAX, height=100,
                           placeholder="What happened, and what is damaged?")
note(st, "description")

st.write("**The cover and the history**")
column_a, column_b, column_c = st.columns(3)
if first["idv"] is not None:
    start_with(field("idv"), first["idv"])
idv = column_a.number_input(f"Insured value, IDV ({RUPEE})", min_value=claim_flow.IDV_MIN, max_value=claim_flow.IDV_MAX,
                            value=None, step=5_000, format="%d", key=field("idv"), placeholder="for example 500000",
                            help=f"The insured declared value of the policy: the most a claim can be paid. "
                                 f"Between {inr(claim_flow.IDV_MIN)} and {inr(claim_flow.IDV_MAX)}.")
if idv:
    column_a.caption(f"= {inr(idv)}")
note(column_a, "idv")
with st.expander("More about the policy (typical values are used for what the form does not say)"):
    column_a, column_b, column_c = st.columns(3)
    start_with(field("zero"), bool(first["zero_dep"]))
    zero_dep = column_a.checkbox("Zero-depreciation add-on", key=field("zero"), help="Tick it if the policy has the zero-depreciation add-on.")
    note(column_a, "zero_dep")
    start_with(field("previous"), first["previous_claims"])
    previous_claims = column_b.slider("Earlier claims on this policy", ranges["previous_claims_count"][0],
                                      ranges["previous_claims_count"][1], key=field("previous"))
    note(column_b, "previous_claims")
    start_with(field("age"), first["policy_age_months"])
    policy_age = column_c.number_input("Policy age at the incident (months)", min_value=0, max_value=claim_flow.POLICY_AGE_MAX,
                                       step=1, key=field("age"))
    note(column_c, "policy_age_months")
    column_a, column_b = st.columns(2)
    start_with(field("time"), first["time_of_day"], claims.TIMES_OF_DAY)
    time_of_day = column_a.selectbox("Time of day of the incident", claims.TIMES_OF_DAY, key=field("time"))
    note(column_a, "time_of_day")
    start_with(field("ncb"), first["ncb"], options["ncb_steps"])
    ncb = column_b.select_slider("No-claim bonus (NCB) at the claim, %", options=options["ncb_steps"], key=field("ncb"))
    note(column_b, "ncb")
    usage, helmet, experience = None, 0, 0
    if kind == "bike":
        column_a, column_b, column_c = st.columns(3)
        start_with(field("usage"), first["usage_type"], options["usage_types"])
        usage = column_a.selectbox("Bike use", options["usage_types"], key=field("usage"))
        note(column_a, "usage_type")
        start_with(field("helmet"), bool(first["helmet_worn"]))
        helmet = int(column_b.checkbox("Helmet worn", key=field("helmet")))
        note(column_b, "helmet_worn")
        start_with(field("experience"), first["rider_experience"])
        experience = column_c.slider("Rider experience (years)", ranges["rider_experience_years"][0],
                                     ranges["rider_experience_years"][1], key=field("experience"))
        note(column_c, "rider_experience")

threshold = None
if user["role"] == "manager":
    with st.expander("Manager: approval threshold"):
        threshold = st.number_input("Approval threshold (a claim below it goes to review)", 0.05, 0.95,
                                    float(config.APPROVAL_THRESHOLD), 0.05, format="%.2f", key="claim_threshold",
                                    help="Customers always get the default from the settings (APPROVAL_THRESHOLD in .env, 0.5).")

current = {"policy_number": policy_number, "incident_date": incident_date, "registration": registration, "make": make,
           "model": model, "city": city, "damage_type": damage_type, "severity": severity, "parts": parts,
           "description": description, "idv": idv, "zero_dep": int(zero_dep), "previous_claims": previous_claims,
           "policy_age_months": policy_age, "time_of_day": time_of_day, "ncb": ncb, "usage_type": usage,
           "helmet_worn": helmet, "rider_experience": experience}
errors, warnings = claim_flow.check_values(kind, current, options, today)
for text in warnings:
    st.warning(text)

# ------------------------------------------------------------------ 4. file the claim
st.subheader("4. File the claim")
if st.button("Submit claim and get my estimate", type="primary", disabled=not storage_ok, icon=":material/send:"):
    if errors:
        st.error("Please fix this before the claim can be filed:\n\n" + "\n".join(f"- {text}" for text in errors))
    else:
        with st.spinner("Calculating the estimate and saving your files ..."):
            try:
                filed = claim_flow.file_claim(user["id"], kind, current, docs.photo_jpeg, docs.form_pdf, engine, threshold=threshold)
            except claim_flow.FilingError as error:
                st.error(str(error))
                st.stop()
        state["claim_filed"] = {"filed": filed, "values": current}
        st.rerun()
show_recent()
