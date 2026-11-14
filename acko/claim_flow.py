"""The logic behind the File a Claim page, kept apart from the screen so that it can be tested on its own.

    read_documents()   Gemini reads the damage photo and the claim form; the form decides car or bike
    suggest()          the first values of the details the customer checks, and where each of them came from
    check_values()     what must be fixed before a claim can be filed, and what is worth a second look
    file_claim()       the estimate, then the two files to S3 (or the local folder), then the row in the claims table

Nothing in here needs Streamlit. The page (views/claims.py) only draws the screen and calls these functions.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime

from acko import claims, config, db, storage, vision
from acko.fmt import inr

IDV_MIN, IDV_MAX = 10_000, 500_000_000      # the same limits the form reader uses: a value outside them is a misreading
POLICY_AGE_MAX = 240                        # months
DESCRIPTION_MAX = 500                       # characters (the column of the claims table)
EARLIEST_DATE = date(2000, 1, 1)
POLICY_NUMBER = re.compile(r"[A-Za-z0-9][A-Za-z0-9/_.\- ]{2,49}")
STATUS_TEXT = {"approved": "Approved", "review": "Sent for review"}
SOURCE_TEXT = {"form": "read from the claim form", "photo": "read from the photo",
               "quote": "from your latest quotation, please check it", "typical": "a typical value: the form does not say"}


# ------------------------------------------------------------------------------------------------ small helpers
ZERO_WIDTH = chr(0x200B)        # a space with no width: invisible, but it stops a web address from being recognised as one


def plain(text) -> str:
    """Text that Streamlit shows exactly as it is: no links, bold, colours or formulas from the words of a file or of Gemini.
    Markdown signs are escaped. Bare web and e-mail addresses are turned into links by Markdown even when their signs are
    escaped, so an invisible zero-width space is put inside them."""
    raw = str(text or "")
    raw = re.sub(r"(?i)\b(https?|ftp)://", lambda found: f"{found.group(1)}:{ZERO_WIDTH}//", raw)
    raw = re.sub(r"(?i)\bwww\.", lambda found: found.group(0)[:-1] + ZERO_WIDTH + ".", raw)
    raw = raw.replace("@", "@" + ZERO_WIDTH)
    return re.sub(r"([!-/:-@\[-`{-~])", r"\\\1", raw)


def probability_text(probability: float) -> str:
    """0.87 -> '87 %'. The model is never shown as completely sure or completely against."""
    percent = float(probability) * 100
    if percent >= 99.5:
        return ">99 %"
    if percent <= 0.5:
        return "<1 %"
    return f"{percent:.0f} %"


def clean_registration(text) -> str:
    """'tn 09 ab-1234' -> 'TN09AB1234'."""
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())[:15]


def join_parts(parts: list[str], limit: int = 200) -> str:
    """'Front bumper, Bonnet' for the claims table, cut between two names (never in the middle of one) when it is too long."""
    text = ""
    for part in parts:
        joined = f"{text}, {part}" if text else part
        if len(joined) > limit:
            break
        text = joined
    return text


# ------------------------------------------------------------------------------------------------ 1. reading the documents
@dataclass
class Problem:
    which: str              # photo / form
    message: str            # a sentence the customer can understand
    kind: str = "other"     # file / quota / key / busy / model / format / other


@dataclass
class Documents:
    """What the two uploaded files turned into. Every part may be missing: the customer can still type the details."""
    photo_jpeg: bytes | None = None                  # the cleaned photo that is stored (JPEG, upright, no location data)
    form_pdf: bytes | None = None                    # the form exactly as uploaded (only when it is a usable PDF)
    photo: vision.PhotoReading | None = None
    form: vision.FormReading | None = None
    routing: vision.Routing | None = None
    problems: list[Problem] = field(default_factory=list)
    seconds: float = 0.0


def safe_lookup() -> vision.VehicleLookup:
    """The make and model list of the quote model. An empty list when that model is missing: the form and photo still work."""
    try:
        return vision.default_lookup()
    except Exception:
        return vision.VehicleLookup({})


def read_documents(photo: bytes | None, form: bytes | None, lookup: vision.VehicleLookup | None = None) -> Documents:
    """Check both files, let Gemini read them, and decide car or bike. A file that fails is reported in .problems and the
    other file is still used. Order: the form first (it decides car or bike, which makes the photo question more exact)."""
    started = time.monotonic()
    docs = Documents()
    lookup = lookup or safe_lookup()

    def failed(which: str, error: vision.VisionError) -> None:
        docs.problems.append(Problem(which, str(error), error.kind))

    if photo:
        try:
            docs.photo_jpeg = vision.prepare_photo(photo)
        except vision.VisionError as error:
            failed("photo", error)
    if form:
        try:
            vision.inspect_pdf(form)
            docs.form_pdf = form
            docs.form = vision.read_form(form)
        except vision.VisionError as error:
            failed("form", error)
    hint = vision.route_vehicle(docs.form, None, lookup).kind if docs.form else None
    if docs.photo_jpeg:
        try:
            docs.photo = vision.analyze_photo(docs.photo_jpeg, hint, prepared=True)
        except vision.VisionError as error:
            failed("photo", error)
    docs.routing = vision.route_vehicle(docs.form, docs.photo, lookup)
    docs.seconds = time.monotonic() - started
    return docs


# ------------------------------------------------------------------------------------------------ 2. what Gemini found (for the screen)
def _shown(key: str, value) -> str:
    if isinstance(value, date):
        return f"{value:%d %b %Y}"
    if key == "idv":
        return inr(value)
    if key in ("zero_dep", "helmet_worn"):
        return "Yes" if value else "No"
    if key == "ncb":
        return f"{value} %"
    if key == "vehicle_type":
        return str(value).title()
    return str(value)


FORM_ROWS = [("policy_number", "Policy number"), ("incident_date", "Date of incident"), ("incident_time", "Time of incident"),
             ("vehicle_registration", "Registration number"), ("vehicle_type", "Vehicle type"), ("vehicle_make", "Make"),
             ("vehicle_model", "Model"), ("city", "City"), ("idv", "Insured value (IDV)"), ("policy_start_date", "Policy start date"),
             ("previous_claims", "Earlier claims"), ("zero_dep", "Zero-depreciation add-on"), ("ncb", "No-claim bonus"),
             ("usage_type", "Bike use"), ("helmet_worn", "Helmet worn"), ("rider_experience", "Rider experience (years)"),
             ("description", "Damage description")]


def form_rows(form: vision.FormReading | None) -> list[dict]:
    """One row per detail the form gave: {'Detail': ..., 'Read from the form': ...}."""
    if form is None:
        return []
    return [{"Detail": label, "Read from the form": _shown(key, form.values[key])} for key, label in FORM_ROWS if key in form.values]


def form_json(form: vision.FormReading | None) -> dict:
    """The same details as a plain dictionary that can be shown as JSON (dates written as text)."""
    if form is None:
        return {}
    return {key: (value.isoformat() if isinstance(value, date) else value) for key, value in form.values.items()}


def photo_rows(photo: vision.PhotoReading | None) -> list[dict]:
    if photo is None:
        return []
    seen = {"car": "a car", "bike": "a bike", "none": "no vehicle", "unclear": "unclear"}.get(photo.vehicle, photo.vehicle)
    rows = [("Vehicle in the photo", seen), ("Damage visible", "Yes" if photo.damage_visible else "No")]
    if photo.damage_type:
        rows.append(("Damage type", photo.damage_type))
    if photo.severity_score:
        rows.append(("Severity", f"{photo.severity_score} of 10 ({photo.severity})"))
    if photo.parts:
        rows.append(("Damaged parts", ", ".join(photo.parts)))
    rows.append(("Gemini's confidence", photo.confidence))
    return [{"Detail": label, "Read from the photo": text} for label, text in rows]


# ------------------------------------------------------------------------------------------------ 3. the first values of the details
def suggest(docs: Documents, kind: str, options: dict, latest_quote=None, today: date | None = None) -> tuple[dict, dict]:
    """(values, source): the first value of every detail the customer checks, and where it came from
    ('form', 'photo', 'quote', 'typical' or '' for an empty box). latest_quote is the customer's newest quotation of this kind."""
    today = today or config.now_local().date()
    form = docs.form.values if docs.form else {}
    seen = docs.photo if docs.photo is not None and docs.photo.damage_visible else None
    defaults, ranges = options["defaults"], options["ranges"]
    values: dict = {}
    source: dict = {}

    for name, form_key in (("policy_number", "policy_number"), ("registration", "vehicle_registration"),
                           ("make", "vehicle_make"), ("model", "vehicle_model")):
        values[name] = str(form.get(form_key) or "")
        source[name] = "form" if values[name] else ""
    day = form.get("incident_date")
    if not isinstance(day, date) or not EARLIEST_DATE <= day <= today:
        day = None
    values["incident_date"], source["incident_date"] = day, "form" if day else ""

    values["damage_type"] = seen.damage_type if seen else None
    source["damage_type"] = "photo" if values["damage_type"] else ""
    values["severity"] = seen.severity_score if seen and seen.severity_score else 5
    source["severity"] = "photo" if seen and seen.severity_score else "typical"
    values["parts"] = [part for part in (seen.parts if seen else []) if part in vision.PARTS[kind]][:claims.MAX_PARTS[kind]]
    source["parts"] = "photo" if values["parts"] else ""

    if form.get("description"):
        values["description"], source["description"] = form["description"], "form"
    elif docs.photo is not None and docs.photo.summary:
        values["description"], source["description"] = docs.photo.summary, "photo"
    else:
        values["description"], source["description"] = "", ""
    values["description"] = values["description"][:DESCRIPTION_MAX]

    idv = form.get("idv")
    if idv is not None and IDV_MIN <= idv <= IDV_MAX:
        values["idv"], source["idv"] = int(round(idv)), "form"
    elif latest_quote is not None and latest_quote.idv and IDV_MIN <= latest_quote.idv <= IDV_MAX:
        values["idv"], source["idv"] = int(round(latest_quote.idv)), "quote"
    else:
        values["idv"], source["idv"] = None, ""

    cities = options["cities"]
    city = vision.match_city(form.get("city"), cities)
    if city:
        values["city"], source["city"] = city, "form"
    elif latest_quote is not None and latest_quote.city in cities:
        values["city"], source["city"] = latest_quote.city, "quote"
    else:
        values["city"], source["city"] = ("Mumbai" if "Mumbai" in cities else cities[0]), "typical"

    def take(name: str, form_key: str, fallback, allowed=None, low=None, high=None) -> None:
        found = form.get(form_key)
        if found is not None and (allowed is None or found in allowed):
            if low is not None:
                found = min(max(found, low), high)
            values[name], source[name] = found, "form"
        else:
            values[name], source[name] = fallback, "typical"

    take("zero_dep", "zero_dep", defaults["zero_dep_addon"], (0, 1))
    take("previous_claims", "previous_claims", defaults["previous_claims_count"], low=ranges["previous_claims_count"][0],
         high=ranges["previous_claims_count"][1])
    take("policy_age_months", "policy_age_months", defaults["policy_age_months"], low=0, high=POLICY_AGE_MAX)
    take("time_of_day", "time_of_day", defaults["incident_time_of_day"], claims.TIMES_OF_DAY)
    take("ncb", "ncb", defaults["ncb_at_claim_percent"], options["ncb_steps"])
    if kind == "bike":
        take("usage_type", "usage_type", defaults["usage_type"], options["usage_types"])
        take("helmet_worn", "helmet_worn", defaults["helmet_worn"], (0, 1))
        take("rider_experience", "rider_experience", defaults["rider_experience_years"],
             low=ranges["rider_experience_years"][0], high=ranges["rider_experience_years"][1])
    return values, source


# ------------------------------------------------------------------------------------------------ 4. checking before filing
def check_values(kind: str, values: dict, options: dict, today: date | None = None) -> tuple[list[str], list[str]]:
    """(errors, warnings). An error stops the claim from being filed; a warning is only a hint."""
    today = today or config.now_local().date()
    errors: list[str] = []
    warnings: list[str] = []
    if kind not in claims.KINDS:
        return ["Choose whether this is a car or a bike."], warnings

    number = str(values.get("policy_number") or "").strip()
    if not number:
        errors.append("Enter the policy number. It is printed on the claim form and on your policy.")
    elif not POLICY_NUMBER.fullmatch(number):
        errors.append("The policy number may only have letters, digits and the signs - / _ . and must be 3 to 50 characters long.")
    day = values.get("incident_date")
    if not isinstance(day, date):
        errors.append("Enter the date of the incident.")
    elif day > today:
        errors.append("The date of the incident cannot be in the future.")
    elif day < EARLIEST_DATE:
        errors.append("The date of the incident is too far in the past.")
    if values.get("damage_type") not in claims.DAMAGE_TYPES:
        errors.append("Choose the damage type.")
    severity = values.get("severity")
    if not isinstance(severity, int) or isinstance(severity, bool) or not 1 <= severity <= 10:
        errors.append("The damage severity must be a whole number from 1 to 10.")
    parts = values.get("parts") or []
    limit = claims.MAX_PARTS[kind]
    if not parts:
        errors.append("Choose at least one damaged part. The estimate depends on how many parts are damaged.")
    elif len(parts) > limit:
        errors.append(f"A {kind} claim can list at most {limit} damaged parts.")
    elif any(part not in vision.PARTS[kind] for part in parts):
        errors.append(f"Some of the chosen parts do not belong to a {kind}.")
    idv = values.get("idv")
    if not isinstance(idv, (int, float)) or isinstance(idv, bool) or not IDV_MIN <= idv <= IDV_MAX:
        errors.append(f"Enter the insured value (IDV) of the policy, between {inr(IDV_MIN)} and {inr(IDV_MAX)}.")
    if values.get("city") not in options["cities"]:
        errors.append("Choose the city.")

    registration = clean_registration(values.get("registration"))
    if registration and not vision.REGISTRATION.fullmatch(registration):
        warnings.append("The registration number does not look like an Indian vehicle number (for example TN09AB1234). "
                        "Please check it.")
    return errors, warnings


# ------------------------------------------------------------------------------------------------ 5. filing the claim
class FilingError(Exception):
    """The claim was NOT filed. str(error) says why in plain words; error.stage is input, estimate, storage or database."""

    def __init__(self, message: str, stage: str):
        super().__init__(message)
        self.stage = stage


@dataclass
class Filed:
    claim_id: str
    estimate: claims.ClaimEstimate
    image_key: str
    form_key: str | None
    created_at: datetime
    registration: str = ""


def _remove(keys: list[str]) -> None:
    for key in keys:
        try:
            storage.remove(key)
        except Exception:               # best effort: the claim has failed already, this only tidies up
            pass


def file_claim(user_id, kind: str, values: dict, photo_jpeg: bytes | None, form_pdf: bytes | None,
               engine: claims.ClaimEngine, *, threshold: float | None = None) -> Filed:
    """Estimate the claim, store the photo (and the form), and write the row. Either everything is saved or nothing is:
    when a later step fails, the files stored before it are deleted again. Check the values with check_values() first."""
    if not photo_jpeg:
        raise FilingError("The damage photo is missing.", "input")
    parts = list(values["parts"])
    bike = kind == "bike"
    try:
        estimate = engine.estimate(
            kind, values["damage_type"], int(values["severity"]), len(parts), float(values["idv"]),
            zero_dep=int(values["zero_dep"]), previous_claims=int(values["previous_claims"]),
            policy_age_months=int(values["policy_age_months"]), time_of_day=values["time_of_day"], ncb=int(values["ncb"]),
            city=values["city"], usage_type=values.get("usage_type") if bike else None,
            helmet_worn=int(values["helmet_worn"]) if bike else None,
            rider_experience=int(values["rider_experience"]) if bike else None, threshold=threshold)
    except (ValueError, KeyError, TypeError) as error:
        raise FilingError(f"The estimate could not be calculated ({error}).", "estimate") from error

    claim_id = uuid.uuid4()
    stored: list[str] = []
    form_key = None
    try:
        image_key = storage.put(storage.photo_key(user_id, claim_id), photo_jpeg, storage.JPEG)
        stored.append(image_key)
        if form_pdf:
            form_key = storage.put(storage.form_key(user_id, claim_id), form_pdf, storage.PDF)
            stored.append(form_key)
    except storage.StorageError as error:
        _remove(stored)
        raise FilingError(f"Your files could not be stored, so the claim was not filed. {error}", "storage") from error

    registration = clean_registration(values.get("registration"))
    description = str(values.get("description") or "").strip()[:DESCRIPTION_MAX]
    try:
        row = db.save_claim(
            user_id, id=claim_id, vehicle_type=kind, policy_number=str(values["policy_number"]).strip()[:50],
            incident_date=values["incident_date"], damage_type=values["damage_type"], affected_parts=join_parts(parts),
            damage_severity=estimate.severity, image_s3_key=image_key, form_s3_key=form_key,
            predicted_amount=round(estimate.amount, 2), approval_probability=round(estimate.probability, 4),
            status=estimate.status, city=values["city"], vehicle_make=str(values.get("make") or "").strip()[:50] or None,
            vehicle_model=str(values.get("model") or "").strip()[:80] or None, idv=float(values["idv"]),
            vehicle_registration=registration or None, damage_description=description or None)
    except Exception as error:
        _remove(stored)
        raise FilingError(f"The claim could not be saved in the database ({type(error).__name__}), so nothing was filed. "
                          "Please try again.", "database") from error
    return Filed(claim_id=str(claim_id), estimate=estimate, image_key=image_key, form_key=form_key,
                 created_at=row.created_at, registration=registration)
