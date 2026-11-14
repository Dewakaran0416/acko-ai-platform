"""Helpers shared by the Get a Quote page and the chat page.

* get_engine() / warm_up()   the trained quote models, loaded once and kept by Streamlit.
* find_details()             reads a sentence such as "quote for my 2020 Honda City in Chennai" and returns the
                             vehicle, year, city ... it could find, so the chat can fill in the quote form.
* prefill_from_question()    find_details() for the chat page (it never raises: the chat must keep working).

find_details() uses plain rules, no AI call, so it is instant and free. It only returns what it is sure about,
and the quote page shows the person what was filled in so they can check it.
"""
from __future__ import annotations

import re

import streamlit as st

from acko import config, quote
from acko.fmt import inr

# ------------------------------------------------------------------------------------------ the cached model


def model_stamp() -> float:
    """Changes whenever train_quote.py writes a new model file, so a running app picks it up without a restart."""
    return quote.MODEL_FILE.stat().st_mtime if quote.MODEL_FILE.exists() else 0.0


@st.cache_resource(show_spinner="Loading the quote model (only the first time) ...")
def _load_engine(stamp: float) -> quote.QuoteEngine:
    return quote.QuoteEngine.load()


@st.cache_resource(show_spinner="Preparing the explanation chart (only the first time) ...")
def _warm_up(stamp: float) -> bool:
    _load_engine(stamp).warm_up()
    return True


def get_engine() -> quote.QuoteEngine:
    """The trained models. Raises quote.ModelMissing when scripts.train_quote has not been run yet."""
    return _load_engine(model_stamp())


def warm_up() -> None:
    """Build the SHAP explainers now, so the first customer does not wait for them."""
    _warm_up(model_stamp())


# ------------------------------------------------------------------------------------------ reading a sentence
GENERIC = {"electric", "energy", "motocorp", "motors", "royal"}        # parts of a make name that say nothing on their own
EXTRA_ALIASES = {"Volkswagen": {"vw"}, "Mercedes-Benz": {"merc"}}
NEEDS_MAKE = {"city", "sport", "seal", "aura", "star", "venue", "shine", "safari", "compass", "classic", "grand", "super",
              "adventure", "land", "urban", "continental", "access", "metro"}   # ordinary words: only a model if the make is named too
CITY_ALIASES = {"bangalore": "Bengaluru", "bombay": "Mumbai", "madras": "Chennai", "calcutta": "Kolkata",
                "gurugram": "Gurgaon", "poona": "Pune", "new delhi": "Delhi", "cochin": "Kochi",
                "trivandrum": "Thiruvananthapuram", "vizag": "Visakhapatnam", "baroda": "Vadodara", "mysore": "Mysuru",
                "banaras": "Varanasi", "benares": "Varanasi", "trichy": "Tiruchirappalli", "pondicherry": "Puducherry",
                "calicut": "Kozhikode", "hubli": "Hubli-Dharwad", "dharwad": "Hubli-Dharwad"}
FUEL_WORDS = {"petrol": "Petrol", "gasoline": "Petrol", "diesel": "Diesel", "cng": "CNG", "lpg": "LPG",
              "electric": "Electric", "ev": "Electric", "hybrid": "Hybrid"}
BIKE_WORDS = {"bike", "bikes", "scooter", "scooty", "motorcycle", "motorbike", "two wheeler", "2 wheeler", "moped"}
CAR_WORDS = {"car", "cars", "suv", "sedan", "hatchback", "four wheeler", "4 wheeler"}
UNITS = {"lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000, "l": 100_000, "k": 1_000, "thousand": 1_000}
YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
IDV = re.compile(r"(?:idv|insured\s+(?:declared\s+)?value|worth|valued\s+at|value\s+of)\D{0,12}?(\d[\d,]*(?:\.\d+)?)\s*"
                 r"(lakhs?|lacs?|l\b|k\b|thousand)?")
NCB_AFTER = re.compile(r"(?:ncb|no[- ]claim[- ]bonus)\s*(?:of|is|at|:|=)?\s*(\d{1,2})\b")
NCB_BEFORE = re.compile(r"\b(\d{1,2})\s*(?:%|percent)\s*(?:ncb|no[- ]claim)")


def _norm(text: str) -> str:
    """Lower case, letters and digits only, one space between words: 'City e:HEV' -> 'city e hev'."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _has(text: str, phrase: str) -> bool:
    """True when the phrase appears in the text as whole words (the text is padded with a space on both sides)."""
    return f" {phrase} " in text


def _aliases(make: str) -> set[str]:
    """The ways a make is written in a sentence: 'Maruti Suzuki' -> 'maruti suzuki', 'maruti', 'suzuki'."""
    full = _norm(make)
    words = {word for word in full.split() if len(word) >= 3 and word not in GENERIC}
    return {full} | words | EXTRA_ALIASES.get(make, set())


def _find_vehicle(text: str, options: dict[str, dict]) -> dict:
    """The best matching make and model, or only a make (or only car/bike) when the model is not named."""
    full_hits, part_hits, make_hits = [], [], []     # (score, minus name length, name, kind, make, model): the biggest wins
    for kind, opts in options.items():
        for make in opts["makes"]:
            make_named = any(_has(text, alias) for alias in _aliases(make))
            if make_named:
                make_hits.append((kind, make))
            first_words: dict[str, list[str]] = {}
            for model in opts["models"][make]:
                name = _norm(model)
                squashed = name.replace(" ", "")
                allowed = make_named or (len(squashed) >= 3 and name not in NEEDS_MAKE)
                if allowed and (_has(text, name) or _has(text, squashed)):
                    full_hits.append((len(name) + (100 if make_named else 0), -len(name), name, kind, make, model))
                if " " in name:
                    first_words.setdefault(name.split()[0], []).append(model)
            for word, models in first_words.items():     # "pulsar" -> the Pulsar models of Bajaj, "r15" -> the R15 models
                code_like = len(word) >= 3 and any(c.isdigit() for c in word) and any(c.isalpha() for c in word)
                if _has(text, word) and (make_named or code_like or (len(word) >= 5 and word not in NEEDS_MAKE)):
                    shortest = min(models, key=lambda m: (len(m), m))
                    part_hits.append((len(word) + (100 if make_named else 0) - 50, -len(shortest), _norm(shortest),
                                      kind, make, shortest))
    hits = full_hits or part_hits
    if hits:
        _, _, _, kind, make, model = max(hits)
        return {"kind": kind, "make": make, "model": model}
    words_bike = any(_has(text, word) for word in BIKE_WORDS)
    words_car = any(_has(text, word) for word in CAR_WORDS)
    kind_hint = "bike" if words_bike and not words_car else "car" if words_car and not words_bike else None
    named = [(k, m) for k, m in make_hits if kind_hint in (None, k)]
    if len(named) == 1:                                   # one make, and it exists only as a car or only as a bike
        return {"kind": named[0][0], "make": named[0][1]}
    if kind_hint:
        return {"kind": kind_hint}
    return {}


def _find_city(text: str, cities: list[str]) -> str | None:
    """The longest city name in the sentence, also by its old or short name (Bombay, Bangalore ...)."""
    by_name = {_norm(city): city for city in cities}
    by_name.update({alias: city for alias, city in CITY_ALIASES.items() if city in cities})
    found = [name for name in by_name if _has(text, name)]
    return by_name[max(found, key=len)] if found else None


def _find_amount(raw: str, low: int, high: int) -> int | None:
    """'IDV 8 lakh' / 'worth Rs 6,50,000' -> rupees, when it is inside the range the form accepts."""
    match = IDV.search(raw)
    if not match:
        return None
    number = float(match.group(1).replace(",", "")) * UNITS.get(match.group(2) or "", 1)
    return int(number) if low <= number <= high else None


def find_details(question: str, options: dict[str, dict]) -> dict:
    """Pick what the quote form asks for out of a sentence. Returns only the keys it found.

    options is {"car": engine.options("car"), "bike": engine.options("bike")}. Possible keys: kind, make, model, fuel,
    year, city, idv, ncb, policy_type. An empty dict means the sentence has no vehicle details.
    """
    raw = question.lower()
    text = f" {_norm(question)} "
    details = _find_vehicle(text, options)
    kind = details.get("kind")
    opts = options.get(kind) if kind else next(iter(options.values()))
    if kind and details.get("model"):
        fuels = opts["models"][details["make"]][details["model"]]
        wanted = {FUEL_WORDS[word] for word in text.split() if word in FUEL_WORDS} & set(fuels)
        if len(fuels) == 1:
            details["fuel"] = fuels[0]
        elif len(wanted) == 1:
            details["fuel"] = wanted.pop()
    newest = max(config.now_local().year, quote.PRICING_YEAR)
    years = [int(y) for y in YEAR.findall(raw) if opts["year_min"] <= int(y) <= newest]
    if years:
        details["year"] = years[0]
    city = _find_city(text, opts["cities"])
    if city:
        details["city"] = city
    if kind:
        amount = _find_amount(raw, opts["idv_form"]["min"], opts["idv_form"]["max"])
        if amount:
            details["idv"] = amount
    for pattern in (NCB_AFTER, NCB_BEFORE):
        match = pattern.search(raw)
        if match and int(match.group(1)) in opts["ncb_steps"]:
            details["ncb"] = int(match.group(1))
            break
    policies = [policy for policy in opts["policy_types"] if _has(text, _norm(policy))]
    if len(policies) == 1:
        details["policy_type"] = policies[0]
    return details


def describe(details: dict) -> str:
    """A short line for the person: 'Honda City (Petrol), 2020, Chennai, IDV Rs 6,00,000, NCB 25 %'."""
    vehicle = " ".join(part for part in (details.get("make"), details.get("model")) if part)
    if details.get("model") and details.get("fuel"):
        vehicle += f" ({details['fuel']})"
    elif details.get("kind") and not details.get("model"):
        vehicle = f"{vehicle} ({details['kind']})" if vehicle else details["kind"]
    parts = [vehicle, str(details.get("year") or ""), details.get("city") or "",
             f"IDV {inr(details['idv'])}" if details.get("idv") else "",
             f"NCB {details['ncb']} %" if "ncb" in details else "", details.get("policy_type") or ""]
    return ", ".join(part for part in parts if part)


def prefill_from_question(question: str) -> dict | None:
    """{"details": ..., "text": ...} for the chat page, or None when nothing was found (or the model is not trained yet)."""
    try:
        engine = get_engine()
        details = find_details(question, {kind: engine.options(kind) for kind in quote.KINDS})
    except Exception:
        return None
    return {"details": details, "text": describe(details)} if details else None
