"""Module 2: the premium quote model.

Everything the training script (scripts/train_quote.py), the notebook and the web page (views/quote.py) share:

  QuoteFeatures   feature engineering: turns the simple answers of the quote form into the numbers the model learns from
  build_model     QuoteFeatures + gradient boosting that learns log(premium)
  evaluate        R2 and RMSE on the held-out test rows (overall, inside the form's IDV range, per IDV band)
  QuoteEngine     loads models/quote_model.pkl, gives one quote and explains it with SHAP (in rupees)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from acko import config
from acko.fmt import inr

KINDS = ("car", "bike")
PRICING_YEAR = 2025        # the data was priced in 2025: vehicle_age_years = 2025 - year of manufacture
BRIEF_IDV_MIN = 50_000     # the brief: IDV from Rs 50,000 ...
FORM_IDV_MAX = 1_000_000   # ... to Rs 10,00,000
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ROUNDS = 1000          # boosting rounds at most; training stops earlier by itself when more rounds stop helping
BUNDLE_FORMAT = 1
MODEL_FILE = config.MODEL_DIR / "quote_model.pkl"

# The form answers the model needs (the first seven are the brief's form fields, the rest are optional with typical defaults).
# The three LOOKUP columns are only used to build look-up tables while training.
LOOKUP_COLUMNS = ["city_risk_score", "engine_cc", "segment"]


def input_columns(kind: str) -> list[str]:
    columns = ["vehicle_make", "vehicle_model", "fuel_type", "manufacturing_year", "city", "idv", "ncb_percent",
               "policy_type", "num_addons", "claim_history_count", "customer_age"]
    return columns + (["usage_type"] if kind == "bike" else [])


def feature_names(kind: str) -> list[str]:
    """The numbers the model sees, in a fixed order."""
    names = ["idv", "city_risk_score", "vehicle_age_years", "ncb_percent", "engine_cc", "policy_type", "num_addons",
             "claim_history_count", "customer_age", "fuel_type", "segment"]
    return names + (["usage_type"] if kind == "bike" else [])


COLUMN_LABELS = {
    "idv": "Insured value (IDV)", "city_risk_score": "City risk", "vehicle_age_years": "Vehicle age",
    "ncb_percent": "No-claim bonus (NCB)", "engine_cc": "Engine size", "policy_type": "Cover type",
    "num_addons": "Add-ons", "claim_history_count": "Past claims", "customer_age": "Driver age",
    "fuel_type": "Fuel type", "segment": "Body type", "usage_type": "Bike use",
}
# What the customer sees: model columns that describe the same thing are added up into one factor.
FACTORS = [
    ("idv", "Insured value (IDV)", ["idv"]),
    ("city", "City risk", ["city_risk_score"]),
    ("age", "Vehicle age", ["vehicle_age_years"]),
    ("ncb", "No-claim bonus (NCB)", ["ncb_percent"]),
    ("vehicle", "Vehicle type", ["engine_cc", "segment", "fuel_type"]),
    ("cover", "Cover type", ["policy_type"]),
    ("addons", "Add-ons", ["num_addons"]),
    ("claims", "Past claims", ["claim_history_count"]),
    ("driver", "Driver age", ["customer_age"]),
    ("usage", "Bike use", ["usage_type"]),
]


class ModelMissing(RuntimeError):
    """The quote model file is not there (or cannot be read); the message says how to create it."""


# --------------------------------------------------------------------------- 1. feature engineering
def _most_common(values: pd.Series):
    return values.mode().iat[0]


class QuoteFeatures(BaseEstimator, TransformerMixin):
    """Turns the answers of the quote form into the numbers the model learns from.

    fit() learns small look-up tables from the training rows:
      city               -> city risk score
      make+model+fuel    -> engine size and body type (these never change inside one make/model/fuel)
      every category     -> a code number (0, 1, 2, ...)
      typical answers    -> used when the customer skips an optional question
    transform() then needs only the simple answers: vehicle age = 2025 - year of manufacture, city risk and
    engine size come from the tables, categories become code numbers, the rest is passed on unchanged.
    """

    def __init__(self, kind: str = "car"):
        self.kind = kind

    def _category_columns(self) -> list[str]:
        return ["policy_type", "fuel_type", "segment"] + (["usage_type"] if self.kind == "bike" else [])

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.city_risk_ = X.groupby("city")["city_risk_score"].median().to_dict()
        self.risk_default_ = float(X["city_risk_score"].median())
        by_model_fuel = X.groupby(["vehicle_make", "vehicle_model", "fuel_type"])
        by_model = X.groupby(["vehicle_make", "vehicle_model"])
        self.engine_ = by_model_fuel["engine_cc"].median().to_dict()
        self.engine_by_model_ = by_model["engine_cc"].median().to_dict()
        self.engine_default_ = float(X["engine_cc"].median())
        self.segment_ = by_model_fuel["segment"].agg(_most_common).to_dict()
        self.segment_by_model_ = by_model["segment"].agg(_most_common).to_dict()
        self.segment_default_ = _most_common(X["segment"])
        self.categories_ = {column: sorted(X[column].dropna().unique().tolist()) for column in self._category_columns()}
        self.codes_ = {column: {value: number for number, value in enumerate(values)}
                       for column, values in self.categories_.items()}
        self.defaults_ = {"policy_type": _most_common(X["policy_type"]), "num_addons": 0, "claim_history_count": 0,
                          "customer_age": int(round(X["customer_age"].median()))}
        if self.kind == "bike":
            self.defaults_["usage_type"] = _most_common(X["usage_type"])
        return self

    def transform(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for column, value in self.defaults_.items():            # optional answers the customer skipped
            if column not in X.columns:
                X[column] = value
        keys = list(zip(X["vehicle_make"], X["vehicle_model"], X["fuel_type"]))
        engine = [self.engine_.get(k, self.engine_by_model_.get(k[:2], self.engine_default_)) for k in keys]
        segment = pd.Series([self.segment_.get(k, self.segment_by_model_.get(k[:2], self.segment_default_))
                             for k in keys], index=X.index)
        out = pd.DataFrame(index=X.index)
        out["idv"] = X["idv"].astype(float)
        out["city_risk_score"] = X["city"].map(self.city_risk_).fillna(self.risk_default_).astype(float)
        out["vehicle_age_years"] = (PRICING_YEAR - X["manufacturing_year"].astype(float)).clip(lower=0)
        out["ncb_percent"] = X["ncb_percent"].astype(float)
        out["engine_cc"] = np.asarray(engine, dtype=float)
        out["num_addons"] = X["num_addons"].astype(float)
        out["claim_history_count"] = X["claim_history_count"].astype(float)
        out["customer_age"] = X["customer_age"].astype(float)
        for column in self._category_columns():
            values = segment if column == "segment" else X[column]
            out[column] = values.map(self.codes_[column]).fillna(-1).astype(float)    # unknown value -> -1
        return out[feature_names(self.kind)]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(feature_names(self.kind), dtype=object)

    def decode(self, column: str, code: float) -> str:
        """Code number back to the word (for explanations)."""
        values = self.categories_[column]
        return str(values[int(code)]) if 0 <= int(code) < len(values) else "unknown"


def build_model(kind: str) -> TransformedTargetRegressor:
    """Feature engineering + gradient boosting. The model learns log(premium), predict() gives rupees again.

    Learning the logarithm fits a price that grows by percentages (more IDV = a higher premium in proportion),
    and it makes the SHAP explanation add up as 'percent steps' that multiply: IDV +30%, city +5%, NCB -20% ...
    """
    steps = [("features", QuoteFeatures(kind)),
             ("model", HistGradientBoostingRegressor(max_iter=MAX_ROUNDS, learning_rate=0.1, early_stopping=True,
                                                     random_state=0))]
    return TransformedTargetRegressor(regressor=Pipeline(steps), func=np.log, inverse_func=np.exp, check_inverse=False)


def build_linear_baseline(kind: str) -> TransformedTargetRegressor:
    """A fair 'straight-line' comparison model on the same features (used only to show why boosting was chosen)."""
    categories = QuoteFeatures(kind)._category_columns()
    numbers = [name for name in feature_names(kind) if name not in categories]
    prepare = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore"), categories),
                                 ("numbers", StandardScaler(), numbers)])
    steps = [("features", QuoteFeatures(kind)), ("prepare", prepare), ("model", Ridge(alpha=1.0))]
    return TransformedTargetRegressor(regressor=Pipeline(steps), func=np.log, inverse_func=np.exp, check_inverse=False)


# --------------------------------------------------------------------------- 2. data, split, evaluation
def raw_path(kind: str) -> Path:
    return config.RAW_DIR / config.RAW_FILES[f"{kind}_quote"]


def load_training_data(kind: str, path: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    """Read the quotation CSV, keep the columns we need, drop repeated ids and incomplete rows."""
    path = Path(path) if path else raw_path(kind)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Finish step 3 first (python -m scripts.import_data).")
    wanted = ["record_id"] + input_columns(kind) + LOOKUP_COLUMNS + ["annual_premium"]
    df = pd.read_csv(path, usecols=lambda column: column in wanted)
    absent = [column for column in wanted if column not in df.columns]
    if absent:
        raise ValueError(f"{path.name} has no column named {', '.join(absent)}.")
    rows_read = len(df)
    df = df.drop_duplicates("record_id")
    repeated = rows_read - len(df)
    before = len(df)
    df = df.dropna(subset=wanted)
    df = df[df["annual_premium"] > 0]
    info = {"file": path.name, "rows_read": rows_read, "repeated_ids_dropped": repeated,
            "incomplete_rows_dropped": before - len(df), "rows_used": len(df)}
    return df.reset_index(drop=True), info


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80 percent to learn from, 20 percent held back. The held-back rows are only used to measure the result."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def xy(df: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.Series]:
    return df[input_columns(kind) + LOOKUP_COLUMNS], df["annual_premium"]


def score(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {"n": int(len(y_true)),
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "median_pct_error": float(100 * np.median(np.abs(y_pred / y_true - 1)))}


IDV_BANDS = [(0, 100_000, "up to 1 lakh"), (100_000, 300_000, "1 to 3 lakh"), (300_000, 600_000, "3 to 6 lakh"),
             (600_000, 1_000_000, "6 to 10 lakh"), (1_000_000, 2_500_000, "10 to 25 lakh"),
             (2_500_000, 10 ** 12, "above 25 lakh")]


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """R2 / RMSE on the held-out rows: all rows, rows inside the form's IDV range, and per IDV band."""
    predicted = np.asarray(model.predict(X_test), dtype=float)
    actual = np.asarray(y_test, dtype=float)
    idv = X_test["idv"].to_numpy(dtype=float)
    in_range = idv <= FORM_IDV_MAX
    result = {"all": score(actual, predicted), "form_range": score(actual[in_range], predicted[in_range]), "bands": []}
    for low, high, label in IDV_BANDS:
        rows = (idv > low) & (idv <= high)
        if rows.sum() >= 30:
            result["bands"].append({"label": label, **score(actual[rows], predicted[rows])})
    return result


def compare_baselines(kind: str, X_train, y_train, X_test, y_test, model) -> list[dict]:
    """Same test rows (form range only), three models: a plain guess, straight-line rules, and ours."""
    in_range = (X_test["idv"] <= FORM_IDV_MAX).to_numpy()
    actual = np.asarray(y_test, dtype=float)[in_range]
    guess = np.full(len(actual), float(np.median(y_train)))
    linear = build_linear_baseline(kind).fit(X_train, y_train)
    return [
        {"name": "Guess the median premium for everyone", **score(actual, guess)},
        {"name": "Linear regression (straight-line rules)", **score(actual, linear.predict(X_test)[in_range])},
        {"name": "Gradient boosting (this project's model)", **score(actual, model.predict(X_test)[in_range])},
    ]


# --------------------------------------------------------------------------- 3. what the form offers
def build_options(df: pd.DataFrame, kind: str) -> dict:
    """Everything the form needs to know about the data: makes, models, fuels, cities, IDV hints, typical answers."""
    models: dict = {}
    for (make, model), group in df.groupby(["vehicle_make", "vehicle_model"]):
        models.setdefault(make, {})[model] = sorted(group["fuel_type"].unique().tolist())
    risk = df.groupby("city")["city_risk_score"].median()
    age = (PRICING_YEAR - df["manufacturing_year"]).clip(lower=0)
    idv_by_model_age = df.assign(age=age).groupby(["vehicle_make", "vehicle_model", "fuel_type", "age"])["idv"].median()
    idv_by_model = df.groupby(["vehicle_make", "vehicle_model", "fuel_type"])["idv"].median()
    smallest = int(df["idv"].min())
    common = df[df["idv"] <= FORM_IDV_MAX].groupby(["vehicle_make", "vehicle_model", "fuel_type"]).size().sort_values(ascending=False)
    return {
        "makes": sorted(models),
        "models": models,                                           # {make: {model: [fuel, ...]}}
        "default_vehicle": list(common.index[0]),                   # the form opens on the most common vehicle inside the IDV limit
        "cities": sorted(risk.index.tolist()),
        "city_risk": {city: float(value) for city, value in risk.items()},
        "policy_types": sorted(df["policy_type"].unique().tolist()),
        "usage_types": sorted(df["usage_type"].unique().tolist()) if kind == "bike" else [],
        "ncb_steps": sorted(int(v) for v in df["ncb_percent"].unique()),
        "year_min": int(df["manufacturing_year"].min()),
        "idv_seen": {"min": smallest, "max": int(df["idv"].max())},
        # the form accepts everything the brief allows AND everything the data contains (bikes start far below Rs 50,000)
        "idv_form": {"min": int(min(BRIEF_IDV_MIN, (smallest // 5000) * 5000)), "max": FORM_IDV_MAX},
        "ranges": {"num_addons": [0, int(df["num_addons"].max())], "claim_history_count": [0, int(df["claim_history_count"].max())],
                   "customer_age": [int(df["customer_age"].min()), int(df["customer_age"].max())]},
        "defaults": {"policy_type": _most_common(df["policy_type"]), "num_addons": 0, "claim_history_count": 0,
                     "customer_age": int(round(df["customer_age"].median())),
                     "usage_type": _most_common(df["usage_type"]) if kind == "bike" else None},
        "typical_idv": {k: float(v) for k, v in idv_by_model_age.items()},
        "typical_idv_model": {k: float(v) for k, v in idv_by_model.items()},
        "premium_median": float(df["annual_premium"].median()),
    }


# --------------------------------------------------------------------------- 4. SHAP
def _parts(model):
    """The fitted feature step and the fitted gradient boosting model inside a trained TransformedTargetRegressor."""
    pipeline = model.regressor_
    return pipeline.named_steps["features"], pipeline.named_steps["model"]


def make_explainer(model):
    import shap          # imported here because it takes a few seconds to load
    return shap.TreeExplainer(_parts(model)[1])


def shap_matrix(model, X: pd.DataFrame, explainer=None):
    """Returns (feature table, SHAP values, base value). SHAP values are in log-premium units, so
    log(premium) = base + sum of the SHAP values of one row, exactly."""
    features, _ = _parts(model)
    table = features.transform(X)
    explainer = explainer or make_explainer(model)
    values = np.asarray(explainer.shap_values(table))
    base = float(np.ravel(explainer.expected_value)[0])
    return table, values, base


def log_mean(a: float, b: float) -> float:
    """(a - b) / ln(a / b): turns log-size effects into rupees so that they add up to exactly a - b."""
    if abs(a - b) < 1e-9 * max(abs(a), 1.0):
        return float(a)
    return float((a - b) / np.log(a / b))


@dataclass
class Factor:
    key: str
    label: str
    detail: str
    rupees: float          # how many rupees this factor added to (+) or took off (-) the typical premium


@dataclass
class Quote:
    kind: str
    premium: float
    typical: float         # the typical premium of all policies in the training data: where the chart starts
    factors: list[Factor]  # biggest effect first; they add up to premium - typical
    notes: list[str] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)


def _detail(key: str, row: pd.Series, features: QuoteFeatures, frame_row: pd.Series) -> str:
    plural = lambda n, word: f"{int(n)} {word}" + ("" if int(n) == 1 else "s")
    if key == "idv":
        return inr(row["idv"])
    if key == "city":
        return f"{frame_row['city']} (risk score {row['city_risk_score']:.2f})"
    if key == "age":
        return f"{plural(row['vehicle_age_years'], 'year')} old (made in {int(frame_row['manufacturing_year'])})"
    if key == "ncb":
        return f"{int(row['ncb_percent'])}% bonus"
    if key == "vehicle":
        body = features.decode("segment", row["segment"]).replace("_", " ")
        engine = f"{int(row['engine_cc'])} cc" if row["engine_cc"] > 0 else "electric motor"
        return f"{body}, {engine}, {features.decode('fuel_type', row['fuel_type'])}"
    if key == "cover":
        return features.decode("policy_type", row["policy_type"])
    if key == "addons":
        return plural(row["num_addons"], "add-on")
    if key == "claims":
        return plural(row["claim_history_count"], "past claim")
    if key == "driver":
        return f"age {int(row['customer_age'])}"
    if key == "usage":
        return features.decode("usage_type", row["usage_type"])
    return ""


def explain_frame(model, kind: str, frame: pd.DataFrame, explainer=None, idv_seen: dict | None = None) -> Quote:
    """One quote plus its explanation, from a one-row table of form answers."""
    frame = frame.iloc[[0]]
    features, _ = _parts(model)
    premium = float(model.predict(frame)[0])
    table, values, base = shap_matrix(model, frame, explainer)
    typical = float(np.exp(base))
    scale = log_mean(premium, typical)
    effect = dict(zip(table.columns, values[0] * scale))      # rupees per model column
    row, frame_row = table.iloc[0], frame.iloc[0]
    factors = []
    for key, label, columns in FACTORS:
        columns = [c for c in columns if c in effect]
        if columns:
            factors.append(Factor(key, label, _detail(key, row, features, frame_row), float(sum(effect[c] for c in columns))))
    factors.sort(key=lambda f: abs(f.rupees), reverse=True)
    notes = []
    if idv_seen and row["idv"] < idv_seen["min"]:
        notes.append(f"This IDV is lower than any {kind} in the training data (lowest: {inr(idv_seen['min'])}), "
                     "so the estimate is less reliable.")
    if idv_seen and row["idv"] > FORM_IDV_MAX:
        notes.append(f"This IDV is above {inr(FORM_IDV_MAX)}; the model is checked only up to that value.")
    if row["vehicle_age_years"] > 20:
        notes.append("The data covers vehicles up to 20 years old; this one is priced like a 20-year-old vehicle.")
    return Quote(kind=kind, premium=premium, typical=typical, factors=factors, notes=notes,
                 inputs={"table": row.to_dict(), "answers": frame_row.to_dict()})


def grouped_effects(model, X: pd.DataFrame, explainer=None):
    """For the 'which factors matter most' picture: average percent change each factor causes, plus raw SHAP data."""
    table, values, base = shap_matrix(model, X, explainer)
    columns = list(table.columns)
    result = []
    for key, label, names in FACTORS:
        names = [c for c in names if c in columns]
        if names:
            per_row = values[:, [columns.index(c) for c in names]].sum(axis=1)
            result.append((label, float(100 * np.mean(np.abs(np.exp(per_row) - 1)))))
    return sorted(result, key=lambda item: item[1], reverse=True), table, values


# --------------------------------------------------------------------------- 5. pictures for the report (matplotlib)
RED, GREEN, GREY = "#c0392b", "#1e8449", "#566573"


def save_waterfall_png(quote: Quote, path: Path, title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Typical policy"] + [f.label for f in quote.factors] + ["Your quote"]
    figure, axis = plt.subplots(figsize=(10, 5.2))
    running = quote.typical
    axis.bar(0, running, color=GREY)
    axis.text(0, running, inr(running), ha="center", va="bottom", fontsize=8)
    for position, factor in enumerate(quote.factors, start=1):
        bottom = running if factor.rupees >= 0 else running + factor.rupees
        axis.bar(position, abs(factor.rupees), bottom=bottom, color=RED if factor.rupees >= 0 else GREEN)
        axis.text(position, max(running, running + factor.rupees), f"{'+' if factor.rupees >= 0 else '-'}{inr(abs(factor.rupees))}",
                  ha="center", va="bottom", fontsize=8)
        running += factor.rupees
    axis.bar(len(labels) - 1, quote.premium, color=GREY)
    axis.text(len(labels) - 1, quote.premium, inr(quote.premium), ha="center", va="bottom", fontsize=8)
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel("Annual premium (Rs)")
    axis.set_title(title, fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_shap_pictures(kind: str, model, sample: pd.DataFrame, out_dir: Path, example: Quote | None = None) -> list[Path]:
    """Three pictures per vehicle type: dot plot of every column, bar chart per factor, one worked example."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    out_dir.mkdir(parents=True, exist_ok=True)
    effects, table, values = grouped_effects(model, sample)
    files = []
    shap.summary_plot(values, table, feature_names=[COLUMN_LABELS[c] for c in table.columns], show=False,
                      max_display=len(table.columns))
    figure = plt.gcf()
    figure.set_size_inches(9, 5.5)
    plt.title(f"{kind.capitalize()} premium: how each model column pushes the price (log scale, each dot = one policy)", fontsize=9)
    path = out_dir / f"quote_{kind}_summary.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    files.append(path)

    figure, axis = plt.subplots(figsize=(9, 4.6))
    labels = [label for label, _ in effects][::-1]
    axis.barh(labels, [value for _, value in effects][::-1], color=GREY)
    for position, (_, value) in enumerate(effects[::-1]):
        axis.text(value, position, f" {value:.1f}%", va="center", fontsize=8)
    axis.set_xlabel("Average change of the premium caused by this factor (%)")
    axis.set_title(f"{kind.capitalize()} premium: which factors matter most", fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    path = out_dir / f"quote_{kind}_importance.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    files.append(path)

    if example is not None:
        files.append(save_waterfall_png(example, out_dir / f"quote_{kind}_example.png",
                                        f"One {kind} quote explained in rupees: {inr(example.premium)}"))
    return files


# --------------------------------------------------------------------------- 6. the saved bundle and the engine
def save_bundle(bundle: dict, path: Path | str = MODEL_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: Path | str | None = None) -> dict:
    path = Path(path) if path else MODEL_FILE
    if not path.exists():
        raise ModelMissing(f"The quote model file {path.name} was not found. Create it with:  python -m scripts.train_quote")
    try:
        bundle = joblib.load(path)
    except Exception as exc:          # a file from other library versions, or a broken download
        raise ModelMissing(f"The quote model file could not be read ({type(exc).__name__}). "
                           "Create it again with:  python -m scripts.train_quote") from exc
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise ModelMissing("The quote model file is from an older version. Create it again with:  python -m scripts.train_quote")
    return bundle


class QuoteEngine:
    """The trained car and bike models plus everything the form needs. Used by the web page."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self._explainers: dict = {}

    @classmethod
    def load(cls, path: Path | str | None = None) -> "QuoteEngine":
        return cls(load_bundle(path))

    def options(self, kind: str) -> dict:
        return self.bundle["options"][kind]

    def metrics(self, kind: str) -> dict:
        return self.bundle["metrics"][kind]

    def explainer(self, kind: str):
        if kind not in self._explainers:
            self._explainers[kind] = make_explainer(self.bundle["models"][kind])
        return self._explainers[kind]

    def warm_up(self) -> None:
        """Load SHAP and build both explainers now, so the first customer does not wait for it."""
        for kind in KINDS:
            self.explainer(kind)

    def typical_idv(self, kind: str, make: str, model: str, fuel: str, year: int) -> int | None:
        """The usual IDV of this make/model/fuel at this age in the data (a starting value for the form)."""
        options = self.options(kind)
        age = int(min(max(PRICING_YEAR - year, 1), 20))
        value = options["typical_idv"].get((make, model, fuel, age)) or options["typical_idv_model"].get((make, model, fuel))
        return int(round(value, -3)) if value else None

    def quote(self, kind: str, make: str, model: str, fuel: str, year: int, city: str, idv: float, ncb: int,
              policy_type: str | None = None, num_addons: int | None = None, past_claims: int | None = None,
              driver_age: int | None = None, usage_type: str | None = None) -> Quote:
        if kind not in KINDS:
            raise ValueError("kind must be 'car' or 'bike'")
        if idv <= 0:
            raise ValueError("IDV must be more than zero")
        answers = {"vehicle_make": make, "vehicle_model": model, "fuel_type": fuel, "manufacturing_year": int(year),
                   "city": city, "idv": float(idv), "ncb_percent": int(ncb), "policy_type": policy_type,
                   "num_addons": num_addons, "claim_history_count": past_claims, "customer_age": driver_age,
                   "usage_type": usage_type if kind == "bike" else None}
        frame = pd.DataFrame([{k: v for k, v in answers.items() if v is not None}])   # skipped answers get typical defaults
        return explain_frame(self.bundle["models"][kind], kind, frame, self.explainer(kind), self.options(kind)["idv_seen"])
