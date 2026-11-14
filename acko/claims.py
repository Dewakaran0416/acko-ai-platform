"""Module 3: the four claims models.

Everything the training script (scripts/train_claims.py), the notebooks and the claim page share:

  ClaimFeatures        turns the details of one claim into the numbers the models learn from
  ClaimAmountModel     car or bike: predicts the claim amount in rupees (it learns amount / IDV)
  ClaimApprovalModel   car or bike: predicts the probability that the claim is approved
  evaluate_amount, evaluate_approval   the numbers for the report (held-out test rows)
  ClaimEngine          loads the four .pkl files and answers for one claim: amount, approval probability, status

The four models are: car amount, car approval, bike amount, bike approval. Each one is saved in its own file
(models/claim_car_amount.pkl and so on), as the brief asks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from acko import config
from acko.fmt import inr

KINDS = ("car", "bike")
TASKS = ("amount", "approval")
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ROUNDS = 1000          # boosting rounds at most; training stops earlier by itself when more rounds stop helping
BUNDLE_FORMAT = 1
MODEL_FILES = {(kind, task): config.MODEL_DIR / f"claim_{kind}_{task}.pkl" for kind in KINDS for task in TASKS}

# What the brief asks (Module 3): classification accuracy and F1 above 80 %. For the claim amount it only says "an acceptable
# RMSE", so this project sets its own bar: R2 above 0.85 and a typical (median) error below 15 % of the claim.
TARGET_ACCURACY = 0.80
TARGET_F1 = 0.80
TARGET_R2 = 0.85
TARGET_MEDIAN_ERROR = 15.0

# The words the data uses (the vision step must answer with exactly these)
DAMAGE_TYPES = ["bumper damage", "collision damage", "crack", "dent", "fire damage", "flood damage", "panel damage",
                "partial loss", "rollover", "scratch", "theft", "total loss", "windshield damage"]
TIMES_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]
USAGE_TYPES = ["Personal", "Commercial", "Delivery"]
SEVERITY_BANDS = [(1, 3, "minor"), (4, 6, "moderate"), (7, 10, "major")]    # damage_severity_score 1-10 -> the brief's words
MAX_PARTS = {"car": 6, "bike": 5}                                           # the most parts one claim lists in the data

# The details of a claim that each model needs. The first five are enough for the amount; the approval models also use the
# claim history, the policy age, the time of day, the NCB and the city (looked up as a city risk score).
AMOUNT_COLUMNS = ["damage_type", "damage_severity_score", "num_parts_affected", "idv", "zero_dep_addon"]
APPROVAL_COLUMNS = AMOUNT_COLUMNS + ["previous_claims_count", "policy_age_months", "incident_time_of_day",
                                     "ncb_at_claim_percent", "city"]
BIKE_EXTRA_COLUMNS = ["usage_type", "helmet_worn", "rider_experience_years"]
CATEGORY_COLUMNS = ["damage_type", "incident_time_of_day", "usage_type"]
YES_NO_COLUMNS = ["zero_dep_addon", "helmet_worn"]          # 0 or 1: a skipped answer gets the more common one
ALWAYS_NEEDED = ["damage_type", "damage_severity_score", "num_parts_affected", "idv"]

COLUMN_LABELS = {
    "damage_type": "Damage type", "damage_severity_score": "Damage severity (1-10)", "num_parts_affected": "Parts damaged",
    "idv": "Insured value (IDV)", "zero_dep_addon": "Zero-depreciation add-on", "previous_claims_count": "Earlier claims",
    "policy_age_months": "Policy age (months)", "incident_time_of_day": "Time of day", "ncb_at_claim_percent": "NCB at claim",
    "city_risk_score": "City risk", "usage_type": "Bike use", "helmet_worn": "Helmet worn",
    "rider_experience_years": "Rider experience (years)",
}


def severity_label(score: float) -> str:
    """damage_severity_score (1 to 10) -> minor / moderate / major."""
    for low, high, label in SEVERITY_BANDS:
        if low <= round(score) <= high:
            return label
    return "major" if score > 10 else "minor"


def input_columns(kind: str, task: str) -> list[str]:
    """The details a claim must give for this model (the city is turned into a city risk score inside the model)."""
    if task == "amount":
        return list(AMOUNT_COLUMNS)
    return APPROVAL_COLUMNS + (BIKE_EXTRA_COLUMNS if kind == "bike" else [])


def lookup_columns(task: str) -> list[str]:
    """Columns only used while training, to build look-up tables."""
    return ["city_risk_score"] if task == "approval" else []


def feature_names(kind: str, task: str) -> list[str]:
    """The numbers the model sees, in a fixed order."""
    return ["city_risk_score" if name == "city" else name for name in input_columns(kind, task)]


def categorical_positions(kind: str, task: str) -> list[int]:
    """Positions of the category columns among the model's numbers (gradient boosting treats them as categories)."""
    return [i for i, name in enumerate(feature_names(kind, task)) if name in CATEGORY_COLUMNS]


class ModelMissing(RuntimeError):
    """A claims model file is not there (or cannot be read); the message says how to create it."""


def _most_common(values: pd.Series):
    return values.mode().iat[0]


# --------------------------------------------------------------------------- 1. feature engineering
class ClaimFeatures(BaseEstimator, TransformerMixin):
    """Turns the details of a claim into the numbers the model learns from.

    fit() learns from the training rows: a code number for every category (damage type, time of day, bike use),
    the city -> city risk look-up, and typical answers for the optional questions.
    transform() needs only the claim's own details: category words become code numbers (an unknown word becomes
    'missing'), the city becomes its risk score, a skipped optional answer gets the typical value.
    """

    def __init__(self, kind: str = "car", task: str = "amount"):
        self.kind = kind
        self.task = task

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        names = feature_names(self.kind, self.task)
        self.categories_ = {c: sorted(X[c].dropna().unique().tolist()) for c in CATEGORY_COLUMNS if c in names}
        self.codes_ = {c: {value: number for number, value in enumerate(values)} for c, values in self.categories_.items()}
        self.defaults_ = {}
        for column in input_columns(self.kind, self.task):
            if column in ALWAYS_NEEDED or column == "city":
                continue
            common = column in CATEGORY_COLUMNS or column in YES_NO_COLUMNS
            self.defaults_[column] = _most_common(X[column]) if common else float(X[column].median())
        if self.task == "approval":
            self.city_risk_ = X.groupby("city")["city_risk_score"].median().to_dict()
            self.risk_default_ = float(X["city_risk_score"].median())
        return self

    def transform(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for column, value in self.defaults_.items():            # optional details the customer skipped
            X[column] = X[column].fillna(value) if column in X.columns else value
        out = pd.DataFrame(index=X.index)
        for name in feature_names(self.kind, self.task):
            if name == "city_risk_score":      # no city given (or an unknown one) -> the typical city risk
                known = X["city"].map(self.city_risk_) if "city" in X.columns else pd.Series(np.nan, index=X.index)
                out[name] = known.fillna(self.risk_default_).astype(float)
            elif name in self.codes_:
                out[name] = X[name].map(self.codes_[name]).astype(float)       # unknown word -> missing (NaN)
            else:
                out[name] = X[name].astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(feature_names(self.kind, self.task), dtype=object)

    def decode(self, column: str, code: float) -> str:
        values = self.categories_[column]
        return str(values[int(code)]) if code == code and 0 <= int(code) < len(values) else "unknown"


# --------------------------------------------------------------------------- 2. the models
class ClaimAmountModel(BaseEstimator, RegressorMixin):
    """Predicts the claim amount in rupees.

    A claim is a share of the insured value (a scratch costs about 3 % of the IDV, a total loss all of it), so the model
    learns log(amount / IDV) and predict() multiplies the answer by the IDV again. This works much better than learning
    the rupees directly, because the IDV of a car ranges from 90,000 to over 2 crore.
    """

    def __init__(self, kind: str = "car"):
        self.kind = kind

    def fit(self, X, y):
        X = pd.DataFrame(X)
        idv = X["idv"].to_numpy(dtype=float)
        self.features_ = ClaimFeatures(self.kind, "amount").fit(X)
        self.booster_ = HistGradientBoostingRegressor(
            max_iter=MAX_ROUNDS, learning_rate=0.1, early_stopping=True, random_state=0,
            categorical_features=categorical_positions(self.kind, "amount"))
        self.booster_.fit(self.features_.transform(X), np.log(np.asarray(y, dtype=float) / idv))
        self.n_rounds_ = int(self.booster_.n_iter_)
        return self

    def predict(self, X):
        X = pd.DataFrame(X)
        share = np.exp(self.booster_.predict(self.features_.transform(X)))
        return share * X["idv"].to_numpy(dtype=float)


class ClaimApprovalModel(BaseEstimator, ClassifierMixin):
    """Predicts the probability that a claim is approved (and approved / rejected at a chosen threshold)."""

    def __init__(self, kind: str = "car"):
        self.kind = kind

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.features_ = ClaimFeatures(self.kind, "approval").fit(X)
        self.booster_ = HistGradientBoostingClassifier(
            max_iter=MAX_ROUNDS, learning_rate=0.1, early_stopping=True, random_state=0,
            categorical_features=categorical_positions(self.kind, "approval"))
        self.booster_.fit(self.features_.transform(X), np.asarray(y, dtype=int))
        self.classes_ = self.booster_.classes_
        self.n_rounds_ = int(self.booster_.n_iter_)
        return self

    def predict_proba(self, X):
        return self.booster_.predict_proba(self.features_.transform(pd.DataFrame(X)))

    def approval_probability(self, X) -> np.ndarray:
        """Probability (0 to 1) that the claim is approved."""
        return self.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float | None = None):
        threshold = config.APPROVAL_THRESHOLD if threshold is None else threshold
        return (self.approval_probability(X) >= threshold).astype(int)


def build_model(kind: str, task: str):
    return ClaimAmountModel(kind) if task == "amount" else ClaimApprovalModel(kind)


class LinearAmountBaseline(BaseEstimator, RegressorMixin):
    """A fair 'straight-line' comparison for the amount: ridge regression on log(amount / IDV), same features."""

    def __init__(self, kind: str = "car"):
        self.kind = kind

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.features_ = ClaimFeatures(self.kind, "amount").fit(X)
        self.pipeline_ = _linear_pipeline(self.kind, "amount", Ridge(alpha=1.0))
        table = self.features_.transform(X)
        self.pipeline_.fit(table, np.log(np.asarray(y, dtype=float) / X["idv"].to_numpy(dtype=float)))
        return self

    def predict(self, X):
        X = pd.DataFrame(X)
        return np.exp(self.pipeline_.predict(self.features_.transform(X))) * X["idv"].to_numpy(dtype=float)


def _linear_pipeline(kind: str, task: str, estimator):
    names = feature_names(kind, task)
    categories = [name for name in names if name in CATEGORY_COLUMNS]
    numbers = [name for name in names if name not in CATEGORY_COLUMNS]
    prepare = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore"), categories),
                                 ("numbers", StandardScaler(), numbers)])
    return Pipeline([("prepare", prepare), ("model", estimator)])


class LogisticApprovalBaseline(BaseEstimator, ClassifierMixin):
    """A fair 'straight-line' comparison for the approval: logistic regression on the same features."""

    def __init__(self, kind: str = "car"):
        self.kind = kind

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.features_ = ClaimFeatures(self.kind, "approval").fit(X)
        self.pipeline_ = _linear_pipeline(self.kind, "approval", LogisticRegression(max_iter=2000))
        self.pipeline_.fit(self.features_.transform(X), np.asarray(y, dtype=int))
        self.classes_ = self.pipeline_.classes_
        return self

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(self.features_.transform(pd.DataFrame(X)))


# --------------------------------------------------------------------------- 3. data, split, evaluation
def raw_path(kind: str) -> Path:
    return config.RAW_DIR / config.RAW_FILES[f"{kind}_claims"]


def load_training_data(kind: str, path: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    """Read the claims CSV, keep the columns we need, drop repeated ids and incomplete rows."""
    path = Path(path) if path else raw_path(kind)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Finish step 3 first (python -m scripts.import_data).")
    wanted = (["record_id", "claim_amount", "claim_approved"] + input_columns(kind, "approval")
              + lookup_columns("approval"))
    df = pd.read_csv(path, usecols=lambda column: column in wanted)
    absent = [column for column in wanted if column not in df.columns]
    if absent:
        raise ValueError(f"{path.name} has no column named {', '.join(absent)}.")
    rows_read = len(df)
    df = df.drop_duplicates("record_id")
    repeated = rows_read - len(df)
    before = len(df)
    df = df.dropna(subset=wanted)
    df = df[(df["claim_amount"] > 0) & (df["idv"] > 0)]
    info = {"file": path.name, "rows_read": rows_read, "repeated_ids_dropped": repeated,
            "incomplete_rows_dropped": before - len(df), "rows_used": len(df)}
    return df.reset_index(drop=True), info


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80 percent to learn from, 20 percent held back. The held-back rows are only used to measure the result."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def xy(df: pd.DataFrame, kind: str, task: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df[input_columns(kind, task) + lookup_columns(task)]
    return X, (df["claim_amount"] if task == "amount" else df["claim_approved"].astype(int))


def score_amount(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    gap = np.abs(predicted / actual - 1)
    return {"n": int(len(actual)), "r2": float(r2_score(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted)), "mean_claim": float(actual.mean()),
            "median_pct_error": float(100 * np.median(gap)), "within_10": float(100 * np.mean(gap <= 0.10)),
            "within_25": float(100 * np.mean(gap <= 0.25))}


IDV_BANDS = {
    "car": [(0, 300_000, "up to 3 lakh"), (300_000, 600_000, "3 to 6 lakh"), (600_000, 1_000_000, "6 to 10 lakh"),
            (1_000_000, 2_500_000, "10 to 25 lakh"), (2_500_000, 10 ** 12, "above 25 lakh")],
    "bike": [(0, 50_000, "up to 50,000"), (50_000, 100_000, "50,000 to 1 lakh"), (100_000, 200_000, "1 to 2 lakh"),
             (200_000, 10 ** 12, "above 2 lakh")],
}


def evaluate_amount(model, kind: str, X_test: pd.DataFrame, y_test) -> dict:
    """R2, RMSE and error sizes on the held-out rows: all of them and per IDV band."""
    predicted = np.asarray(model.predict(X_test), dtype=float)
    actual = np.asarray(y_test, dtype=float)
    idv = X_test["idv"].to_numpy(dtype=float)
    result = {"all": score_amount(actual, predicted), "bands": []}
    for low, high, label in IDV_BANDS[kind]:
        rows = (idv > low) & (idv <= high)
        if rows.sum() >= 30:
            result["bands"].append({"label": label, **score_amount(actual[rows], predicted[rows])})
    return result


def score_approval(y_true, probability, threshold: float | None = None) -> dict:
    threshold = config.APPROVAL_THRESHOLD if threshold is None else threshold
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "threshold": float(threshold), "accuracy": float(np.mean(y == pred)),
            "precision_approved": float(precision_score(y, pred, zero_division=0)),
            "recall_approved": float(recall_score(y, pred, zero_division=0)),
            "f1_approved": float(f1_score(y, pred, zero_division=0)),
            "f1_rejected": float(f1_score(1 - y, 1 - pred, zero_division=0)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "auc": float(roc_auc_score(y, probability)), "brier": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6))),
            "approved_rate": float(y.mean()), "predicted_approved_rate": float(pred.mean()),
            "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]]}


THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


CALIBRATION_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def calibration_table(y_true, probability) -> list[dict]:
    """Claims grouped by the probability the model gave: how many of them were really approved? (Close = trustworthy.)"""
    y = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    group = np.digitize(probability, CALIBRATION_BINS[1:-1], right=True)
    rows = []
    for number in range(len(CALIBRATION_BINS) - 1):
        chosen = group == number
        if chosen.sum() >= 30:
            rows.append({"low": CALIBRATION_BINS[number], "high": CALIBRATION_BINS[number + 1], "n": int(chosen.sum()),
                         "predicted": float(probability[chosen].mean()), "actual": float(y[chosen].mean())})
    return rows


def evaluate_approval(model, X_test: pd.DataFrame, y_test) -> dict:
    """Accuracy, F1 and friends at the project's threshold, the same numbers at other thresholds, and a calibration table."""
    probability = model.approval_probability(X_test)
    result = {"at_threshold": score_approval(y_test, probability), "thresholds": [],
              "calibration": calibration_table(y_test, probability)}
    for threshold in THRESHOLDS:
        result["thresholds"].append(score_approval(y_test, probability, threshold))
    return result


def amount_baselines(kind: str, X_train, y_train, X_test, y_test, model) -> list[dict]:
    """Same test rows, three ways of guessing the amount: a plain guess, straight-line rules, and ours."""
    actual = np.asarray(y_test, dtype=float)
    share = float(np.median(np.asarray(y_train, dtype=float) / X_train["idv"].to_numpy(dtype=float)))
    guess = share * X_test["idv"].to_numpy(dtype=float)
    linear = LinearAmountBaseline(kind).fit(X_train, y_train)
    return [{"name": "Typical share of the IDV for everyone", **score_amount(actual, guess)},
            {"name": "Linear regression (straight-line rules)", **score_amount(actual, linear.predict(X_test))},
            {"name": "Gradient boosting (this project's model)", **score_amount(actual, model.predict(X_test))}]


def approval_baselines(kind: str, X_train, y_train, X_test, y_test, model) -> list[dict]:
    """Same test rows: approve everything, straight-line rules (logistic regression), and ours."""
    y = np.asarray(y_test, dtype=int)
    everything = np.full(len(y), float(np.mean(np.asarray(y_train, dtype=int))) >= 0.5)
    logistic = LogisticApprovalBaseline(kind).fit(X_train, y_train)
    rows = []
    for name, probability in (("Approve every claim", np.where(everything, 1.0, 0.0)),
                              ("Logistic regression (straight-line rules)", logistic.predict_proba(X_test)[:, 1]),
                              ("Gradient boosting (this project's model)", model.approval_probability(X_test))):
        rows.append({"name": name, **score_approval(y, probability)})
    return rows


# --------------------------------------------------------------------------- 4. what the claim page offers
def build_options(df: pd.DataFrame, kind: str, parts: list[str]) -> dict:
    """Everything the claim page needs to know about the data: choices, ranges and typical answers."""
    risk = df.groupby("city")["city_risk_score"].median()
    options = {
        "damage_types": DAMAGE_TYPES, "times_of_day": TIMES_OF_DAY, "parts": parts, "max_parts": MAX_PARTS[kind],
        "cities": sorted(risk.index.tolist()), "city_risk": {city: float(value) for city, value in risk.items()},
        "ncb_steps": sorted(int(v) for v in df["ncb_at_claim_percent"].unique()),
        "ranges": {"previous_claims_count": [int(df["previous_claims_count"].min()), int(df["previous_claims_count"].max())],
                   "policy_age_months": [int(df["policy_age_months"].min()), int(df["policy_age_months"].max())],
                   "damage_severity_score": [1, 10]},
        "idv_seen": {"min": int(df["idv"].min()), "max": int(df["idv"].max())},
        "defaults": {"zero_dep_addon": int(_most_common(df["zero_dep_addon"])),
                     "previous_claims_count": int(_most_common(df["previous_claims_count"])),
                     "policy_age_months": int(round(df["policy_age_months"].median())),
                     "incident_time_of_day": _most_common(df["incident_time_of_day"]),
                     "ncb_at_claim_percent": int(round(df["ncb_at_claim_percent"].median()))},
    }
    if kind == "bike":
        options["usage_types"] = sorted(df["usage_type"].unique().tolist())
        options["ranges"]["rider_experience_years"] = [int(df["rider_experience_years"].min()),
                                                       int(df["rider_experience_years"].max())]
        options["defaults"].update({"usage_type": _most_common(df["usage_type"]), "helmet_worn": int(_most_common(df["helmet_worn"])),
                                    "rider_experience_years": int(round(df["rider_experience_years"].median()))})
    return options


def parts_in_data(kind: str, path: Path | str | None = None) -> list[str]:
    """Every damaged part named in the claims file (about 25 for cars, 20 for bikes)."""
    path = Path(path) if path else raw_path(kind)
    column = pd.read_csv(path, usecols=["affected_parts"])["affected_parts"]
    return sorted({part for row in column.dropna().str.split(", ") for part in row})


def approval_rule_check(kind: str, path: Path | str | None = None) -> dict:
    """Two facts about the data that the report states: the file's own 'approval_probability' column decides the
    approved / rejected column exactly (so it can never be an input), and a few rows with an amount above the IDV."""
    path = Path(path) if path else raw_path(kind)
    table = pd.read_csv(path, usecols=["approval_probability", "claim_approved", "claim_amount", "idv"])
    rule = (table["approval_probability"] >= 0.5).astype(int)
    return {"rows": int(len(table)), "label_equals_rule": float(100 * np.mean(rule == table["claim_approved"])),
            "amount_above_idv": float(100 * np.mean(table["claim_amount"] > table["idv"]))}


def approval_ceiling(kind: str, train: pd.DataFrame, test: pd.DataFrame, path: Path | str | None = None) -> dict:
    """How predictable is the approval at all? Train a model that predicts the file's own hidden 'approval_probability'
    score (never used as an input anywhere else) and see how much of its ups and downs the claim details explain.
    If it explains only part, the rest is chance, and no classifier can beat the accuracy this check reaches."""
    path = Path(path) if path else raw_path(kind)
    scores = (pd.read_csv(path, usecols=["record_id", "approval_probability"]).drop_duplicates("record_id")
              .set_index("record_id")["approval_probability"])
    X_train, _ = xy(train, kind, "approval")
    X_test, y_test = xy(test, kind, "approval")
    features = ClaimFeatures(kind, "approval").fit(X_train)
    booster = HistGradientBoostingRegressor(max_iter=MAX_ROUNDS, learning_rate=0.1, early_stopping=True, random_state=0,
                                            categorical_features=categorical_positions(kind, "approval"))
    booster.fit(features.transform(X_train), train["record_id"].map(scores).to_numpy(dtype=float))
    predicted = booster.predict(features.transform(X_test))
    actual = test["record_id"].map(scores).to_numpy(dtype=float)
    return {"r2": float(r2_score(actual, predicted)), "accuracy": float(np.mean((predicted >= 0.5) == np.asarray(y_test, dtype=bool))),
            "rows": int(len(actual))}


# --------------------------------------------------------------------------- 5. the saved files and the engine
def save_bundle(bundle: dict, kind: str, task: str) -> Path:
    path = MODEL_FILES[(kind, task)]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(kind: str, task: str) -> dict:
    path = MODEL_FILES[(kind, task)]
    if not path.exists():
        raise ModelMissing(f"The claims model file {path.name} was not found. Create it with:  python -m scripts.train_claims")
    try:
        bundle = joblib.load(path)
    except Exception as exc:          # a file from other library versions, or a broken download
        raise ModelMissing(f"The claims model file {path.name} could not be read ({type(exc).__name__}). "
                           "Create it again with:  python -m scripts.train_claims") from exc
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise ModelMissing(f"{path.name} is from an older version. Create it again with:  python -m scripts.train_claims")
    return bundle


def status_for(probability: float, threshold: float | None = None) -> str:
    """'approved' when the approval probability reaches the threshold, otherwise the claim goes to a human: 'review'."""
    threshold = config.APPROVAL_THRESHOLD if threshold is None else threshold
    return "approved" if probability >= threshold else "review"


@dataclass
class ClaimEstimate:
    kind: str
    amount: float                  # the claim amount in rupees (never more than the IDV)
    probability: float             # 0 to 1
    status: str                    # approved / review
    threshold: float
    severity: str                  # minor / moderate / major
    raw_amount: float = 0.0        # what the model said before the amount was limited to the IDV
    notes: list[str] = field(default_factory=list)
    answers: dict = field(default_factory=dict)


class ClaimEngine:
    """The four trained models (car amount, car approval, bike amount, bike approval). Used by the claim page."""

    def __init__(self, bundles: dict):
        self.bundles = bundles          # {(kind, task): bundle}

    @classmethod
    def load(cls) -> "ClaimEngine":
        return cls({(kind, task): load_bundle(kind, task) for kind in KINDS for task in TASKS})

    def options(self, kind: str) -> dict:
        return self.bundles[(kind, "approval")]["options"]

    def metrics(self, kind: str, task: str) -> dict:
        return self.bundles[(kind, task)]["metrics"]

    def estimate(self, kind: str, damage_type: str, severity_score: int, num_parts: int, idv: float, *,
                 zero_dep: int | None = None, previous_claims: int | None = None, policy_age_months: int | None = None,
                 time_of_day: str | None = None, ncb: int | None = None, city: str | None = None,
                 usage_type: str | None = None, helmet_worn: int | None = None, rider_experience: int | None = None,
                 threshold: float | None = None) -> ClaimEstimate:
        if kind not in KINDS:
            raise ValueError("kind must be 'car' or 'bike'")
        if damage_type not in DAMAGE_TYPES:
            raise ValueError(f"damage_type must be one of: {', '.join(DAMAGE_TYPES)}")
        if not 1 <= int(severity_score) <= 10:
            raise ValueError("severity_score must be between 1 and 10")
        if not 1 <= int(num_parts) <= MAX_PARTS[kind]:
            raise ValueError(f"num_parts must be between 1 and {MAX_PARTS[kind]} for a {kind}")
        if idv <= 0:
            raise ValueError("IDV must be more than zero")
        threshold = config.APPROVAL_THRESHOLD if threshold is None else threshold
        answers = {"damage_type": damage_type, "damage_severity_score": int(severity_score), "num_parts_affected": int(num_parts),
                   "idv": float(idv), "zero_dep_addon": zero_dep, "previous_claims_count": previous_claims,
                   "policy_age_months": policy_age_months, "incident_time_of_day": time_of_day, "ncb_at_claim_percent": ncb,
                   "city": city, "usage_type": usage_type if kind == "bike" else None,
                   "helmet_worn": helmet_worn if kind == "bike" else None,
                   "rider_experience_years": rider_experience if kind == "bike" else None}
        frame = pd.DataFrame([{k: v for k, v in answers.items() if v is not None}])     # skipped answers get typical values
        raw = float(self.bundles[(kind, "amount")]["model"].predict(frame)[0])
        probability = float(self.bundles[(kind, "approval")]["model"].approval_probability(frame)[0])
        notes = []
        amount = min(raw, float(idv))
        if raw > idv:
            notes.append(f"The model's estimate was {inr(raw)}; a claim is never paid above the insured value, so it is limited to {inr(idv)}.")
        seen = self.options(kind)["idv_seen"]
        if idv < seen["min"] or idv > seen["max"]:
            notes.append(f"This IDV is outside the {kind} claims the model has seen ({inr(seen['min'])} to {inr(seen['max'])}), "
                         "so the estimate is less reliable.")
        low, high = self.options(kind)["ranges"]["policy_age_months"]
        if policy_age_months is not None and not low <= policy_age_months <= high:
            notes.append(f"The model has seen policies that are {low} to {high} months old; this one is outside that range.")
        return ClaimEstimate(kind=kind, amount=amount, probability=probability, status=status_for(probability, threshold),
                             threshold=float(threshold), severity=severity_label(severity_score), raw_amount=raw,
                             notes=notes, answers=answers)
