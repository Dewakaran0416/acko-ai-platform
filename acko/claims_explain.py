"""Module 3: explaining the four claims models with SHAP.

The four claims notebooks (notebooks/03 to 06) use this file. The claim page does not need it.

Why SHAP's Permutation explainer and not shap.TreeExplainer: the claims models are gradient boosting trees that treat damage
type, time of day and bike use as real categories. For trees like that, TreeExplainer's pieces do not add up to what the model
says (every notebook shows the gap). The Permutation explainer only asks the model questions - "what would you say if this
detail were a typical one?" - so it works for any model, and its pieces add up exactly:
    typical answer + the SHAP values of one claim = the model's answer for that claim.
It is slower, which is why the notebooks explain 300 test claims and not thousands.

What is explained:
    amount    the claim as a SHARE of the insured value (a scratch is about 3 % of the IDV, a total loss all of it).
              Multiplied by the claim's own IDV, the pieces are rupees that add up to the estimate.
    approval  the PROBABILITY (0 to 1) that the claim is approved. The pieces are percentage points of that probability.

Pictures are drawn on a matplotlib axis that the notebook supplies, so what the notebook shows is exactly what is saved
(save_figure writes it to reports/shap/).
"""
from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from acko import claims, config
from acko.fmt import group_indian, inr

SHAP_DIR = config.ROOT / "reports" / "shap"
N_BACKGROUND = 100        # training claims that stand for "a typical claim"
N_EXPLAINED = 300         # test claims whose answers are explained
N_PERMUTATIONS = 15       # random orders of the columns tried for every claim (more = steadier numbers, slower)
TOP_STEPS = 7             # a worked example names this many columns; the rest are added up as "other"

# The quiet chart style of the notebooks: one blue for one series, grey for context, thin solid grid lines.
# The three category colours (blue, orange, aqua) were checked for colour-blind readers.
SURFACE, INK, INK2, MUTED, GRID, CONTEXT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RUPEES = FuncFormatter(lambda value, _position: group_indian(value))      # axis labels such as 10,00,000
PERCENT = FuncFormatter(lambda value, _position: f"{value:.0f} %")


def apply_style() -> None:
    """The look of every chart in the claims notebooks."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "axes.edgecolor": CONTEXT, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "legend.fontsize": 8})
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)


def tidy(axis, grid: str = "y"):
    """No top/right frame; thin solid grid lines on the chosen axis ('x', 'y' or 'both') behind the marks."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    return axis


# --------------------------------------------------------------------------- 1. the SHAP values
@dataclass
class Explained:
    kind: str
    task: str
    names: list              # the model's columns, in order
    table: pd.DataFrame      # the numbers the model sees for the explained claims (category columns hold code numbers)
    values: np.ndarray       # SHAP values: one row per claim, one column per model column
    bases: np.ndarray        # the typical answer (the model's average answer over the background claims), one per claim
    output: np.ndarray       # the model's own answer for each claim: the share of the IDV, or the probability
    seconds: float

    @property
    def labels(self) -> list:
        return [claims.COLUMN_LABELS[name] for name in self.names]

    @property
    def base(self) -> float:
        return float(np.mean(self.bases))


def output_function(model, task: str, names: list):
    """The question SHAP asks the model: numbers in, one answer per row out (the share of the IDV, or the probability)."""
    booster = model.booster_
    if task == "amount":
        def answer(numbers):
            return np.exp(booster.predict(pd.DataFrame(numbers, columns=names)))
    else:
        def answer(numbers):
            return booster.predict_proba(pd.DataFrame(numbers, columns=names))[:, 1]
    return answer


def background_claims(model, task: str, X_train: pd.DataFrame, size: int = N_BACKGROUND) -> pd.DataFrame:
    """Training claims that stand for "a typical claim": picked evenly along the model's own answers, from the lowest to
    the highest. Their average answer then matches the whole training data's (a random handful can miss it by a lot)."""
    names = list(model.features_.get_feature_names_out())
    answers = output_function(model, task, names)(model.features_.transform(X_train).to_numpy(dtype=float))
    order = np.argsort(answers, kind="stable")
    return X_train.iloc[order[np.linspace(0, len(order) - 1, size).round().astype(int)]]


def explain(model, task: str, background: pd.DataFrame, sample: pd.DataFrame, seed: int = 0) -> Explained:
    """SHAP values of the claims in `sample` (claim details, as claims.xy gives them); `background` stands for a typical claim."""
    import shap          # imported here because it takes a few seconds to load
    started = time.time()
    names = list(model.features_.get_feature_names_out())
    typical = model.features_.transform(background)
    table = model.features_.transform(sample)
    answer = output_function(model, task, names)
    masker = shap.maskers.Independent(typical.to_numpy(dtype=float), max_samples=len(typical))
    explainer = shap.explainers.Permutation(answer, masker, feature_names=names, seed=seed)
    result = explainer(table.to_numpy(dtype=float), max_evals=2 * len(names) * N_PERMUTATIONS + 1, silent=True)
    return Explained(model.kind, task, names, table, np.asarray(result.values, dtype=float),
                     np.asarray(result.base_values, dtype=float).ravel(), np.asarray(answer(table.to_numpy(dtype=float)), dtype=float),
                     time.time() - started)


def additivity_gap(explained: Explained) -> float:
    """How far 'typical answer + SHAP values' is from the model's own answer (it should be about 1e-15: rounding only)."""
    return float(np.max(np.abs(explained.bases + explained.values.sum(axis=1) - explained.output)))


def importance(explained: Explained) -> pd.Series:
    """Average size of the push of each column, in points (amount: points of the IDV; approval: points of probability)."""
    push = 100 * np.abs(explained.values).mean(axis=0)
    return pd.Series(push, index=explained.labels).sort_values(ascending=False)


def colour_codes(model, table: pd.DataFrame, damage_order: list) -> pd.DataFrame:
    """The same table with the category columns renumbered in a meaningful order - only for the colours of the dot plot.

    The model's own code numbers follow the alphabet (bumper damage 0, collision damage 1, ...), so a colour scale on them would
    mean nothing. Damage types are numbered from the cheapest to the dearest, times of day in clock order, bike use from
    personal to delivery."""
    orders = {"damage_type": list(damage_order), "incident_time_of_day": claims.TIMES_OF_DAY, "usage_type": claims.USAGE_TYPES}
    out = table.copy()
    for column, order in orders.items():
        if column in out.columns:
            place = {word: number for number, word in enumerate(order)}
            out[column] = [place.get(model.features_.decode(column, code), np.nan) for code in table[column]]
    return out


# --------------------------------------------------------------------------- 2. pictures
def draw_importance(axis, explained: Explained, unit: str) -> pd.Series:
    """Horizontal bars: how far each column moves the answer on average (biggest on top)."""
    push = importance(explained)[::-1]
    axis.barh(push.index, push.values, color=BLUE, height=0.6)
    for position, value in enumerate(push.values):
        axis.text(value, position, f" {value:.1f}", va="center", fontsize=8, color=INK2)
    axis.set_xlim(0, float(push.max()) * 1.15)
    axis.set_xlabel(unit)
    tidy(axis, "x")
    return push[::-1]


def detail_text(column: str, row: pd.Series) -> str:
    """One claim detail in words, for the labels under a worked example."""
    value = row[column]
    if column == "idv":
        return inr(value)
    if column in ("zero_dep_addon", "helmet_worn"):
        return "yes" if int(value) else "no"
    if column == "damage_severity_score":
        return f"{int(value)} of 10"
    if column == "policy_age_months":
        return f"{int(value)} months"
    if column == "ncb_at_claim_percent":
        return f"{int(value)} %"
    if column == "rider_experience_years":
        return f"{int(value)} years"
    if column == "city_risk_score":
        return f"{row['city']} ({value:.2f})" if "city" in row.index else f"{value:.2f}"
    if column in ("damage_type", "incident_time_of_day", "usage_type"):
        return str(value)
    return str(int(value))


def example_steps(explained: Explained, position: int, row: pd.Series):
    """What every column did to one explained claim: (steps, start, end).

    A step is (column, the claim's value in words, push). amount: pushes in rupees (the pieces times the claim's IDV);
    approval: percentage points. The steps run from the biggest push to the smallest; everything beyond TOP_STEPS columns is
    added up as one "other" step. start + all the pushes = end, exactly."""
    scale = float(row["idv"]) if explained.task == "amount" else 100.0
    pieces = [(label, detail_text(name, row), float(value * scale))
              for name, label, value in zip(explained.names, explained.labels, explained.values[position])]
    pieces.sort(key=lambda piece: abs(piece[2]), reverse=True)
    steps, rest = pieces[:TOP_STEPS], pieces[TOP_STEPS:]
    if rest:
        steps.append(("Other columns", f"{len(rest)} columns", float(sum(piece[2] for piece in rest))))
    return steps, float(explained.bases[position] * scale), float(explained.output[position] * scale)


def draw_waterfall(axis, steps: list, start: float, end: float, start_label: str, end_label: str, money: bool) -> None:
    """Start at the typical answer, add or take off the push of each column, arrive at the model's answer."""
    def step_text(value):
        return ("+" if value >= 0 else "-") + (inr(abs(value)) if money else f"{abs(value):.0f}")

    def total_text(value):
        return inr(value) if money else f"{value:.0f} %"

    def wrap(text):
        return textwrap.fill(text, 11, break_long_words=False)

    labels = [start_label] + [f"{wrap(label)}\n{wrap(detail)}" for label, detail, _ in steps] + [end_label]
    running, highest, lowest = start, max(start, end), 0.0
    axis.bar(0, running, color=CONTEXT, width=0.7)
    axis.text(0, running, total_text(running), ha="center", va="bottom", fontsize=8, color=INK2)
    for position, (_, _, value) in enumerate(steps, start=1):
        bottom = running if value >= 0 else running + value
        axis.bar(position, abs(value), bottom=bottom, color=BLUE if value >= 0 else ORANGE, width=0.7)
        axis.text(position, max(running, running + value), step_text(value), ha="center", va="bottom", fontsize=8, color=INK2)
        running += value
        highest, lowest = max(highest, running), min(lowest, running)
    axis.bar(len(labels) - 1, end, color=CONTEXT, width=0.7)
    axis.text(len(labels) - 1, end, total_text(end), ha="center", va="bottom", fontsize=8, color=INK2)
    axis.set_xticks(range(len(labels)), labels, fontsize=7.5)
    axis.yaxis.set_major_formatter(RUPEES if money else PERCENT)
    axis.set_ylim(lowest * 1.12, highest * 1.12)
    tidy(axis, "y")


def save_figure(figure, name: str) -> Path:
    """Save a notebook picture in reports/shap/ (the folder for the report's SHAP images)."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    path = SHAP_DIR / name
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path
