# Source of notebooks/03_claims_car_amount.ipynb. Build and run it with:  python -m scripts.make_notebooks car-amount
# Cells start with a line "# %%"; text cells start with "# %% [markdown]" and every line of text begins with "# ".

# %% [markdown]
# # Module 3 - Claim amount model for cars
#
# **Goal (from the brief):** estimate what a motor claim will cost, in rupees, from what is known when it is filed: the kind of
# damage, how bad it is, how many parts are damaged, the insured value of the car (IDV) and whether the policy has the
# zero-depreciation add-on. The brief asks for "an acceptable RMSE" but gives no number, so this project sets its own bar on
# held-out claims: **R2 above 0.85** and a **typical (median) error below 15 %** of the claim.
#
# This notebook follows the same steps as `python -m scripts.train_claims`, the script that writes `models/claim_car_amount.pkl`
# (the file the claim page loads). Here every step is shown together with its output:
#
# 1. the data
# 2. what the data looks like
# 3. train / test split
# 4. feature engineering
# 5. training
# 6. results against the targets
# 7. why gradient boosting, and why it learns a share of the IDV
# 8. would more columns help?
# 9. SHAP: why does the model give this amount?
# 10. checking the saved `.pkl` file
#
# Use **Run All**; it takes a few minutes. The notebook never changes anything in `models/`. Besides this page, its only output
# is three SHAP pictures that it saves in `reports/shap/` (section 9).

# %%
import inspect
import os
import platform
import sys
import textwrap
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="IProgress not found")      # a harmless notice from the progress bar inside shap
warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded")      # a harmless notice from shap's dot plot

# Find the project folder (the one that holds the acko package), so the notebook also works when it is opened from notebooks/.
ROOT = Path.cwd().resolve()
while not (ROOT / "acko" / "claims.py").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Open this notebook inside the AckoAI project folder (the folder that contains acko/).")
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from IPython.display import Markdown, display
from sklearn.ensemble import HistGradientBoostingRegressor

from acko import claims, claims_explain as explain
from acko.claims_explain import AQUA, BLUE, INK2, ORANGE, PERCENT, RUPEES, SURFACE, tidy
from acko.fmt import inr

KIND, TASK = "car", "amount"          # this notebook: the car claim-amount model
STARTED = time.time()
explain.apply_style()
print(f"project folder: {ROOT.name}   |   Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
      f"pandas {pd.__version__}, numpy {np.__version__}, shap {shap.__version__}")

# %% [markdown]
# ## 1. The data
# The course supplies the data: one claims file for cars and one for bikes. It is **synthetic** (made by the course authors, not real
# Acko claims), so the model learns the rules of this dataset. This notebook does not generate data; it starts from the file.
# `claims.load_training_data` keeps the columns the models need, removes repeated record ids and drops rows with missing values.

# %%
data, info = claims.load_training_data(KIND)
print(f"{info['file']}: {info['rows_read']:,} rows read, {info['repeated_ids_dropped']} repeated record ids and "
      f"{info['incomplete_rows_dropped']} incomplete rows removed -> {info['rows_used']:,} car claims")
display(data[claims.input_columns(KIND, TASK) + ["claim_amount"]].head())

# %%
data["share"] = data["claim_amount"] / data["idv"]          # the claim as a share of the insured value
overview = pd.DataFrame(
    {"lowest": [inr(data["claim_amount"].min()), inr(data["idv"].min()), f"{100 * data['share'].min():.1f} %"],
     "middle (median)": [inr(data["claim_amount"].median()), inr(data["idv"].median()), f"{100 * data['share'].median():.1f} %"],
     "highest": [inr(data["claim_amount"].max()), inr(data["idv"].max()), f"{100 * data['share'].max():.1f} %"]},
    index=["claim amount", "insured value (IDV)", "claim as a share of the IDV"])
display(overview)

# %% [markdown]
# ## 2. What the data looks like
# A claim is easiest to understand as a **share of the insured value (IDV)**: a scratch costs a few percent of what the car is
# worth, a total loss costs all of it. The first picture shows what sets that share.

# %%
figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1.5, 1, 0.9]})
by_type = (100 * data.groupby("damage_type")["share"].median()).sort_values()
medians = {}
axes[0].barh(by_type.index, by_type.values, color=BLUE, height=0.6)
for position, value in enumerate(by_type.values):
    axes[0].text(value, position, f" {value:.0f} %", va="center", fontsize=8, color=INK2)
axes[0].set_xlim(0, 120)
axes[0].xaxis.set_major_formatter(PERCENT)
axes[0].set_title("by damage type")
tidy(axes[0], "x")
for axis, column, title in ((axes[1], "damage_severity_score", "by severity score (1 = light, 10 = worst)"),
                            (axes[2], "num_parts_affected", "by number of damaged parts")):
    by_value = 100 * data.groupby(column)["share"].median()
    medians[column] = by_value
    axis.bar(by_value.index.astype(str), by_value.values, color=BLUE, width=0.7)
    for position, value in enumerate(by_value.values):
        axis.text(position, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    axis.set_ylim(0, 120)
    axis.yaxis.set_major_formatter(PERCENT)
    axis.set_title(title)
    tidy(axis, "y")
figure.suptitle(f"Middle value of the claim as a share of the IDV ({len(data):,} car claims)", fontsize=10)
figure.tight_layout()
plt.show()
by_severity, by_parts = medians["damage_severity_score"], medians["num_parts_affected"]
all_of_it = [name for name in by_type.index[::-1] if by_type[name] >= 95]
display(Markdown(f"The damage type sets the level: {', '.join(all_of_it)} cost about all of the IDV (the middle claim is {by_type[all_of_it].min():.0f} % or more of it), "
                 f"while {by_type.index[0]} costs {by_type.iloc[0]:.0f} % and {by_type.index[1]} {by_type.iloc[1]:.0f} %. A higher severity score and more damaged parts push the share up "
                 f"further: from {by_severity.iloc[0]:.0f} % at severity {by_severity.index[0]} to {by_severity.iloc[-1]:.0f} % at severity {by_severity.index[-1]}, and from "
                 f"{by_parts.iloc[0]:.0f} % with {by_parts.index[0]} damaged part to {by_parts.iloc[-1]:.0f} % with {by_parts.index[-1]}."))

# %% [markdown]
# The second picture shows the claims themselves, coloured with the brief's three severity words (minor, moderate, major).

# %%
figure, axis = plt.subplots(figsize=(8.5, 4.9))
picked = data.sample(6000, random_state=0)
for (low, high, band), colour in zip(claims.SEVERITY_BANDS, (BLUE, ORANGE, AQUA)):
    part = picked[picked["damage_severity_score"].map(claims.severity_label) == band]
    axis.scatter(part["idv"], part["claim_amount"], s=7, alpha=0.35, color=colour, linewidths=0,
                 label=f"{band} damage (severity {low} to {high})")
limits = [picked["idv"].min(), picked["idv"].max()]
axis.plot(limits, limits, color=INK2, linewidth=1, label="claim = IDV")
axis.set(xscale="log", yscale="log", xlabel="insured value, IDV (Rs)", ylabel="claim amount (Rs)",
         title="6,000 car claims (both axes logarithmic)")
axis.xaxis.set_major_formatter(RUPEES)
axis.yaxis.set_major_formatter(RUPEES)
axis.legend(loc="upper left", markerscale=2)
tidy(axis, "both")
plt.show()

# %% [markdown]
# Claims of the same kind of damage lie along lines that run parallel to the line "claim = IDV": the more the car is worth, the
# more the claim costs, in about the same proportion. That is why the model below learns the **share** of the IDV and not the
# rupees (section 7 compares the two).
#
# One more detail, the zero-depreciation add-on:

# %%
by_addon = 100 * data.groupby("zero_dep_addon")["share"].median()
display(by_addon.round(1).rename(index={0: "no zero-depreciation add-on", 1: "zero-depreciation add-on"}).to_frame("middle share of the IDV (%)"))
display(Markdown(f"With the add-on the middle claim is {by_addon[1]:.1f} % of the IDV, without it {by_addon[0]:.1f} %: a difference of "
                 f"{abs(by_addon[1] - by_addon[0]):.1f} points, {'higher' if by_addon[1] > by_addon[0] else 'lower'} with the add-on."))

# %% [markdown]
# ## 3. Train / test split
# 80 % of the claims are used to learn, 20 % are held back and only used to measure the result (`random_state` 42, so everyone gets
# the same split).

# %%
train, test = claims.split_data(data)
X_train, y_train = claims.xy(train, KIND, TASK)
X_test, y_test = claims.xy(test, KIND, TASK)
print(f"{len(train):,} claims to learn from, {len(test):,} claims held back for testing")
print("details the model gets:", ", ".join(X_train.columns))

# %% [markdown]
# ## 4. Feature engineering
# The model cannot read words, so `ClaimFeatures` (in `acko/claims.py`) turns the details of a claim into numbers:
#
# | From the claim | Becomes |
# |---|---|
# | damage type (a word) | a code number for each damage type, learned from the training claims |
# | severity (1 to 10), damaged parts, IDV | passed on unchanged |
# | zero-depreciation add-on | 0 or 1 (a skipped answer gets the more common one) |
#
# `fit()` learns the code numbers and the typical answers from the **training claims only**. `transform()` then needs nothing but
# the details of the claim itself. This is its code (the branch for `city_risk_score` is only used by the approval models):

# %%
print(textwrap.dedent(inspect.getsource(claims.ClaimFeatures.transform)))

# %%
features = claims.ClaimFeatures(KIND, TASK).fit(X_train)
shown = X_test.head(5)
print("Five test claims, as the customer gives them:")
display(shown)
print("The same five, as the model sees them:")
display(features.transform(shown).round(2))
print("Code numbers learned for the damage types:")
print({word: int(number) for word, number in features.codes_["damage_type"].items()})
skipped = features.transform(shown.drop(columns=["zero_dep_addon"]))["zero_dep_addon"]
print("If the zero-depreciation answer is skipped it becomes", skipped.unique().tolist(), "(the more common answer in the training claims)")
assert skipped.nunique() == 1

# %% [markdown]
# ## 5. Training
# The model is a **gradient boosting** regressor (`HistGradientBoostingRegressor`): many small decision trees, each one correcting the
# mistakes of the trees before it. Two choices matter:
#
# - **It learns the share, not the rupees.** `ClaimAmountModel` learns log(amount / IDV) and multiplies its answer by the IDV
#   again. One pattern ("a dent costs about a tenth of what the car is worth") then applies to a car of any IDV.
# - **Damage type is a real category.** The trees may put "theft" and "total loss" on one side of a split without pretending that one
#   damage type is "bigger" than another, which a plain code number would suggest.
#
# Training stops by itself when 10 % of the training claims, kept aside inside the training data, show no more gain for 10 rounds in
# a row (at most 1,000 rounds). This is the code of `fit()`:

# %%
print(textwrap.dedent(inspect.getsource(claims.ClaimAmountModel.fit)))

# %%
started = time.time()
model = claims.build_model(KIND, TASK).fit(X_train, y_train)
print(f"trained in {time.time() - started:.1f} s, {model.n_rounds_} boosting rounds")

# %% [markdown]
# ## 6. Results against the targets
# All numbers below are measured on the held-back test claims, which the model never saw while learning.

# %%
metrics = claims.evaluate_amount(model, KIND, X_test, y_test)
score = metrics["all"]
display(pd.DataFrame(
    [["R2", f"{score['r2']:.4f}", "share of the ups and downs of the claim that the model explains (1 = all of it)"],
     ["RMSE", inr(score["rmse"]), "typical size of a mistake in rupees; the biggest claims weigh most"],
     ["MAE", inr(score["mae"]), "average mistake in rupees"],
     ["average claim", inr(score["mean_claim"]), "to compare the mistakes with"],
     ["typical error", f"{score['median_pct_error']:.1f} %", "half of the estimates are closer than this to the real claim"],
     ["within 10 %", f"{score['within_10']:.0f} %", "share of the estimates less than 10 % away from the real claim"],
     ["within 25 %", f"{score['within_25']:.0f} %", "share of the estimates less than 25 % away from the real claim"]],
    columns=["measure", f"{score['n']:,} test claims", "what it says"]).set_index("measure"))
met = score["r2"] > claims.TARGET_R2 and score["median_pct_error"] < claims.TARGET_MEDIAN_ERROR
print(f"targets: R2 above {claims.TARGET_R2} and typical error below {claims.TARGET_MEDIAN_ERROR:g} %   ->   "
      f"R2 {score['r2']:.4f}, typical error {score['median_pct_error']:.1f} %   ->   {'MET' if met else 'NOT MET'}")
assert met, "The model misses the targets"

# %% [markdown]
# The same numbers for claims of different insured value (IDV), and for each damage type:

# %%
display(pd.DataFrame([{"IDV": b["label"], "test claims": f"{b['n']:,}", "R2": round(b["r2"], 4), "RMSE": inr(b["rmse"]),
                       "typical error": f"{b['median_pct_error']:.1f} %"} for b in metrics["bands"]]).set_index("IDV"))
predicted = np.asarray(model.predict(X_test), dtype=float)
actual = y_test.to_numpy(dtype=float)
errors = pd.DataFrame({"damage type": X_test["damage_type"].to_numpy(), "gap": np.abs(predicted / actual - 1)})
by_type = errors.groupby("damage type")["gap"].agg(["size", "median"])
by_type["median"] = (100 * by_type["median"]).round(1)
display(by_type.rename(columns={"size": "test claims", "median": "typical error (%)"}).sort_values("typical error (%)"))
band_errors = [b["median_pct_error"] for b in metrics["bands"]]
worst = by_type["median"].idxmax()
display(Markdown(f"The typical error is **{band_errors[0]:.1f} %** in the cheapest IDV band and **{band_errors[-1]:.1f} %** in the dearest one, and the largest "
                 f"of the damage types is **{by_type.loc[worst, 'median']:.1f} %** for {worst}. Small claims have the larger errors in percent: the same rupee "
                 "mistake weighs more on a small claim."))
weak = [f"{b['label']} ({b['median_pct_error']:.1f} %)" for b in metrics["bands"] if b["median_pct_error"] >= claims.TARGET_MEDIAN_ERROR]
weak += [f"{name} ({by_type.loc[name, 'median']:.1f} %)" for name in by_type.index if by_type.loc[name, "median"] >= claims.TARGET_MEDIAN_ERROR]
if weak:
    display(Markdown(f"**Read this honestly:** the target (a typical error below {claims.TARGET_MEDIAN_ERROR:g} %) is met for all claims together, but not inside every group. "
                     f"The typical error is {claims.TARGET_MEDIAN_ERROR:g} % or more for: {', '.join(weak)}."))

# %%
figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
picked = np.random.default_rng(0).choice(len(actual), size=4000, replace=False)
axes[0].scatter(actual[picked], predicted[picked], s=6, alpha=0.3, color=BLUE, linewidths=0)
both = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
axes[0].plot(both, both, color=INK2, linewidth=1)
axes[0].set(xscale="log", yscale="log", xlabel="real claim (Rs)", ylabel="estimate (Rs)", title="estimate against real claim (4,000 test claims)")
axes[0].xaxis.set_major_formatter(RUPEES)
axes[0].yaxis.set_major_formatter(RUPEES)
tidy(axes[0], "both")
signed = 100 * (predicted / actual - 1)
axes[1].hist(np.clip(signed, -40, 40), bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
axes[1].axvline(0, color=INK2, linewidth=1)
axes[1].set(xlabel="estimate compared with the real claim (%, cut at -40 and +40)", ylabel="test claims", title="how far off the estimates are")
tidy(axes[1], "y")
figure.tight_layout()
plt.show()
print(f"middle of the signed gaps: {np.median(signed):+.2f} %   (0 would mean no leaning towards too high or too low)")

# %% [markdown]
# ## 7. Why gradient boosting, and why it learns a share of the IDV
# The same test claims, estimated four ways. The first three are simple on purpose; the fourth is the same kind of model as ours,
# but it learns the rupees directly instead of the share of the IDV.

# %%
rows = claims.amount_baselines(KIND, X_train, y_train, X_test, y_test, model)
code_numbers = model.features_.transform(X_train)
rupee_model = HistGradientBoostingRegressor(max_iter=claims.MAX_ROUNDS, learning_rate=0.1, early_stopping=True, random_state=0,
                                            categorical_features=claims.categorical_positions(KIND, TASK))
rupee_model.fit(code_numbers, y_train)
direct = claims.score_amount(y_test, rupee_model.predict(model.features_.transform(X_test)))
rows.insert(2, {"name": "Gradient boosting that learns rupees directly", **direct})
display(pd.DataFrame([{"method": r["name"], "R2": round(r["r2"], 4), "RMSE": inr(r["rmse"]), "typical error": f"{r['median_pct_error']:.1f} %",
                       "within 10 %": f"{r['within_10']:.0f} %"} for r in rows]).set_index("method"))
linear = rows[1]
assert score["rmse"] < direct["rmse"] and score["median_pct_error"] < direct["median_pct_error"], "The text below says that learning the share is better; check it."
display(Markdown(f"Learning the share instead of the rupees lowers the RMSE from {inr(direct['rmse'])} to {inr(score['rmse'])} and the typical "
                 f"error from {direct['median_pct_error']:.1f} % to {score['median_pct_error']:.1f} %. Straight-line rules (linear regression) end at a typical "
                 f"error of {linear['median_pct_error']:.1f} %: trees can let one detail matter differently for different damage types, a straight-line rule cannot."))

# %% [markdown]
# ## 8. Would more columns help?
# The claims file has more columns than the five the model uses (city, state, the make and model, the year, the premium paid, the day
# of the week, and so on). Were they left out for a good reason? The test: give the same kind of model **every other column** of the
# file, except the answers and the date, and measure it on the same test claims.

# %%
raw = pd.read_csv(claims.raw_path(KIND)).drop_duplicates("record_id").set_index("record_id")
used = set(claims.input_columns(KIND, TASK)) | set(claims.feature_names(KIND, TASK)) | set(claims.lookup_columns(TASK))
never = {"claim_amount", "claim_approved", "approval_probability", "affected_parts", "incident_date"}       # answers, a hidden score, a list of words, a date
extra = [column for column in raw.columns if column not in used | never]
words = [column for column in extra if not pd.api.types.is_numeric_dtype(raw[column])]
print(f"{len(extra)} more columns ({len(words)} of them hold words): {', '.join(extra)}")

# %%
more_train = raw.loc[train["record_id"], extra].reset_index(drop=True)
more_test = raw.loc[test["record_id"], extra].reset_index(drop=True)
for column in words:                                    # words -> code numbers, learned from the training claims only
    codes = {word: number for number, word in enumerate(sorted(more_train[column].dropna().unique()))}
    more_train[column] = more_train[column].map(codes).astype(float)
    more_test[column] = more_test[column].map(codes).astype(float)
learn = pd.concat([model.features_.transform(X_train).reset_index(drop=True), more_train.astype(float)], axis=1)
check = pd.concat([model.features_.transform(X_test).reset_index(drop=True), more_test.astype(float)], axis=1)
is_category = [column in claims.CATEGORY_COLUMNS or column in words for column in learn.columns]
wide = HistGradientBoostingRegressor(max_iter=claims.MAX_ROUNDS, learning_rate=0.1, early_stopping=True, random_state=0,
                                     categorical_features=is_category)
wide.fit(learn, np.log(y_train.to_numpy(dtype=float) / X_train["idv"].to_numpy(dtype=float)))
wide_score = claims.score_amount(y_test, np.exp(wide.predict(check)) * X_test["idv"].to_numpy(dtype=float))
display(pd.DataFrame(
    [{"model": f"the {len(X_train.columns)} details we use", "R2": round(score["r2"], 4), "RMSE": inr(score["rmse"]), "typical error": f"{score['median_pct_error']:.1f} %"},
     {"model": f"the same plus {len(extra)} more columns", "R2": round(wide_score["r2"], 4), "RMSE": inr(wide_score["rmse"]),
      "typical error": f"{wide_score['median_pct_error']:.1f} %"}]).set_index("model"))

# %%
assert wide_score["median_pct_error"] > score["median_pct_error"] - 0.5, "The text below says that the extra columns do not help; check it."
display(Markdown(f"With {len(extra)} more columns the typical error is {wide_score['median_pct_error']:.1f} % (with the five details: {score['median_pct_error']:.1f} %) and the RMSE "
                 f"{inr(wide_score['rmse'])} (with the five details: {inr(score['rmse'])}). The extra columns do not make the estimates better, so this test finds nothing "
                 "in them that explains the part of the claim the five details leave open. Keeping the model to five details also keeps the claim page short: "
                 "a customer is asked only for what matters."))

# %% [markdown]
# ## 9. SHAP: why does the model give this amount?
# SHAP answers: "for this claim, how much did each detail push the estimate up or down?" It starts from a **typical claim** and gives
# every detail its share of the difference, so that
#
#     typical claim + the push of every detail = the model's estimate, exactly.
#
# Two choices, both explained in `acko/claims_explain.py`:
#
# - The answer that is explained is the claim as a **share of the IDV**. Multiplied by the IDV of a claim, the pushes are rupees.
# - SHAP's fast `TreeExplainer` is **not** used. These trees split on categories (the damage type), and for such splits its pieces do
#   not add up to the model's answer; the check below shows by how much. SHAP's `Permutation` explainer only asks the model
#   questions, so it is exact for any model. It is slower, which is why 300 test claims are explained, not all of them.

# %%
background = explain.background_claims(model, TASK, X_train)          # 100 training claims that stand for a typical claim
sample = X_test.sample(explain.N_EXPLAINED, random_state=0)           # 300 test claims to explain
result = explain.explain(model, TASK, background, sample)
add_gap = explain.additivity_gap(result)
print(f"SHAP values for {len(sample)} test claims, computed in {result.seconds:.0f} s")
print(f"typical share of the IDV: {100 * result.base:.1f} %;   largest gap between 'typical + pushes' and the model's own answer: {add_gap:.1e}")
assert add_gap < 1e-9, "the pieces must add up to the model's answer"

# %%
# For comparison: SHAP's fast TreeExplainer on the same claims. It runs without an error, but its pieces do not add up.
try:
    tree = shap.TreeExplainer(model.booster_)
    tree_pieces = np.asarray(tree.shap_values(result.table))
    tree_gap = float(np.abs(np.ravel(tree.expected_value)[0] + tree_pieces.sum(axis=1) - model.booster_.predict(result.table)).max())
    print(f"TreeExplainer: largest gap between 'typical + pieces' and the model's own answer: {tree_gap:.1f} on the model's log scale "
          f"(the Permutation explainer above: {add_gap:.0e})")
except Exception as problem:
    print("TreeExplainer could not be used on this model:", type(problem).__name__)

# %% [markdown]
# **Which details matter most.** The bars show how far each detail moves the estimate on average, in points of the IDV (10 points =
# 10 % of the insured value). The picture is saved in `reports/shap/`.

# %%
figure, axis = plt.subplots(figsize=(8.5, 3.4))
explain.draw_importance(axis, result, "average push on the claim, in points of the IDV (10 points = 10 % of the insured value)")
axis.set_title(f"car claim amount: which details move the estimate (average over {len(sample)} test claims)")
figure.tight_layout()
print("saved: reports/shap/" + explain.save_figure(figure, f"claims_{KIND}_{TASK}_importance.png").name)
plt.show()

# %% [markdown]
# **Every test claim as a dot.** Left of the middle line the detail pulls the estimate down, right of it pushes it up. The colour is
# the value of the detail: red = high, blue = low. For the damage type, "high" means a dearer type of damage: the types are coloured
# in the order of how much of the IDV they cost, from the cheapest to the dearest (`claims_explain.colour_codes`).

# %%
damage_order = list(data.groupby("damage_type")["share"].median().sort_values().index)        # cheapest damage type first
shap.summary_plot(result.values, explain.colour_codes(model, result.table, damage_order), feature_names=result.labels, show=False,
                  max_display=len(result.names), plot_size=(9, 3.9))
plt.xlabel("push on the claim, as a share of the IDV (0.1 = 10 % of the insured value)", fontsize=9)
plt.title(f"car claim amount: the push of each detail on {len(sample)} test claims", fontsize=10)
print("saved: reports/shap/" + explain.save_figure(plt.gcf(), f"claims_{KIND}_{TASK}_summary.png").name)
plt.show()

# %% [markdown]
# **One claim, step by step.** The claim below is the one whose estimate is closest to the middle estimate of the 300. The grey bar on
# the left is a typical claim for the same IDV; each coloured bar is what one detail adds (blue) or takes off (orange), from the
# biggest push to the smallest; the grey bar on the right is the model's estimate.

# %%
rupees = result.output * sample["idv"].to_numpy(dtype=float)           # the model's estimate for each explained claim
position = int(np.argmin(np.abs(rupees - np.median(rupees))))
row = sample.iloc[position]
steps, start, end = explain.example_steps(result, position, row)
display(pd.DataFrame([{"detail": label, "this claim": detail, "push on the claim": ("+" if value >= 0 else "-") + inr(abs(value))}
                      for label, detail, value in steps]).set_index("detail"))
own_estimate = float(model.predict(sample.iloc[[position]])[0])
print(f"typical claim for this IDV {inr(start)}  +  the pushes  =  {inr(end)}   (the model's own estimate: {inr(own_estimate)})")
assert abs(start + sum(value for _, _, value in steps) - end) < 0.01
assert abs(end - own_estimate) < 0.01

# %%
figure, axis = plt.subplots(figsize=(9.5, 4.8))
explain.draw_waterfall(axis, steps, start, end, "typical claim\nfor this IDV", "model's\nestimate", money=True)
axis.set_title(f"one car claim: {row['damage_type']}, severity {int(row['damage_severity_score'])}, IDV {inr(row['idv'])}")
axis.set_ylabel("claim amount (Rs)")
figure.tight_layout()
print("saved: reports/shap/" + explain.save_figure(figure, f"claims_{KIND}_{TASK}_example.png").name)
plt.show()

# %% [markdown]
# ## 10. Checking the saved `.pkl` file
# The claim page does not train anything: it loads `models/claim_car_amount.pkl`. This step opens that file and checks that it
# gives the same answers as the model trained above, and that it meets the targets on its own.

# %%
try:
    bundle = claims.load_bundle(KIND, TASK)
except claims.ModelMissing as problem:
    bundle = None
    print(problem)
if bundle is not None:
    saved = bundle["model"]
    path = claims.MODEL_FILES[(KIND, TASK)]
    saved_metrics = claims.evaluate_amount(saved, KIND, X_test, y_test)["all"]
    difference = float(np.abs(np.asarray(saved.predict(X_test), dtype=float) - predicted).max())
    print(f"file: models/{path.name} ({path.stat().st_size / 1e6:.2f} MB), trained {bundle['trained_at']}, scikit-learn {bundle['versions']['scikit-learn']}")
    print(f"R2 of the saved model on the test claims: {saved_metrics['r2']:.4f}   (trained here: {score['r2']:.4f})")
    print(f"largest difference between the saved model and the one trained here: {inr(difference)}   (0 or a few paise: the same model)")
    assert saved_metrics["r2"] > claims.TARGET_R2 and saved_metrics["median_pct_error"] < claims.TARGET_MEDIAN_ERROR

# %% [markdown]
# And the way the claim page asks for an estimate: one claim in, an amount (never more than the IDV), an approval probability and a
# status out. Details that are skipped get typical values.

# %%
if bundle is not None:
    engine = claims.ClaimEngine.load()
    estimate = engine.estimate("car", "dent", 3, 2, 600_000, zero_dep=1)
    print("car, dent, severity 3, 2 parts, IDV Rs 6,00,000, zero-depreciation add-on:")
    print(f"   estimated claim {inr(estimate.amount)}  ({estimate.severity} damage)   approval probability {100 * estimate.probability:.1f} %  ->  {estimate.status}")

# %% [markdown]
# ## Summary and limits

# %%
facts = claims.approval_rule_check(KIND)
groups_note = f" Groups at or above the {claims.TARGET_MEDIAN_ERROR:g} % bar: {', '.join(weak)}." if weak else ""
display(Markdown(f"""
**Car claim amount model**

| | |
|---|---|
| model | gradient boosting on the share of the IDV (`HistGradientBoostingRegressor`), {model.n_rounds_} rounds |
| test claims | {score['n']:,} |
| R2 | {score['r2']:.4f} (target: above {claims.TARGET_R2}) |
| RMSE | {inr(score['rmse'])} |
| typical error | {score['median_pct_error']:.1f} % (target: below {claims.TARGET_MEDIAN_ERROR:g} %) |
| targets | {'MET' if met else 'NOT MET'} |
| saved model | `models/claim_car_amount.pkl` |
| SHAP pictures | `reports/shap/claims_car_amount_importance.png`, `_summary.png`, `_example.png` |
| run time of this notebook | {(time.time() - STARTED) / 60:.1f} minutes |

**Limits you should know about**

- The data is synthetic. The model learns the rules of this dataset, not those of real Acko claims.
- The typical error of {score['median_pct_error']:.1f} % is what is left after the five details are used. None of the {len(extra)} other columns brought it down
  (section 8), so it looks like random variation in the data itself. The typical error grows as claims get smaller: {band_errors[0]:.1f} % in the IDV band
  "{metrics['bands'][0]['label']}" against {band_errors[-1]:.1f} % in "{metrics['bands'][-1]['label']}" (section 6).{groups_note}
- {facts['amount_above_idv']:.1f} % of the claims in the file are above the insured value. A claim is never paid above the IDV, so the claim page limits
  the estimate to the IDV and says so.
- The claim page takes three of the five details (damage type, severity, damaged parts) from the photo with Gemini. The model was trained on the
  dataset's own values, so a wrong reading of a photo gives a wrong estimate. The page shows what Gemini read and lets the customer correct it.
- The test claims were also used to look at other columns (section 8). A stricter project would keep a third set that is touched only once.
- SHAP explains how the model arrives at its estimate. It does not say why a real insurer would pay that amount.
"""))
