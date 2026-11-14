"""Step 5 (Module 2): train the premium quote model and write its evaluation report.

Run from the project folder:
    python -m scripts.train_quote

It takes about 1 to 3 minutes and creates:
    models/quote_model.pkl     the trained model (one car model and one bike model, plus what the quote form needs)
    reports/quote_eval.md      the R2 / RMSE tables for your project report
    reports/shap/*.png         SHAP pictures: which factors drive the premium
"""
from __future__ import annotations

import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn

from acko import config, quote
from acko.fmt import inr

REPORT = config.ROOT / "reports" / "quote_eval.md"
SHAP_DIR = config.ROOT / "reports" / "shap"
TARGET_R2, TARGET_RMSE = 0.85, 2000          # the brief: R2 above 0.85 and RMSE within Rs 2,000 on held-out data


def rs(amount: float) -> str:
    return inr(amount, "Rs ")                 # plain letters for the terminal; the report file uses the rupee sign


def passed(metrics: dict) -> bool:
    inside = metrics["form_range"]
    return inside["r2"] > TARGET_R2 and inside["rmse"] <= TARGET_RMSE


def example_quote(model, kind: str, X_test: pd.DataFrame, explainer) -> quote.Quote:
    """A typical test policy (premium closest to the middle) explained in rupees, for the report picture."""
    inside = X_test[X_test["idv"] <= quote.FORM_IDV_MAX]
    predicted = model.predict(inside)
    pick = inside.iloc[[int(np.argmin(np.abs(predicted - np.median(predicted))))]]
    return quote.explain_frame(model, kind, pick, explainer)


def write_report(bundle: dict, pictures: list, seconds: float) -> None:
    kinds = list(bundle["models"])
    lines = [
        "# Module 2: premium quote model - evaluation",
        "",
        f"Created by `python -m scripts.train_quote` on {bundle['trained_at']} ({config.APP_TIMEZONE}), "
        f"running {seconds:.0f} seconds.",
        "",
        "## What was built",
        "- **Data:** the course dataset (synthetic prices, not real Acko prices). " +
        "; ".join(f"`{bundle['data_info'][k]['file']}`: {bundle['data_info'][k]['rows_used']:,} rows used "
                  f"({bundle['data_info'][k]['repeated_ids_dropped']} repeated record id{'' if bundle['data_info'][k]['repeated_ids_dropped'] == 1 else 's'} removed)" for k in kinds) + ".",
        "- **Feature engineering** (`QuoteFeatures`, a scikit-learn transformer in `acko/quote.py`): vehicle age = 2025 - year of "
        "manufacture; city risk score, engine size and body type come from look-up tables learned from the training rows; "
        "cover type, fuel type, body type and bike use become code numbers; IDV, NCB, add-ons, past claims and driver age are passed on. "
        "Missing optional answers get the typical value.",
        f"- **Model:** gradient boosting (`HistGradientBoostingRegressor`, up to {quote.MAX_ROUNDS} rounds, stopping early when 10 % of the training rows "
        f"show no more gain) that learns log(premium); one model for cars ({bundle['rounds']['car']} rounds) and one for bikes "
        f"({bundle['rounds']['bike']} rounds), saved together in `models/quote_model.pkl`.",
        f"- **Test method:** {int((1 - quote.TEST_SIZE) * 100)} % of the rows to learn from, {int(quote.TEST_SIZE * 100)} % held back "
        f"(random_state {quote.RANDOM_STATE}). The held-back rows were never used to train the model or to stop its training. "
        "(While exploring, a few extra columns - record id, colour, variant, state, previous insurer - were tried and none lowered the error on these rows, "
        "so the final model uses only the answers of the quote form plus the look-ups described above.)",
        "- **Left out on purpose:** the five premium parts (`od_premium_before_ncb`, `ncb_discount_amount`, `tp_premium`, "
        "`addon_premium`, `gst_amount`). They are pieces of the answer and unknown when a customer asks for a quote, so using them would be cheating.",
        f"- **Brief targets:** R2 above {TARGET_R2} and RMSE within {inr(TARGET_RMSE)} on a held-out test set. "
        f"The quote form accepts an IDV up to {inr(quote.FORM_IDV_MAX)}, so the targets are checked on the test rows in that range.",
        "",
        "## Result",
        "",
        f"Test rows with IDV up to {inr(quote.FORM_IDV_MAX)} (the range of the quote form):",
        "",
        "| Vehicle | Test rows | R2 | RMSE | MAE | Typical error | Target met |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for k in kinds:
        m = bundle["metrics"][k]["form_range"]
        lines.append(f"| {k} | {m['n']:,} | {m['r2']:.4f} | {inr(m['rmse'])} | {inr(m['mae'])} | {m['median_pct_error']:.1f} % | "
                     f"{'yes' if passed(bundle['metrics'][k]) else 'NO'} |")
    lines += ["", "All test rows, every IDV (the data also contains cars worth up to 2.4 crore):", "",
              "| Vehicle | Test rows | R2 | RMSE | MAE | Typical error |", "|---|---:|---:|---:|---:|---:|"]
    for k in kinds:
        m = bundle["metrics"][k]["all"]
        lines.append(f"| {k} | {m['n']:,} | {m['r2']:.4f} | {inr(m['rmse'])} | {inr(m['mae'])} | {m['median_pct_error']:.1f} % |")
    lines += ["", "The rupee error grows with the price (a 25-lakh car has a premium of about a lakh), while the percentage error stays "
              "about the same. That is why the RMSE target is checked inside the form's IDV range."]
    for k in kinds:
        lines += ["", f"### {k.capitalize()}: error by IDV band", "", "| IDV band | Test rows | R2 | RMSE | Typical error |",
                  "|---|---:|---:|---:|---:|"]
        for b in bundle["metrics"][k]["bands"]:
            lines.append(f"| {b['label']} | {b['n']:,} | {b['r2']:.4f} | {inr(b['rmse'])} | {b['median_pct_error']:.1f} % |")
    lines += ["", "## Why gradient boosting", "",
              "Same test rows (IDV up to the form limit), three ways of predicting:", "",
              "| Vehicle | Method | R2 | RMSE |", "|---|---|---:|---:|"]
    for k in kinds:
        for b in bundle["baselines"][k]:
            lines.append(f"| {k} | {b['name']} | {b['r2']:.4f} | {inr(b['rmse'])} |")
    lines += ["", "The premium is not a straight-line function of the inputs (cover type, IDV and NCB interact), "
              "so a model made of many small decision trees fits far better than a linear one.",
              "", "## Which factors drive the premium (SHAP)", "",
              "Average change of the premium that each factor causes (measured on 3,000 test policies per vehicle type):", ""]
    for k in kinds:
        lines.append(f"- **{k.capitalize()}:** " + ", ".join(f"{label} {value:.1f} %" for label, value in bundle["importance"][k]))
    lines += ["", "SHAP pictures (also in `reports/shap/`):", ""]
    lines += [f"- `reports/shap/{p.name}`" for p in pictures]
    lines += ["", "In the web page every quote shows the same kind of chart for that customer: a waterfall that starts at the typical premium "
              "and adds or removes rupees for IDV, city risk, vehicle age, NCB and the other factors until it reaches the quote. "
              "The bars add up exactly to the quote (SHAP values of the log-premium, converted to rupees with the logarithmic mean).",
              "", "## Limits you should know about", "",
              "- The prices in the dataset are synthetic; the model learns the dataset's pricing, not real Acko prices.",
              f"- The quote form is limited to an IDV of {inr(quote.FORM_IDV_MAX)} (the brief). Above that the model still answers, "
              "but it was not checked there against the RMSE target.",
              f"- Vehicle age is counted as {quote.PRICING_YEAR} - year of manufacture, exactly as in the training data.",
              "- The form accepts the brief's lowest IDV of Rs 50,000 for cars, although the lowest car in the data is "
              f"{inr(bundle['options']['car']['idv_seen']['min'])}; below that the page warns that the estimate is less reliable. "
              f"For bikes the form starts at {inr(bundle['options']['bike']['idv_form']['min'])}, because many bikes in the data are worth less than Rs 50,000.",
              "- Only makes, models and fuel types that exist in the data are offered, so engine size and body type are always known.",
              f"- Library versions of this run: scikit-learn {bundle['versions']['scikit-learn']}, pandas {bundle['versions']['pandas']}, "
              f"numpy {bundle['versions']['numpy']}, shap {bundle['versions']['shap']}, Python {bundle['versions']['python']}."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import shap

    started = time.time()
    print("Module 2: training the premium quote model\n")
    bundle = {"format": quote.BUNDLE_FORMAT, "trained_at": config.now_local().strftime("%Y-%m-%d %H:%M"),
              "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                           "numpy": np.__version__, "shap": shap.__version__},
              "settings": {"test_size": quote.TEST_SIZE, "random_state": quote.RANDOM_STATE,
                           "pricing_year": quote.PRICING_YEAR, "form_idv_max": quote.FORM_IDV_MAX},
              "models": {}, "options": {}, "metrics": {}, "baselines": {}, "data_info": {}, "importance": {}, "rounds": {}}
    pictures = []
    for kind in quote.KINDS:
        print(f"--- {kind.upper()} ---")
        df, info = quote.load_training_data(kind)
        train, test = quote.split_data(df)
        X_train, y_train = quote.xy(train, kind)
        X_test, y_test = quote.xy(test, kind)
        print(f"  {info['file']}: {info['rows_read']:,} rows read; removed {info['repeated_ids_dropped']} rows with a repeated id "
              f"and {info['incomplete_rows_dropped']} incomplete rows")
        print(f"  {len(train):,} rows to learn from, {len(test):,} rows held back for testing")
        step = time.time()
        model = quote.build_model(kind).fit(X_train, y_train)
        rounds = model.regressor_.named_steps["model"].n_iter_
        print(f"  trained in {time.time() - step:.0f} s ({rounds} boosting rounds)")
        metrics = quote.evaluate(model, X_test, y_test)
        baselines = quote.compare_baselines(kind, X_train, y_train, X_test, y_test, model)
        inside, everything = metrics["form_range"], metrics["all"]
        print(f"  test rows with IDV up to {rs(quote.FORM_IDV_MAX)} (the quote form's range): {inside['n']:,}")
        print(f"      R2 = {inside['r2']:.4f}    RMSE = {rs(inside['rmse'])}    typical error {inside['median_pct_error']:.1f} %"
              f"      -> {'PASS' if passed(metrics) else 'FAIL'}  (target: R2 > {TARGET_R2}, RMSE <= {rs(TARGET_RMSE)})")
        print(f"  all {everything['n']:,} test rows, any IDV:  R2 = {everything['r2']:.4f}    RMSE = {rs(everything['rmse'])}")
        print("  same rows, other methods:  " + "   |   ".join(f"{b['name'].split(' (')[0]}: R2 {b['r2']:.2f}" for b in baselines))
        explainer = quote.make_explainer(model)
        sample = X_test.sample(min(3000, len(X_test)), random_state=0)
        step = time.time()
        pictures += quote.save_shap_pictures(kind, model, sample, SHAP_DIR, example_quote(model, kind, X_test, explainer))
        bundle["importance"][kind] = quote.grouped_effects(model, sample, explainer)[0]
        print(f"  SHAP pictures saved in {time.time() - step:.0f} s;  biggest factors: " +
              ", ".join(f"{label} {value:.0f} %" for label, value in bundle["importance"][kind][:4]))
        bundle["models"][kind], bundle["metrics"][kind], bundle["baselines"][kind] = model, metrics, baselines
        bundle["rounds"][kind] = int(rounds)
        bundle["options"][kind] = quote.build_options(df, kind)
        bundle["data_info"][kind] = info
        print()

    path = quote.save_bundle(bundle)
    engine = quote.QuoteEngine.load(path)         # open the file again, the way the web page will
    make, model, fuel = engine.options("car")["default_vehicle"]
    check = engine.quote("car", make, model, fuel, 2020, "Mumbai", 450_000, 20)
    print(f"Saved {path.relative_to(config.ROOT)} ({path.stat().st_size / 1e6:.1f} MB); test quote after reloading it: {rs(check.premium)}")
    write_report(bundle, pictures, time.time() - started)
    print(f"Saved {REPORT.relative_to(config.ROOT)} and {len(pictures)} pictures in {SHAP_DIR.relative_to(config.ROOT)}")
    ok = all(passed(bundle["metrics"][k]) for k in quote.KINDS)
    print("\nBrief targets (R2 > 0.85, RMSE within Rs 2,000 inside the form's IDV range): " + ("ALL MET" if ok else "NOT MET"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
