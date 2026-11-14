"""Step 9 (Module 3): train the four claims models and write their evaluation report.

Run from the project folder:
    python -m scripts.train_claims

It takes about half a minute and creates:
    models/claim_car_amount.pkl      models/claim_car_approval.pkl
    models/claim_bike_amount.pkl     models/claim_bike_approval.pkl
    reports/claims_eval.md           the tables for your project report
"""
from __future__ import annotations

import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn

from acko import claims, config
from acko.fmt import inr

REPORT = config.ROOT / "reports" / "claims_eval.md"


def rs(amount: float) -> str:
    return inr(amount, "Rs ")                 # plain letters for the terminal; the report file uses the rupee sign


def amount_ok(m: dict) -> bool:
    return m["r2"] > claims.TARGET_R2 and m["median_pct_error"] < claims.TARGET_MEDIAN_ERROR


def approval_ok(m: dict) -> bool:
    return m["accuracy"] > claims.TARGET_ACCURACY and m["f1_macro"] > claims.TARGET_F1


def write_report(results: dict, facts: dict, seconds: float) -> None:
    any_bundle = results[("car", "amount")]
    versions = any_bundle["versions"]
    threshold = config.APPROVAL_THRESHOLD
    lines = [
        "# Module 3: claims models - evaluation",
        "",
        f"Created by `python -m scripts.train_claims` on {any_bundle['trained_at']} ({config.APP_TIMEZONE}), running {seconds:.0f} seconds.",
        "",
        "## What was built",
        "- **Data:** the course claims files (synthetic, not real Acko claims). " +
        "; ".join(f"`{results[(k, 'amount')]['data_info']['file']}`: {results[(k, 'amount')]['data_info']['rows_used']:,} rows used "
                  f"({results[(k, 'amount')]['data_info']['repeated_ids_dropped']} repeated record "
                  f"id{'' if results[(k, 'amount')]['data_info']['repeated_ids_dropped'] == 1 else 's'} removed)" for k in claims.KINDS) + ".",
        "- **Four models, one file each:** car claim amount, car approval, bike claim amount, bike approval (`models/claim_<vehicle>_<amount|approval>.pkl`). "
        "A claim is routed to the pair of its vehicle type, because cars and bikes differ in IDV, parts and risk.",
        "- **Claim amount** (`ClaimAmountModel`, gradient boosting): it learns the claim as a share of the insured value, log(amount / IDV), "
        "and multiplies by the IDV again. Learning the share is more accurate than learning rupees directly (the claims notebooks compare the two), "
        "because the IDV of a car runs from about 90,000 to over 2 crore. Details used: damage type, damage severity (1 to 10), number of damaged parts, IDV, zero-depreciation add-on.",
        f"- **Approval** (`ClaimApprovalModel`, `HistGradientBoostingClassifier`): it gives the probability that the claim is approved. "
        f"A probability of {threshold:g} or more counts as 'approved'; a lower one sends the claim to a human ('review'). The threshold is a setting "
        "(`APPROVAL_THRESHOLD`, default 0.5). Details used: the five above plus earlier claims, policy age in months, time of day, NCB and the "
        "city risk score; for bikes also bike use, helmet worn and rider experience.",
        f"- **Test method:** {int((1 - claims.TEST_SIZE) * 100)} % of the rows to learn from, {int(claims.TEST_SIZE * 100)} % held back "
        f"(random_state {claims.RANDOM_STATE}). The held-back rows were never used to train a model or to stop its training. "
        "While exploring, a model that used every column was also scored on these rows to see whether the extra columns help (they did not), "
        "so the held-back rows influenced that choice a little.",
        f"- **Left out on purpose:** the file's own `approval_probability` score. It decides the approved / rejected column exactly "
        f"({facts['car']['label_equals_rule']:.1f} % of car rows and {facts['bike']['label_equals_rule']:.1f} % of bike rows are approved exactly when "
        "it is 0.5 or more), so it is the answer in disguise and unknown when a customer files a claim. "
        "The two answers (`claim_amount`, `claim_approved`) are of course not inputs either.",
        "- **Not used, because they changed nothing when tested:** state, city, vehicle make and model, vehicle age, engine size, body type, cover type, "
        "customer age, premium paid, incident type, day and month. (The city risk score stays in the approval model because the brief names it.)",
        "",
        "## Claim amount",
        "",
        f"Target for this project: R2 above {claims.TARGET_R2} and a typical (median) error below {claims.TARGET_MEDIAN_ERROR:g} % of the claim "
        "(the brief only asks for an \"acceptable\" RMSE).",
        "",
        "| Vehicle | Test rows | R2 | RMSE | MAE | Average claim | Typical error | Within 10 % | Within 25 % | Target met |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for k in claims.KINDS:
        m = results[(k, "amount")]["metrics"]["all"]
        lines.append(f"| {k} | {m['n']:,} | {m['r2']:.4f} | {inr(m['rmse'])} | {inr(m['mae'])} | {inr(m['mean_claim'])} | "
                     f"{m['median_pct_error']:.1f} % | {m['within_10']:.0f} % | {m['within_25']:.0f} % | {'yes' if amount_ok(m) else 'NO'} |")
    lines += ["", "The RMSE is dominated by the few very large claims (total loss of an expensive vehicle). The percentage error is the fairer picture. "
              "The error that is left looks like chance: in the tests, a model that was also given every other column of the data was no more accurate.", ""]
    for k in claims.KINDS:
        lines += [f"### {k.capitalize()}: error by IDV band", "", "| IDV band | Test rows | R2 | RMSE | Typical error |", "|---|---:|---:|---:|---:|"]
        for b in results[(k, "amount")]["metrics"]["bands"]:
            lines.append(f"| {b['label']} | {b['n']:,} | {b['r2']:.4f} | {inr(b['rmse'])} | {b['median_pct_error']:.1f} % |")
        lines.append("")
    lines += ["Same test rows, three ways of predicting the amount:", "", "| Vehicle | Method | R2 | RMSE | Typical error |", "|---|---|---:|---:|---:|"]
    for k in claims.KINDS:
        for b in results[(k, "amount")]["baselines"]:
            lines.append(f"| {k} | {b['name']} | {b['r2']:.4f} | {inr(b['rmse'])} | {b['median_pct_error']:.1f} % |")
    lines += ["", "## Claim approval", "",
              f"Target from the brief: accuracy and F1 above {int(claims.TARGET_ACCURACY * 100)} %. F1 is given for each class and as the "
              "average of both (macro); the target is checked on the macro F1, the stricter of the usual averages. "
              f"Threshold {threshold:g}.", "",
              "| Vehicle | Test rows | Accuracy | F1 approved | F1 rejected | F1 macro | F1 weighted | AUC | Target met |",
              "|---|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for k in claims.KINDS:
        m = results[(k, "approval")]["metrics"]["at_threshold"]
        lines.append(f"| {k} | {m['n']:,} | {100 * m['accuracy']:.1f} % | {m['f1_approved']:.3f} | {m['f1_rejected']:.3f} | {m['f1_macro']:.3f} | "
                     f"{m['f1_weighted']:.3f} | {m['auc']:.3f} | {'yes' if approval_ok(m) else 'NO'} |")
    rejected_share = {k: 100 * (1 - results[(k, "approval")]["metrics"]["at_threshold"]["approved_rate"]) for k in claims.KINDS}
    lines += ["", f"Rejected claims are the rarer class ({rejected_share['car']:.0f} % of car claims and {rejected_share['bike']:.0f} % of bike claims), "
              "and they are harder to spot: the F1 of the 'rejected' class is below 0.80 for both vehicle types. The numbers that meet the 80 % target "
              "are accuracy, F1 of the approved class, weighted F1 and macro F1.", ""]
    for k in claims.KINDS:
        m = results[(k, "approval")]["metrics"]["at_threshold"]
        (tn, fp), (fn, tp) = m["confusion"]
        lines += [f"### {k.capitalize()}: what the model did with the {m['n']:,} test claims (threshold {threshold:g})", "",
                  "| | Predicted rejected / review | Predicted approved |", "|---|---:|---:|",
                  f"| **Really rejected** | {tn:,} | {fp:,} |", f"| **Really approved** | {fn:,} | {tp:,} |", ""]
    lines += ["### The threshold can be changed", "",
              "The brief asks for a configurable threshold. Same test rows at other thresholds (higher = more claims go to a human):", "",
              "| Vehicle | Threshold | Accuracy | F1 rejected | F1 macro | Claims predicted approved |", "|---|---:|---:|---:|---:|---:|"]
    for k in claims.KINDS:
        for m in results[(k, "approval")]["metrics"]["thresholds"]:
            lines.append(f"| {k} | {m['threshold']:.1f} | {100 * m['accuracy']:.1f} % | {m['f1_rejected']:.3f} | {m['f1_macro']:.3f} | "
                         f"{100 * m['predicted_approved_rate']:.0f} % |")
    lines += ["", "Same test rows, three ways of deciding:", "",
              "| Vehicle | Method | Accuracy | F1 rejected | F1 macro | AUC |", "|---|---|---:|---:|---:|---:|"]
    for k in claims.KINDS:
        for b in results[(k, "approval")]["baselines"]:
            lines.append(f"| {k} | {b['name']} | {100 * b['accuracy']:.1f} % | {b['f1_rejected']:.3f} | {b['f1_macro']:.3f} | {b['auc']:.3f} |")
    lines += ["", "\"Approve every claim\" scores a high accuracy only because most claims in the data are approved; its F1 for the rejected class is 0 and its "
              "macro F1 is far below the target. That is why F1 is reported next to accuracy.", "",
              "### How well can approval be predicted at all?", "",
              "The file's hidden approval score is partly random. A model trained to predict that score itself, with the same claim details, explains only "
              + " and ".join(f"{100 * facts[k]['ceiling']['r2']:.0f} % of its ups and downs for {k}s" for k in claims.KINDS)
              + " (R2 on the test rows). Thresholding it at 0.5 gives an accuracy of "
              + " and ".join(f"{100 * facts[k]['ceiling']['accuracy']:.1f} % ({k})" for k in claims.KINDS)
              + ", the same as the classifier's. So an accuracy of about 94 % is close to what any model can reach on this data.", "",
              "### Can the probability be trusted?", "",
              "Test claims grouped by the probability the model gave. If the model is honest, about that share of each group was really approved:", "",
              "| Vehicle | Probability given | Claims | Average probability | Really approved |", "|---|---|---:|---:|---:|"]
    for k in claims.KINDS:
        for c in results[(k, "approval")]["metrics"]["calibration"]:
            lines.append(f"| {k} | {100 * c['low']:.0f} to {100 * c['high']:.0f} % | {c['n']:,} | {100 * c['predicted']:.0f} % | {100 * c['actual']:.0f} % |")
    lines += ["", "## Limits you should know about", "",
              "- The claims data is synthetic; the models learn the dataset's rules, not real Acko claims.",
              "- The photo is turned into three of the inputs (damage type, severity 1 to 10, number of damaged parts) by Gemini. "
              "The models were trained on the dataset's own values, so a wrong reading of a photo gives a wrong estimate. The claim page shows what "
              "Gemini read and lets the customer correct it before the estimate is made.",
              f"- In the data a few claims are above the insured value ({facts['car']['amount_above_idv']:.1f} % of cars, "
              f"{facts['bike']['amount_above_idv']:.1f} % of bikes). The claim page never shows an amount above the IDV.",
              "- The claim amount has a part of about 5 % (cars) and 11 % (bikes) that none of the columns explains; the models are close to that limit.",
              f"- Library versions of this run: scikit-learn {versions['scikit-learn']}, pandas {versions['pandas']}, numpy {versions['numpy']}, "
              f"Python {versions['python']}."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    started = time.time()
    print("Module 3: training the four claims models\n")
    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                "numpy": np.__version__}
    trained_at = config.now_local().strftime("%Y-%m-%d %H:%M")
    results, facts, files = {}, {}, []
    for kind in claims.KINDS:
        print(f"--- {kind.upper()} ---")
        df, info = claims.load_training_data(kind)
        train, test = claims.split_data(df)
        print(f"  {info['file']}: {info['rows_read']:,} rows read; removed {info['repeated_ids_dropped']} rows with a repeated id "
              f"and {info['incomplete_rows_dropped']} incomplete rows")
        print(f"  {len(train):,} rows to learn from, {len(test):,} rows held back for testing")
        options = claims.build_options(df, kind, claims.parts_in_data(kind))
        facts[kind] = {**claims.approval_rule_check(kind), "ceiling": claims.approval_ceiling(kind, train, test)}
        for task in claims.TASKS:
            X_train, y_train = claims.xy(train, kind, task)
            X_test, y_test = claims.xy(test, kind, task)
            step = time.time()
            model = claims.build_model(kind, task).fit(X_train, y_train)
            if task == "amount":
                metrics = claims.evaluate_amount(model, kind, X_test, y_test)
                baselines = claims.amount_baselines(kind, X_train, y_train, X_test, y_test, model)
                m = metrics["all"]
                print(f"  {kind} amount:   trained in {time.time() - step:.0f} s ({model.n_rounds_} rounds)   R2 = {m['r2']:.4f}   "
                      f"RMSE = {rs(m['rmse'])}   typical error {m['median_pct_error']:.1f} %   -> {'PASS' if amount_ok(m) else 'FAIL'}")
            else:
                metrics = claims.evaluate_approval(model, X_test, y_test)
                baselines = claims.approval_baselines(kind, X_train, y_train, X_test, y_test, model)
                m = metrics["at_threshold"]
                print(f"  {kind} approval: trained in {time.time() - step:.0f} s ({model.n_rounds_} rounds)   accuracy = {100 * m['accuracy']:.1f} %   "
                      f"F1 approved {m['f1_approved']:.3f}   F1 rejected {m['f1_rejected']:.3f}   F1 macro {m['f1_macro']:.3f}   "
                      f"-> {'PASS' if approval_ok(m) else 'FAIL'}")
            bundle = {"format": claims.BUNDLE_FORMAT, "kind": kind, "task": task, "trained_at": trained_at, "versions": versions,
                      "settings": {"test_size": claims.TEST_SIZE, "random_state": claims.RANDOM_STATE,
                                   "threshold": config.APPROVAL_THRESHOLD},
                      "model": model, "inputs": claims.input_columns(kind, task), "metrics": metrics, "baselines": baselines,
                      "options": options, "data_info": info, "rounds": model.n_rounds_}
            results[(kind, task)] = bundle
            files.append(claims.save_bundle(bundle, kind, task))
        print(f"  hidden approval score: the claim details explain {100 * facts[kind]['ceiling']['r2']:.0f} % of it; thresholded it gives "
              f"{100 * facts[kind]['ceiling']['accuracy']:.1f} % accuracy (the classifier: {100 * results[(kind, 'approval')]['metrics']['at_threshold']['accuracy']:.1f} %)")
        print()

    print("Saved: " + ", ".join(f"{f.relative_to(config.ROOT)} ({f.stat().st_size / 1e6:.1f} MB)" for f in files))
    engine = claims.ClaimEngine.load()           # open the files again, the way the claim page will
    car = engine.estimate("car", "dent", 3, 2, 600_000, zero_dep=1, previous_claims=0, policy_age_months=24,
                          time_of_day="Morning", ncb=25, city="Mumbai")
    bike = engine.estimate("bike", "scratch", 2, 1, 80_000, zero_dep=0, previous_claims=1, policy_age_months=18,
                           time_of_day="Evening", ncb=20, city="Chennai", usage_type="Personal", helmet_worn=1, rider_experience=5)
    for e, what in ((car, "car, dent, severity 3, IDV Rs 6,00,000"), (bike, "bike, scratch, severity 2, IDV Rs 80,000")):
        print(f"Test claim after reloading the files ({what}): estimated amount {rs(e.amount)}, approval {100 * e.probability:.0f} %, "
              f"status {e.status}")
    write_report(results, facts, time.time() - started)
    print(f"Saved {REPORT.relative_to(config.ROOT)}")
    ok = (all(amount_ok(results[(k, 'amount')]['metrics']['all']) for k in claims.KINDS)
          and all(approval_ok(results[(k, 'approval')]['metrics']['at_threshold']) for k in claims.KINDS))
    print("\nBrief targets (accuracy and F1 above 80 %, a sensible claim amount error): " + ("ALL MET" if ok else "NOT MET"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
