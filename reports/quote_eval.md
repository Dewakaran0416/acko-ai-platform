# Module 2: premium quote model - evaluation

Created by `python -m scripts.train_quote` on 2026-11-13 21:53 (Asia/Kolkata), running 59 seconds.

## What was built
- **Data:** the course dataset (synthetic prices, not real Acko prices). `acko_car_quotation.csv`: 199,994 rows used (6 repeated record ids removed); `acko_bike_quotation.csv`: 149,999 rows used (1 repeated record id removed).
- **Feature engineering** (`QuoteFeatures`, a scikit-learn transformer in `acko/quote.py`): vehicle age = 2025 - year of manufacture; city risk score, engine size and body type come from look-up tables learned from the training rows; cover type, fuel type, body type and bike use become code numbers; IDV, NCB, add-ons, past claims and driver age are passed on. Missing optional answers get the typical value.
- **Model:** gradient boosting (`HistGradientBoostingRegressor`, up to 1000 rounds, stopping early when 10 % of the training rows show no more gain) that learns log(premium); one model for cars (765 rounds) and one for bikes (541 rounds), saved together in `models/quote_model.pkl`.
- **Test method:** 80 % of the rows to learn from, 20 % held back (random_state 42). The held-back rows were never used to train the model or to stop its training. (While exploring, a few extra columns - record id, colour, variant, state, previous insurer - were tried and none lowered the error on these rows, so the final model uses only the answers of the quote form plus the look-ups described above.)
- **Left out on purpose:** the five premium parts (`od_premium_before_ncb`, `ncb_discount_amount`, `tp_premium`, `addon_premium`, `gst_amount`). They are pieces of the answer and unknown when a customer asks for a quote, so using them would be cheating.
- **Brief targets:** R2 above 0.85 and RMSE within ₹2,000 on a held-out test set. The quote form accepts an IDV up to ₹10,00,000, so the targets are checked on the test rows in that range.

## Result

Test rows with IDV up to ₹10,00,000 (the range of the quote form):

| Vehicle | Test rows | R2 | RMSE | MAE | Typical error | Target met |
|---|---:|---:|---:|---:|---:|:---:|
| car | 24,145 | 0.9912 | ₹1,323 | ₹882 | 3.3 % | yes |
| bike | 29,733 | 0.9939 | ₹539 | ₹292 | 3.6 % | yes |

All test rows, every IDV (the data also contains cars worth up to 2.4 crore):

| Vehicle | Test rows | R2 | RMSE | MAE | Typical error |
|---|---:|---:|---:|---:|---:|
| car | 39,999 | 0.9965 | ₹4,780 | ₹1,900 | 3.1 % |
| bike | 30,000 | 0.9933 | ₹702 | ₹314 | 3.6 % |

The rupee error grows with the price (a 25-lakh car has a premium of about a lakh), while the percentage error stays about the same. That is why the RMSE target is checked inside the form's IDV range.

### Car: error by IDV band

| IDV band | Test rows | R2 | RMSE | Typical error |
|---|---:|---:|---:|---:|
| 1 to 3 lakh | 7,590 | 0.9742 | ₹1,119 | 3.8 % |
| 3 to 6 lakh | 10,626 | 0.9871 | ₹1,280 | 3.3 % |
| 6 to 10 lakh | 5,912 | 0.9928 | ₹1,610 | 3.0 % |
| 10 to 25 lakh | 8,925 | 0.9951 | ₹2,799 | 2.8 % |
| above 25 lakh | 6,929 | 0.9943 | ₹10,757 | 2.8 % |

### Bike: error by IDV band

| IDV band | Test rows | R2 | RMSE | Typical error |
|---|---:|---:|---:|---:|
| up to 1 lakh | 22,059 | 0.9688 | ₹379 | 3.9 % |
| 1 to 3 lakh | 5,784 | 0.9911 | ₹579 | 3.2 % |
| 3 to 6 lakh | 1,423 | 0.9921 | ₹1,176 | 3.1 % |
| 6 to 10 lakh | 467 | 0.9907 | ₹1,839 | 2.9 % |
| 10 to 25 lakh | 246 | 0.9816 | ₹4,597 | 3.6 % |

## Why gradient boosting

Same test rows (IDV up to the form limit), three ways of predicting:

| Vehicle | Method | R2 | RMSE |
|---|---|---:|---:|
| car | Guess the median premium for everyone | -0.0713 | ₹14,588 |
| car | Linear regression (straight-line rules) | 0.6219 | ₹8,666 |
| car | Gradient boosting (this project's model) | 0.9912 | ₹1,323 |
| bike | Guess the median premium for everyone | -0.0855 | ₹7,216 |
| bike | Linear regression (straight-line rules) | 0.8307 | ₹2,850 |
| bike | Gradient boosting (this project's model) | 0.9939 | ₹539 |

The premium is not a straight-line function of the inputs (cover type, IDV and NCB interact), so a model made of many small decision trees fits far better than a linear one.

## Which factors drive the premium (SHAP)

Average change of the premium that each factor causes (measured on 3,000 test policies per vehicle type):

- **Car:** Cover type 76.8 %, Insured value (IDV) 68.7 %, Vehicle type 19.0 %, No-claim bonus (NCB) 12.1 %, City risk 8.0 %, Driver age 5.2 %, Add-ons 5.0 %, Past claims 4.8 %, Vehicle age 4.5 %
- **Bike:** Cover type 51.9 %, Vehicle type 36.4 %, Insured value (IDV) 33.3 %, Add-ons 12.0 %, No-claim bonus (NCB) 8.6 %, City risk 5.4 %, Driver age 4.4 %, Bike use 3.7 %, Vehicle age 3.2 %, Past claims 2.3 %

SHAP pictures (also in `reports/shap/`):

- `reports/shap/quote_car_summary.png`
- `reports/shap/quote_car_importance.png`
- `reports/shap/quote_car_example.png`
- `reports/shap/quote_bike_summary.png`
- `reports/shap/quote_bike_importance.png`
- `reports/shap/quote_bike_example.png`

In the web page every quote shows the same kind of chart for that customer: a waterfall that starts at the typical premium and adds or removes rupees for IDV, city risk, vehicle age, NCB and the other factors until it reaches the quote. The bars add up exactly to the quote (SHAP values of the log-premium, converted to rupees with the logarithmic mean).

## Limits you should know about

- The prices in the dataset are synthetic; the model learns the dataset's pricing, not real Acko prices.
- The quote form is limited to an IDV of ₹10,00,000 (the brief). Above that the model still answers, but it was not checked there against the RMSE target.
- Vehicle age is counted as 2025 - year of manufacture, exactly as in the training data.
- The form accepts the brief's lowest IDV of Rs 50,000 for cars, although the lowest car in the data is ₹94,691; below that the page warns that the estimate is less reliable. For bikes the form starts at ₹15,000, because many bikes in the data are worth less than Rs 50,000.
- Only makes, models and fuel types that exist in the data are offered, so engine size and body type are always known.
- Library versions of this run: scikit-learn 1.9.1, pandas 3.1.1, numpy 2.4.6, shap 0.51.0, Python 3.11.8.
