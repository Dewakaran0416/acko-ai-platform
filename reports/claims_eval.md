# Module 3: claims models - evaluation

Created by `python -m scripts.train_claims` on 2026-11-14 16:54 (Asia/Kolkata), running 10 seconds.

## What was built
- **Data:** the course claims files (synthetic, not real Acko claims). `acko_car_claims.csv`: 149,995 rows used (5 repeated record ids removed); `acko_bike_claims.csv`: 100,000 rows used (0 repeated record ids removed).
- **Four models, one file each:** car claim amount, car approval, bike claim amount, bike approval (`models/claim_<vehicle>_<amount|approval>.pkl`). A claim is routed to the pair of its vehicle type, because cars and bikes differ in IDV, parts and risk.
- **Claim amount** (`ClaimAmountModel`, gradient boosting): it learns the claim as a share of the insured value, log(amount / IDV), and multiplies by the IDV again. Learning the share is more accurate than learning rupees directly (the claims notebooks compare the two), because the IDV of a car runs from about 90,000 to over 2 crore. Details used: damage type, damage severity (1 to 10), number of damaged parts, IDV, zero-depreciation add-on.
- **Approval** (`ClaimApprovalModel`, `HistGradientBoostingClassifier`): it gives the probability that the claim is approved. A probability of 0.5 or more counts as 'approved'; a lower one sends the claim to a human ('review'). The threshold is a setting (`APPROVAL_THRESHOLD`, default 0.5). Details used: the five above plus earlier claims, policy age in months, time of day, NCB and the city risk score; for bikes also bike use, helmet worn and rider experience.
- **Test method:** 80 % of the rows to learn from, 20 % held back (random_state 42). The held-back rows were never used to train a model or to stop its training. While exploring, a model that used every column was also scored on these rows to see whether the extra columns help (they did not), so the held-back rows influenced that choice a little.
- **Left out on purpose:** the file's own `approval_probability` score. It decides the approved / rejected column exactly (100.0 % of car rows and 100.0 % of bike rows are approved exactly when it is 0.5 or more), so it is the answer in disguise and unknown when a customer files a claim. The two answers (`claim_amount`, `claim_approved`) are of course not inputs either.
- **Not used, because they changed nothing when tested:** state, city, vehicle make and model, vehicle age, engine size, body type, cover type, customer age, premium paid, incident type, day and month. (The city risk score stays in the approval model because the brief names it.)

## Claim amount

Target for this project: R2 above 0.85 and a typical (median) error below 15 % of the claim (the brief only asks for an "acceptable" RMSE).

| Vehicle | Test rows | R2 | RMSE | MAE | Average claim | Typical error | Within 10 % | Within 25 % | Target met |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| car | 29,999 | 0.9971 | ₹57,972 | ₹21,292 | ₹4,82,209 | 5.1 % | 82 % | 98 % | yes |
| bike | 20,000 | 0.9959 | ₹4,613 | ₹2,338 | ₹29,723 | 10.8 % | 47 % | 79 % | yes |

The RMSE is dominated by the few very large claims (total loss of an expensive vehicle). The percentage error is the fairer picture. The error that is left looks like chance: in the tests, a model that was also given every other column of the data was no more accurate.

### Car: error by IDV band

| IDV band | Test rows | R2 | RMSE | Typical error |
|---|---:|---:|---:|---:|
| up to 3 lakh | 6,007 | 0.9874 | ₹8,357 | 7.5 % |
| 3 to 6 lakh | 7,716 | 0.9934 | ₹11,736 | 5.7 % |
| 6 to 10 lakh | 4,509 | 0.9950 | ₹18,352 | 4.8 % |
| 10 to 25 lakh | 6,657 | 0.9957 | ₹35,750 | 4.3 % |
| above 25 lakh | 5,110 | 0.9963 | ₹1,32,202 | 4.0 % |

### Bike: error by IDV band

| IDV band | Test rows | R2 | RMSE | Typical error |
|---|---:|---:|---:|---:|
| up to 50,000 | 8,913 | 0.9472 | ₹2,396 | 16.1 % |
| 50,000 to 1 lakh | 5,864 | 0.9822 | ₹2,798 | 10.8 % |
| 1 to 2 lakh | 2,850 | 0.9913 | ₹3,518 | 7.9 % |
| above 2 lakh | 2,373 | 0.9961 | ₹11,116 | 5.0 % |

Same test rows, three ways of predicting the amount:

| Vehicle | Method | R2 | RMSE | Typical error |
|---|---|---:|---:|---:|
| car | Typical share of the IDV for everyone | 0.3027 | ₹9,05,267 | 72.9 % |
| car | Linear regression (straight-line rules) | 0.9238 | ₹2,99,314 | 9.5 % |
| car | Gradient boosting (this project's model) | 0.9971 | ₹57,972 | 5.1 % |
| bike | Typical share of the IDV for everyone | 0.3190 | ₹59,133 | 66.5 % |
| bike | Linear regression (straight-line rules) | 0.6455 | ₹42,664 | 22.0 % |
| bike | Gradient boosting (this project's model) | 0.9959 | ₹4,613 | 10.8 % |

## Claim approval

Target from the brief: accuracy and F1 above 80 %. F1 is given for each class and as the average of both (macro); the target is checked on the macro F1, the stricter of the usual averages. Threshold 0.5.

| Vehicle | Test rows | Accuracy | F1 approved | F1 rejected | F1 macro | F1 weighted | AUC | Target met |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| car | 29,999 | 94.2 % | 0.966 | 0.792 | 0.879 | 0.941 | 0.973 | yes |
| bike | 20,000 | 94.5 % | 0.969 | 0.745 | 0.857 | 0.944 | 0.971 | yes |

Rejected claims are the rarer class (14 % of car claims and 11 % of bike claims), and they are harder to spot: the F1 of the 'rejected' class is below 0.80 for both vehicle types. The numbers that meet the 80 % target are accuracy, F1 of the approved class, weighted F1 and macro F1.

### Car: what the model did with the 29,999 test claims (threshold 0.5)

| | Predicted rejected / review | Predicted approved |
|---|---:|---:|
| **Really rejected** | 3,324 | 985 |
| **Really approved** | 763 | 24,927 |

### Bike: what the model did with the 20,000 test claims (threshold 0.5)

| | Predicted rejected / review | Predicted approved |
|---|---:|---:|
| **Really rejected** | 1,592 | 688 |
| **Really approved** | 404 | 17,316 |

### The threshold can be changed

The brief asks for a configurable threshold. Same test rows at other thresholds (higher = more claims go to a human):

| Vehicle | Threshold | Accuracy | F1 rejected | F1 macro | Claims predicted approved |
|---|---:|---:|---:|---:|---:|
| car | 0.3 | 93.6 % | 0.739 | 0.851 | 90 % |
| car | 0.4 | 94.0 % | 0.775 | 0.870 | 88 % |
| car | 0.5 | 94.2 % | 0.792 | 0.879 | 86 % |
| car | 0.6 | 94.0 % | 0.794 | 0.879 | 85 % |
| car | 0.7 | 93.3 % | 0.785 | 0.872 | 83 % |
| car | 0.8 | 92.0 % | 0.762 | 0.857 | 81 % |
| bike | 0.3 | 94.0 % | 0.674 | 0.820 | 93 % |
| bike | 0.4 | 94.3 % | 0.715 | 0.842 | 92 % |
| bike | 0.5 | 94.5 % | 0.745 | 0.857 | 90 % |
| bike | 0.6 | 94.4 % | 0.758 | 0.863 | 88 % |
| bike | 0.7 | 93.8 % | 0.751 | 0.858 | 87 % |
| bike | 0.8 | 92.6 % | 0.729 | 0.843 | 84 % |

Same test rows, three ways of deciding:

| Vehicle | Method | Accuracy | F1 rejected | F1 macro | AUC |
|---|---|---:|---:|---:|---:|
| car | Approve every claim | 85.6 % | 0.000 | 0.461 | 0.500 |
| car | Logistic regression (straight-line rules) | 93.4 % | 0.757 | 0.860 | 0.963 |
| car | Gradient boosting (this project's model) | 94.2 % | 0.792 | 0.879 | 0.973 |
| bike | Approve every claim | 88.6 % | 0.000 | 0.470 | 0.500 |
| bike | Logistic regression (straight-line rules) | 93.9 % | 0.701 | 0.834 | 0.949 |
| bike | Gradient boosting (this project's model) | 94.5 % | 0.745 | 0.857 | 0.971 |

"Approve every claim" scores a high accuracy only because most claims in the data are approved; its F1 for the rejected class is 0 and its macro F1 is far below the target. That is why F1 is reported next to accuracy.

### How well can approval be predicted at all?

The file's hidden approval score is partly random. A model trained to predict that score itself, with the same claim details, explains only 78 % of its ups and downs for cars and 73 % of its ups and downs for bikes (R2 on the test rows). Thresholding it at 0.5 gives an accuracy of 94.2 % (car) and 94.6 % (bike), the same as the classifier's. So an accuracy of about 94 % is close to what any model can reach on this data.

### Can the probability be trusted?

Test claims grouped by the probability the model gave. If the model is honest, about that share of each group was really approved:

| Vehicle | Probability given | Claims | Average probability | Really approved |
|---|---|---:|---:|---:|
| car | 0 to 10 % | 1,676 | 4 % | 3 % |
| car | 10 to 30 % | 1,382 | 20 % | 20 % |
| car | 30 to 50 % | 1,029 | 39 % | 41 % |
| car | 50 to 70 % | 934 | 61 % | 64 % |
| car | 70 to 90 % | 2,241 | 82 % | 81 % |
| car | 90 to 100 % | 22,737 | 99 % | 99 % |
| bike | 0 to 10 % | 647 | 5 % | 3 % |
| bike | 10 to 30 % | 762 | 19 % | 19 % |
| bike | 30 to 50 % | 587 | 40 % | 41 % |
| bike | 50 to 70 % | 694 | 60 % | 60 % |
| bike | 70 to 90 % | 1,436 | 82 % | 81 % |
| bike | 90 to 100 % | 15,874 | 99 % | 99 % |

## Limits you should know about

- The claims data is synthetic; the models learn the dataset's rules, not real Acko claims.
- The photo is turned into three of the inputs (damage type, severity 1 to 10, number of damaged parts) by Gemini. The models were trained on the dataset's own values, so a wrong reading of a photo gives a wrong estimate. The claim page shows what Gemini read and lets the customer correct it before the estimate is made.
- In the data a few claims are above the insured value (5.4 % of cars, 4.4 % of bikes). The claim page never shows an amount above the IDV.
- The claim amount has a part of about 5 % (cars) and 11 % (bikes) that none of the columns explains; the models are close to that limit.
- Library versions of this run: scikit-learn 1.9.1, pandas 3.1.1, numpy 2.4.6, Python 3.11.8.
