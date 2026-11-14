# Module 1 evaluation: policy chatbot

Created: 2026-11-13 19:25 (Asia/Kolkata)

## MAIN TEST (20 questions from the brief and the policy PDFs)

| # | Found | Place | Question | Passage |
|---|---|---|---|---|
| 1 | OK | 2 | Is zero depreciation included in my base plan? | Motor_Insurance_Policy_TC p.11 |
| 2 | OK | 3 | What is the waiting period for Acko health insurance? | Insurance_FAQs p.8 |
| 3 | OK | 3 | How do I file a claim for my car? | Insurance_FAQs p.11 |
| 4 | OK | 3 | What add-ons are available for bike insurance? | Insurance_FAQs p.16 |
| 5 | OK | 1 | Does my bike insurance cover theft if I forgot to lock it? | Insurance_FAQs p.6 |
| 6 | OK | 1 | How soon must I report a stolen car to the police and to Acko? | Insurance_FAQs p.4 |
| 7 | OK | 1 | How many claims per year can I make with the Zero Depreciation add-on? | Motor_Insurance_Policy_TC p.11 |
| 8 | OK | 1 | Can I change the IDV that Acko calculates for my car? | Motor_Insurance_Policy_TC p.9 |
| 9 | OK | 1 | When does Acko treat a vehicle as a total loss? | Insurance_FAQs p.4 |
| 10 | OK | 1 | How far will Roadside Assistance tow my car? | Insurance_FAQs p.17 |
| 11 | OK | 5 | What happens to my No Claim Bonus if I make a claim? | Insurance_FAQs p.14 |
| 12 | OK | 2 | What is the initial waiting period in Acko health insurance? | Health_Insurance_Policy_TC p.4 |
| 13 | OK | 1 | How long is the waiting period for pre-existing diseases? | Health_Insurance_Policy_TC p.12 |
| 14 | OK | 1 | How long is the free look period for a new health policy? | Insurance_FAQs p.25 |
| 15 | OK | 1 | What is the grace period for renewing a health policy? | Health_Insurance_Policy_TC p.17 |
| 16 | OK | 1 | How many cashless network hospitals does Acko have? | Health_Insurance_Policy_TC p.20 |
| 17 | OK | 1 | What co-payment do senior citizens above 60 have to pay? | Health_Insurance_Policy_TC p.34 |
| 18 | OK | 1 | Is car insurance mandatory in India and what is the penalty for not having it? | Insurance_FAQs p.3 |
| 19 | OK | 1 | Does car insurance cover engine damage from flooding? | Insurance_FAQs p.5 |
| 20 | OK | 1 | How do I cancel my Acko policy and get a refund? | Insurance_FAQs p.18 |

| Search method | Relevant passage in top 5 | Average search time |
|---|---|---|
| hybrid (used by the chatbot) | 20 / 20  (100%) | 51 ms |
| meaning search only | 19 / 20  (95%) | 50 ms |
| keyword search only | 15 / 20  (75%) | 1 ms |

## EXTRA CHECK (45 more questions, written separately)


| Search method | Relevant passage in top 5 | Average search time |
|---|---|---|
| hybrid (used by the chatbot) | 44 / 45  (98%) | 55 ms |
| meaning search only | 45 / 45  (100%) | 51 ms |
| keyword search only | 38 / 45  (84%) | 1 ms |

Questions where the chatbot's search did not find a relevant passage:

- What are all the waiting periods? (best passage returned: Health_Insurance_Policy_TC p.12)

## Intent routing

20 / 20 example questions were routed correctly.


## Answers from Gemini

| # | Key fact in answer | Seconds | Question | Answer (first 160 characters) |
|---|---|---|---|---|
| 1 | yes | 1.8 | Is zero depreciation included in my base plan? | No, Zero Depreciation is not included in your base plan. It is an optional add-on cover that must be purchased separately and must appear in your Policy Schedul |
| 2 | yes | 1.6 | What is the waiting period for Acko health insurance? | Acko health insurance has five key waiting periods:  *   **Initial/Cooling-off:** 30 days (accidental injuries are covered from Day 1). *   **Pre-Existing Disea |
| 3 | yes | 1.8 | How do I file a claim for my car? | You can register a motor insurance claim with Acko 24x7 using these digital channels:  *   **Acko App (Recommended):** Go to My Policies, select your policy, an |
| 4 | yes | 1.6 | What add-ons are available for bike insurance? | For bike insurance, you can opt for the following add-ons:  *   **Pillion Rider PA Add-on:** Covers your pillion passenger for accidental death (INR 1,00,000) a |
| 5 | yes | 1.1 | Does my bike insurance cover theft if I forgot to lock it? | I could not find information in the policy documents regarding whether theft coverage is affected if the bike is left unlocked. Please contact Acko support for  |
| 6 | yes | 1.4 | How soon must I report a stolen car to the police and to Acko? | You must file an FIR at the police station immediately after the theft. You must also inform Acko via the app within 24 hours of discovering the theft. |
| 7 | yes | 1.2 | How many claims per year can I make with the Zero Depreciation add-on? | You can make a maximum of 2 claims per year under the Zero Depreciation add-on. On the 3rd claim in the same year, standard depreciation will apply. |
| 8 | yes | 1.4 | Can I change the IDV that Acko calculates for my car? | Yes, you can adjust the IDV calculated by Acko. You are allowed to set your IDV within a band of ±15% of the figure calculated by Acko.   Please note: * Setting |
| 9 | yes | 1.1 | When does Acko treat a vehicle as a total loss? | Acko declares a vehicle a Total Loss when the estimated repair cost exceeds 75% of the Insured Declared Value (IDV).   For example, if your car's IDV is INR 8 l |
| 10 | yes | 2.7 | How far will Roadside Assistance tow my car? | Roadside Assistance (RSA) will tow your vehicle to the nearest Acko network garage or repair shop for free up to 50 km. If the distance is beyond 50 km, you wil |
| 11 | yes | 5.9 | What happens to my No Claim Bonus if I make a claim? | If you make an Own Damage (OD) claim, your No Claim Bonus (NCB) resets to 0%. This applies even for small claims.  However, there are exceptions: *   **Third Pa |
| 12 | yes | 5.3 | What is the initial waiting period in Acko health insurance? | The initial waiting period for Acko health insurance is 30 days from the policy inception date. During this time, claims for illnesses are not payable. However, |
| 13 | yes | 1.6 | How long is the waiting period for pre-existing diseases? | The waiting period for pre-existing diseases is 36 months for standard plans and 12 months for Platinum plans. This applies to any illness, condition, or injury |
| 14 | yes | 1.5 | How long is the free look period for a new health policy? | The Free Look Period for a new health policy is:  *   15 days from the date you receive your Policy Document. *   30 days if your policy was sold entirely throu |
| 15 | yes | 1.2 | What is the grace period for renewing a health policy? | The grace period for renewing your health insurance policy is 30 days from the policy expiry date. Please note that there is no coverage during this 30-day grac |
| 16 | yes | 1.9 | How many cashless network hospitals does Acko have? | Acko has a network of over 10,000 cashless hospitals across India. You can check the current list of these hospitals on the Acko app. |
| 17 | yes | 1.3 | What co-payment do senior citizens above 60 have to pay? | For the Acko Senior Citizen Plan, there is a mandatory 20% co-payment on all admissible claims. This means Acko pays 80% of the claim amount, and you are respon |
| 18 | yes | 0.9 | Is car insurance mandatory in India and what is the penalty for not having it? | Yes, car insurance is mandatory in India under the Motor Vehicles Act, 1988. At a minimum, every vehicle on public roads must have a valid Third Party insurance |
| 19 | yes | 1.3 | Does car insurance cover engine damage from flooding? | Standard car insurance covers flood damage to the vehicle body, cabin, and electrical systems. However, engine damage caused by driving through a flooded road ( |
| 20 | yes | 1.3 | How do I cancel my Acko policy and get a refund? | To cancel your Acko motor insurance policy, you can follow these steps:  *   **Via Acko app:** Go to "My Policies," select your policy, tap "More Options," then |

Answers that contain the key fact: 20 / 20.  Gemini gave no answer: 0.  Answered in under 5 s: 18 / 20 (median 1.5 s, average 1.9 s, slowest 5.9 s).
