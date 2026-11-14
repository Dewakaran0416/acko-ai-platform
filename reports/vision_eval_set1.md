# Gemini Vision photo test (Module 3)

Run 1, started 2026-11-14 10:03:09. Model `gemini-3.1-flash-lite`, prompt fingerprint `da1198e1`.
All runs are listed in `reports/vision_eval_history.csv`; one row per photo is in `reports/vision_eval_results.csv`.

## How the test was done

- 9 photos (6 cars, 3 bikes; minor 1, moderate 2, major 6). They are ordinary photos, not rows of the claims data set.
- The labeller looked at each photo and wrote down the vehicle, the damage type and the severity band BEFORE Gemini was asked. `labels.csv` was last saved at 2026-11-14 10:02:28; this run started at 2026-11-14 10:03:09.
- Each photo was sent to Gemini once, with the same prompt as the claim page. Gemini was not told whether the photo shows a car or a bike.
- Damage types overlap (a hit front bumper can be bumper damage, collision damage or panel damage), so the labeller could write down other acceptable types for a photo before the run (column `also_ok`). The strict row counts only the single label.
- Severity band: Gemini gives a score from 1 to 10 and the app turns it into minor (1-3), moderate (4-6) or major (7-10).

## Result

| Check | Photos | Share |
|---|---|---|
| Car or bike right | 9 of 9 | 100 % |
| Damage type right, the single label only | 6 of 9 | 67 % |
| **Damage type right (the label or the labeller's alternates)** | **6 of 9** | **67 %** |
| Severity band right | 5 of 9 | 56 % |
| Severity band right or one band off | 9 of 9 | 100 % |
| Damage type (with alternates) and severity band both right | 3 of 9 | 33 % |

Brief target: at least 8 of 10 photos with the right damage type. Result: **NOT MET** (6 of 9).
With only 9 photos the real success rate could lie anywhere between about 35 % and 88 % (95 % interval), so this is an indication, not a guarantee.

Damaged parts were not labelled, so they were not scored.
Gemini answered in 2.7 s per photo (median), the slowest took 27.4 s.

## Photo by photo

| # | Photo | Source | Label | Gemini | Type | Severity | Gemini says |
|---|---|---|---|---|---|---|---|
| 1 | 2.webp |  | car, dent, minor | car, panel damage, 4 (moderate) | wrong | 1 band off | There is a significant dent and associated paint damage on the front fender, with minor scuffing visible on the bumper. |
| 2 | 3.jpg |  | bike, collision damage, major | bike, collision damage, 8 (major) | right | right | The motorcycle has sustained significant collision damage to the front fairing, visor, and headlight assembly. |
| 3 | 4.jpg |  | car, collision damage, major | car, collision damage, 9 (major) | right | right | The vehicle on the right has sustained severe front-end collision damage, including a crumpled bonnet, missing bumper, and damaged radiator. |
| 4 | 5.jpg |  | car, collision damage, major | car, total loss, 10 (major) | wrong | right | The vehicle has sustained catastrophic front-end impact damage and is in a state consistent with a total loss. |
| 5 | 6.jpg |  | car, collision damage, major | car, collision damage, 5 (moderate) | right | 1 band off | The vehicle has sustained collision damage to the bonnet and front bumper, including visible denting and paint scuffing. |
| 6 | 7.jpg |  | car, collision damage, major | car, collision damage, 7 (major) | right | right | The vehicle has sustained significant collision damage to the front bumper, left fender, and side mirror after leaving the road. |
| 7 | 8.jpg |  | car, collision damage, major | car, panel damage, 8 (major) | wrong | right | The vehicle has sustained significant structural damage to the roof, bonnet, and windshield due to a fallen tree. |
| 8 | 9.jpg |  | bike, collision damage, moderate | bike, collision damage, 8 (major) | right | 1 band off | The motorcycle is lying on its side on the road with significant impact damage to the front fairing, visor, and surrounding components. |
| 9 | 10.jpg |  | bike, collision damage, moderate | bike, collision damage, 8 (major) | right | 1 band off | The motorcycle has sustained severe collision damage, resulting in significant bodywork destruction and debris scattered across the road. |

## Notes

- No photo has a 'source' yet: write where the photos came from (own photos, a website, ...) before you hand in.
- The brief asks for 10 test photos; 9 are labelled.

## Limits of this test

- One person labelled the photos, and the damage types overlap, so some labels are a matter of opinion.
- A small number of photos, chosen by hand and not at random (see the interval above).
- Gemini's answers can change a little from run to run; every run is kept in the history file.
- If the prompt in `acko/vision.py` changes, the prompt fingerprint changes: a change between two runs is visible in the history.
