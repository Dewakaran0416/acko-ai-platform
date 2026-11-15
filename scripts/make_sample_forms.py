"""Step 10: make four SAMPLE claim forms (PDF) for testing the Gemini form reader.

    python -m scripts.make_sample_forms

The forms are made up (fictional policies, no real person) and every page says SAMPLE. They are saved in uploads/sample_forms/.
Next to each PDF a .json file keeps the answers that are printed on it, so scripts.test_vision can check what Gemini read.

  claim_form_car_full     every detail, with the vehicle type written on it
  claim_form_bike_full    every detail of a two-wheeler claim
  claim_form_car_brief    only the four details the brief names; the car shows only in the make and model
  claim_form_bike_brief   only the four details the brief names; the bike shows only in the damage description
"""
from __future__ import annotations

import json
import textwrap

import matplotlib

matplotlib.use("Agg")                                   # draw into files, never open a window
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages    # noqa: E402
from matplotlib.patches import Rectangle                # noqa: E402

from acko import config                                 # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42                # real, copyable text inside the PDF (the default draws the letters as shapes)
FOLDER = config.ROOT / "uploads" / "sample_forms"
RUPEE = chr(0x20B9)

SAMPLES = {
    "claim_form_car_full": {
        "heading": "Motor claim form - car",
        "rows": [("Policy number", "MOT-2025-048213"), ("Vehicle type", "Car"), ("Registration number", "TN 09 AB 1234"),
                 ("Make", "Hyundai"), ("Model", "Creta SX"), ("City", "Chennai"), ("Insured value (IDV)", "Rs. 8,50,000/-"),
                 ("Policy start date", "15/03/2025"), ("Date of incident", "02/11/2026"), ("Time of incident", "14:30"),
                 ("Earlier claims on this policy", "1"), ("Zero depreciation add-on", "Yes"), ("No Claim Bonus (NCB)", "25 %")],
        "description": "I hit a road divider while turning at a junction. The front bumper is cracked and the left headlight is broken.",
        "expect_kind": "car",
        "answers": {"policy_number": "MOT-2025-048213", "vehicle_type": "car", "vehicle_registration": "TN09AB1234",
                    "vehicle_make": "Hyundai", "vehicle_model": "Creta SX", "city": "Chennai", "idv": 850000,
                    "policy_start_date": "2025-03-15", "incident_date": "2026-11-02", "incident_time": "14:30",
                    "previous_claims": 1, "zero_dep": 1, "ncb": 25,
                    "description": "I hit a road divider while turning at a junction. The front bumper is cracked and the left headlight is broken."},
    },
    "claim_form_bike_full": {
        "heading": "Motor claim form - two-wheeler",
        "rows": [("Policy number", "MOT-2025-077120"), ("Vehicle type", "Two-wheeler (bike / scooter)"),
                 ("Registration number", "KA-03-MX-5678"), ("Make", "Royal Enfield"), ("Model", "Classic 350"),
                 ("City", "Bengaluru"), ("Insured value (IDV)", RUPEE + " 1,40,000"), ("Policy start date", "01/08/2025"),
                 ("Date of incident", "21 Oct 2026"), ("Time of incident", "08:10 AM"), ("Earlier claims on this policy", "0"),
                 ("Zero depreciation add-on", "No"), ("No Claim Bonus (NCB)", "20 %"), ("Bike use", "Personal"),
                 ("Helmet worn", "Yes"), ("Rider experience (years)", "6")],
        "description": "My bike skidded on a wet road and fell on the right side. The handlebar is bent and the headlight is cracked.",
        "expect_kind": "bike",
        "answers": {"policy_number": "MOT-2025-077120", "vehicle_type": "bike", "vehicle_registration": "KA03MX5678",
                    "vehicle_make": "Royal Enfield", "vehicle_model": "Classic 350", "city": "Bengaluru", "idv": 140000,
                    "policy_start_date": "2025-08-01", "incident_date": "2026-10-21", "incident_time": "08:10",
                    "previous_claims": 0, "zero_dep": 0, "ncb": 20, "usage_type": "Personal", "helmet_worn": 1,
                    "rider_experience": 6,
                    "description": "My bike skidded on a wet road and fell on the right side. The handlebar is bent and the headlight is cracked."},
    },
    "claim_form_car_brief": {
        "heading": "Motor claim form",
        "rows": [("Policy number", "MOT-2024-120938"), ("Registration number", "MH12DE4321"), ("Date of incident", "05-11-2026"),
                 ("Vehicle", "Maruti Suzuki Swift VXi")],
        "description": "Another vehicle hit my car from behind at a traffic signal. The rear bumper and the boot lid are damaged.",
        "expect_kind": "car",
        "answers": {"policy_number": "MOT-2024-120938", "vehicle_registration": "MH12DE4321", "incident_date": "2026-11-05",
                    "description": "Another vehicle hit my car from behind at a traffic signal. The rear bumper and the boot lid are damaged."},
    },
    "claim_form_bike_brief": {
        "heading": "Motor claim form",
        "rows": [("Policy number", "MOT-2025-031177"), ("Registration number", "DL 8C AF 7788"), ("Date of incident", "09/11/2026")],
        "description": "My scooter slipped on loose gravel and fell. The front mudguard is broken and the footrest is bent.",
        "expect_kind": "bike",
        "answers": {"policy_number": "MOT-2025-031177", "vehicle_registration": "DL8CAF7788", "incident_date": "2026-11-09",
                    "description": "My scooter slipped on loose gravel and fell. The front mudguard is broken and the footrest is bent."},
    },
}


def draw_form(pdf: PdfPages, spec: dict) -> None:
    """One A4 page: a title band, one line per detail, a box for the description."""
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 8.27)
    ax.set_ylim(0, 11.69)
    ax.axis("off")
    ax.add_patch(Rectangle((0.5, 10.25), 7.27, 0.95, color="#1f3a5f"))
    ax.text(0.7, 10.85, spec["heading"], color="white", fontsize=18, fontweight="bold", va="center")
    ax.text(0.7, 10.47, "SAMPLE FORM - made up for testing, not an official document", color="white", fontsize=9, va="center")
    y = 9.8
    step = min(0.56, 6.3 / len(spec["rows"]))           # a long form gets closer lines so that everything fits on the page
    for label, value in spec["rows"]:
        ax.text(0.7, y, label, fontsize=11, color="#444444", va="center")
        ax.text(3.7, y, value, fontsize=12, color="black", va="center")
        ax.plot([0.7, 7.6], [y - step / 2, y - step / 2], color="#cccccc", linewidth=0.8)
        y -= step
    ax.text(0.7, y - 0.1, "Description of the damage", fontsize=11, color="#444444", va="center")
    top = y - 0.4
    ax.add_patch(Rectangle((0.7, top - 1.8), 6.9, 1.8, fill=False, edgecolor="#888888", linewidth=1))
    ax.text(0.88, top - 0.18, "\n".join(textwrap.wrap(spec["description"], 72)), fontsize=11, va="top", linespacing=1.6)
    ax.text(0.7, 0.9, "I confirm that the details given above are correct.", fontsize=9, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def make_all(force: bool = False) -> list:
    """Create the sample forms that do not exist yet (all of them with force=True). Returns the PDF paths."""
    FOLDER.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, spec in SAMPLES.items():
        path = FOLDER / f"{name}.pdf"
        paths.append(path)
        if path.exists() and not force:
            continue
        with PdfPages(path, metadata={"Title": f"{spec['heading']} (sample)", "Author": "sample form for testing"}) as pdf:
            draw_form(pdf, spec)
        (FOLDER / f"{name}.json").write_text(json.dumps({"expect_kind": spec["expect_kind"], "answers": spec["answers"]}, indent=2),
                                             encoding="utf-8")
    return paths


def main() -> int:
    paths = make_all(force=True)
    print(f"Saved {len(paths)} sample claim forms in {FOLDER.relative_to(config.ROOT)}:")
    for path in paths:
        print(f"  {path.name}   ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
