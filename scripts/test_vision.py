"""Step 10: try the Gemini vision code on your own computer.

    python -m scripts.test_vision                     reads the four sample claim forms and the first photo in uploads/test_photos
    python -m scripts.test_vision --photo some.jpg    describes this photo
    python -m scripts.test_vision --form some.pdf     reads this claim form
    python -m scripts.test_vision --details           also shows every detail read from each form

Each sample form is read by Gemini and compared with the answers printed on it. A photo is only described: there is no
'right answer' to compare with yet (the labelled test with ten photos comes in the next step).
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import sys
import time
from datetime import date
from pathlib import Path

from PIL import Image

from acko import config, vision
from scripts import make_sample_forms

PHOTO_FOLDER = config.ROOT / "uploads" / "test_photos"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
PAUSE = 1.5                                    # seconds between two Gemini calls, to stay well inside the free quota


def where(path: Path) -> str:
    """A path as short as possible: relative to the project folder when it is inside it."""
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


def flat(text) -> str:
    return " ".join(str(text).lower().split())


def shown(value):
    return value.isoformat() if isinstance(value, date) else value


def same(key: str, truth, got) -> bool:
    """Is what Gemini read the same as what is printed on the form?"""
    if got is None:
        return False
    if key in ("incident_date", "policy_start_date"):
        return shown(got) == truth
    if key == "idv":
        return abs(float(got) - float(truth)) < 1
    if key in ("previous_claims", "zero_dep", "ncb", "helmet_worn", "rider_experience"):
        return int(got) == int(truth)
    if key == "description":
        return difflib.SequenceMatcher(None, flat(got), flat(truth)).ratio() >= 0.9
    return flat(got) == flat(truth)


def check_form(path: Path, details: bool) -> tuple[bool, bool] | None:
    """Read one form; returns (every detail right, routing right) when the answers are known, else (True, True)."""
    key = f"{path.stem:<24}"
    try:
        reading = vision.read_form(path.read_bytes())
    except vision.VisionError as problem:
        print(f"  {key} FAILED: {problem}")
        return None
    routing = vision.route_vehicle(reading)
    truth_file = path.with_suffix(".json")
    how = f"routed: {routing.kind or 'unknown'} ({routing.source})"
    if truth_file.exists():
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        wrong = [f"{name}: printed {value!r}, Gemini read {shown(reading.values.get(name))!r}"
                 for name, value in truth["answers"].items() if not same(name, value, reading.values.get(name))]
        total = len(truth["answers"])
        right_route = routing.kind == truth["expect_kind"]
        print(f"  {key} {total - len(wrong):>2}/{total} details right   {how}   expected {truth['expect_kind']}: "
              f"{'OK' if right_route else 'WRONG'}")
        for line in wrong:
            print(f"      differs  {line}")
        result = (not wrong, right_route)
    else:
        print(f"  {key} {len(reading.values)} details read   {how}   missing: {', '.join(reading.missing) or 'nothing'}")
        result = (True, True)
    if details:
        for name, value in reading.values.items():
            print(f"      {name:<22} {shown(value)}")
        for note in reading.notes:
            print(f"      note: {note}")
        if reading.source != "pdf":
            print(f"      (read from the text inside the PDF, not from the PDF itself)")
    return result


def show_photo(path: Path) -> bool:
    try:
        jpeg = vision.prepare_photo(path.read_bytes())
        reading = vision.analyze_photo(jpeg, prepared=True)
    except vision.VisionError as problem:
        print(f"  {path.name}: FAILED: {problem}")
        return False
    width, height = Image.open(io.BytesIO(jpeg)).size
    print(f"  Photo {where(path)}   (sent to Gemini as {width} x {height}, {len(jpeg) / 1024:.0f} KB)")
    print(f"    vehicle in the photo : {reading.vehicle}")
    if reading.usable:
        print(f"    damage               : {reading.damage_type}, severity {reading.severity_score} of 10 ({reading.severity}), "
              f"confidence {reading.confidence}")
    else:
        print(f"    damage               : none found that the app can use (damage visible: {reading.damage_visible})")
    print(f"    damaged parts        : {', '.join(reading.parts) or 'none named'}")
    print(f"    Gemini says          : {reading.summary}")
    for note in reading.notes:
        print(f"    note                 : {note}")
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Try the Gemini vision code.")
    parser.add_argument("--photo", type=Path, help="describe this photo")
    parser.add_argument("--form", type=Path, help="read this claim form (PDF)")
    parser.add_argument("--details", action="store_true", help="show every detail read from each form")
    args = parser.parse_args(argv)
    if not config.GEMINI_API_KEY:
        print("GEMINI_API_KEY is not set in this window, so Gemini cannot be called. (It worked for the chatbot, so open the same window you used for that.)")
        return 1
    print(f"Gemini model for photos and forms: {config.GEMINI_VISION_MODEL}\n")

    results = []
    if args.form or not args.photo:
        forms = [args.form] if args.form else make_sample_forms.make_all()
        print(f"Claim forms ({len(forms)}):")
        for number, form in enumerate(forms):
            if number:
                time.sleep(PAUSE)
            results.append(check_form(form, args.details))
        print()
    photo_ok = None
    if args.photo or not args.form:
        photo = args.photo
        if photo is None:
            found = sorted(p for p in PHOTO_FOLDER.glob("*") if p.suffix.lower() in IMAGE_TYPES) if PHOTO_FOLDER.exists() else []
            photo = found[0] if found else None
        if photo is None:
            print(f"No photo yet in {where(PHOTO_FOLDER)}: add one and run this again. (The form test above does not need it.)")
        else:
            print("Damage photo:")
            if results:
                time.sleep(PAUSE)
            photo_ok = show_photo(photo)
    answered = [r for r in results if r is not None]
    if results:
        print(f"Forms read: {len(answered)} of {len(results)}   every detail right: {sum(1 for r in answered if r[0])}   "
              f"car or bike right: {sum(1 for r in answered if r[1])}")
    failed = (results and not answered) or photo_ok is False
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
