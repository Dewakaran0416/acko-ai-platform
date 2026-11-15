"""Step 10b: the labelled photo test for Gemini Vision.

The brief asks that at least 8 of 10 test photos are classified correctly, with photos that differ in damage and vehicle.
This test makes that number honest: YOU say what each photo shows first, and only then is Gemini asked.

    python -m scripts.eval_vision --template   makes uploads/test_photos/labels.csv: one row per photo, labels still empty
        (now open labels.csv, look at each photo and fill in its row. Do NOT ask Gemini before the labels are saved.)
    python -m scripts.eval_vision --check      checks labels.csv without calling Gemini
    python -m scripts.eval_vision              asks Gemini about every labelled photo and compares its answers with your labels

Columns of labels.csv:
    file         the photo's file name
    vehicle      car or bike
    damage_type  one of the 13 damage types the claim models know
    severity     minor, moderate or major
    also_ok      (optional) other damage types that would ALSO be right for this photo, separated by |
    parts        (optional) the damaged parts you can see, separated by |, written as on the parts list
    source       (optional) where the photo came from; it goes into the report

Gemini is NOT told whether a photo shows a car or a bike, and the prompt is the one the claim page uses.
Results: reports/vision_eval.md (the report), reports/vision_eval_results.csv (one row per photo) and
reports/vision_eval_history.csv (one row per run: every run is kept, so the best run cannot be picked quietly).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from acko import claims, config, vision

FOLDER = config.ROOT / "uploads" / "test_photos"
LABELS = FOLDER / "labels.csv"
REPORT = config.ROOT / "reports" / "vision_eval.md"
RESULTS = config.ROOT / "reports" / "vision_eval_results.csv"
HISTORY = config.ROOT / "reports" / "vision_eval_history.csv"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
BANDS = ("minor", "moderate", "major")
COLUMNS = ["file", "vehicle", "damage_type", "severity", "also_ok", "parts", "source"]
HISTORY_COLUMNS = ["run_time", "model", "prompt_fingerprint", "photos", "answered", "vehicle_right", "type_strict", "type_right",
                   "severity_exact", "severity_near", "type_and_severity", "labels_saved"]
TARGET = 0.8                      # the brief: 8 of 10 photos
PAUSE = 4.0                       # seconds between two photos, to stay inside the free quota
RETRY_WAITS = (30, 60)            # a photo that meets "quota" or "busy" is tried again after these pauses (seconds)
STAMP = "%Y-%m-%d %H:%M:%S"


# ------------------------------------------------------------------------------------------------ small helpers
def where(path: Path) -> str:
    """A path as short as possible: relative to the project folder when it is inside it."""
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


def natural(path: Path):
    """Sort key: 2.jpg comes before 10.jpg."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_photos() -> list[Path]:
    if not FOLDER.exists():
        return []
    return sorted((p for p in FOLDER.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES), key=natural)


def split_list(text: str | None) -> list[str]:
    return [part.strip() for part in re.split(r"[|/;]", text or "") if part.strip()]


def cell(text) -> str:
    """Text that is safe inside one cell of a Markdown table."""
    return re.sub(r"\s+", " ", str(text or "")).replace("|", "/").strip()


def pct(count: int, total: int) -> str:
    return f"{100 * count / total:.0f} %" if total else "-"


def fingerprint() -> str:
    """A short code for the exact prompt that is sent with every photo: if the prompt changes, the code changes."""
    return hashlib.sha256(vision.photo_prompt().encode("utf-8")).hexdigest()[:8]


def wilson(right: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 % interval for a share measured on a small number of photos (Wilson score interval)."""
    if total == 0:
        return 0.0, 0.0
    p = right / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


# ------------------------------------------------------------------------------------------------ labels.csv
@dataclass
class Label:
    file: str
    vehicle: str
    damage_type: str
    band: str
    also_ok: list
    parts: list
    source: str


def read_rows() -> tuple[list[dict], list[str]]:
    """labels.csv as a list of dictionaries (column names in lower case, values trimmed) and a list of problems."""
    try:
        raw = LABELS.read_bytes()
    except OSError as exc:
        return [], [f"labels.csv cannot be opened: {exc}"]
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")           # what Excel saves as 'CSV (Comma delimited)' on Windows
    first = text.splitlines()[0] if text.strip() else ""
    delimiter = ";" if first.count(";") > first.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    names = [(name or "").strip().lower() for name in (reader.fieldnames or [])]
    if "file" not in names:
        return [], ["labels.csv has no 'file' column in its first line. Run: python -m scripts.eval_vision --template"]
    reader.fieldnames = names
    rows = []
    for raw_row in reader:
        row = {key: (value or "").strip() for key, value in raw_row.items() if key and not isinstance(value, list)}
        row["_line"] = str(reader.line_num)
        rows.append(row)
    return rows, []


def write_template() -> int:
    """Create labels.csv with one empty row per photo; rows that are already filled in are never touched."""
    photos = list_photos()
    if not photos:
        print(f"There is no photo in {where(FOLDER)} yet.")
        print("Copy your test photos (JPG, PNG or WEBP) into that folder and run this again.")
        return 1
    rows: list[dict] = []
    if LABELS.exists():
        rows, problems = read_rows()
        if problems:
            print(problems[0])
            return 1
    known = {row.get("file", "").lower() for row in rows}
    new = [photo.name for photo in photos if photo.name.lower() not in known]
    if new or not LABELS.exists():
        rows.extend({"file": name} for name in new)
        try:
            with LABELS.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore", restval="")
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            print(f"labels.csv could not be written ({exc}). If it is open in Excel, close it and run this again.")
            return 1
    done = sum(1 for row in rows if row.get("vehicle") and row.get("damage_type") and row.get("severity"))
    print(f"{where(LABELS)} has a row for each of the {len(photos)} photos ({len(new)} new, {done} already labelled).")
    print()
    print("Now open it in Excel or Notepad, LOOK AT EACH PHOTO and fill in these three columns:")
    print("  vehicle      car or bike")
    print("  damage_type  one of the damage types below")
    print("  severity     minor, moderate or major")
    print("Optional columns: also_ok (other damage types that would ALSO be right for the photo, separated by |),")
    print("                  parts (the damaged parts you can see, separated by |), source (where the photo came from).")
    print()
    print("Damage types:")
    for name, _usual, meaning in vision.DAMAGE_GUIDE:
        print(f"  {name:<18} {meaning}")
    print()
    print("Severity: minor = only cosmetic, the vehicle is fully usable; moderate = a repair or a part replacement is needed;")
    print("          major = heavy or structural damage, the vehicle is unsafe or cannot be repaired.")
    print()
    print("Label first, test second: save the labels BEFORE the test, and do not change a label after you have seen Gemini's answer.")
    print("Check the file with:  python -m scripts.eval_vision --check")
    return 0


def load_labels(photos: list[Path]) -> tuple[list[Label], list[str], list[str]]:
    """(the labels that are complete, problems that stop the test, notes). Photos without a label are left out with a note."""
    rows, problems = read_rows()
    notes: list[str] = []
    by_name = {photo.name.lower(): photo for photo in photos}
    labels: list[Label] = []
    used: set[str] = set()
    empty: list[str] = []
    for row in rows:
        name = row.get("file", "")
        if not name:
            continue
        place = f"labels.csv line {row.get('_line', '?')} ({name})"
        photo = by_name.get(name.lower())
        if photo is None:
            problems.append(f"{place}: there is no photo with this name in {where(FOLDER)}.")
            continue
        if photo.name in used:
            problems.append(f"{place}: this photo has two rows.")
            continue
        used.add(photo.name)
        if not (row.get("vehicle") or row.get("damage_type") or row.get("severity")):
            empty.append(photo.name)
            continue
        vehicle = vision._pick(row.get("vehicle"), claims.KINDS)
        damage = vision._pick(row.get("damage_type"), claims.DAMAGE_TYPES)
        band = vision._pick(row.get("severity"), BANDS)
        bad = []
        if vehicle is None:
            bad.append("vehicle must be car or bike")
        if damage is None:
            bad.append("damage_type must be one of the damage types")
        if band is None:
            bad.append("severity must be minor, moderate or major")
        also, odd = [], []
        for item in split_list(row.get("also_ok")):
            found = vision._pick(item, claims.DAMAGE_TYPES)
            if found is None:
                odd.append(item)
            elif found != damage and found not in also:
                also.append(found)
        if odd:
            bad.append("also_ok has words that are not damage types: " + ", ".join(odd))
        parts, odd = [], []
        if vehicle:
            for item in split_list(row.get("parts")):
                found = vision._pick(item, vision.PARTS[vehicle])
                if found is None:
                    odd.append(item)
                elif found not in parts:
                    parts.append(found)
            if odd:
                bad.append(f"parts has words that are not on the parts list of a {vehicle}: " + ", ".join(odd))
        if bad:
            problems.append(f"{place}: " + "; ".join(bad) + ".")
            continue
        labels.append(Label(photo.name, vehicle, damage, band, also, parts, row.get("source", "")[:120]))
    if empty:
        notes.append("Not tested, no label yet: " + ", ".join(empty) + ".")
    unlisted = [photo.name for photo in photos if photo.name not in used]
    if unlisted:
        notes.append("Not tested, no row in labels.csv (run --template to add rows): " + ", ".join(unlisted) + ".")
    order = {photo.name: number for number, photo in enumerate(photos)}
    labels.sort(key=lambda label: order[label.file])
    if labels and not any(label.source for label in labels):
        notes.append("No photo has a 'source' yet: write where the photos came from (own photos, a website, ...) before you hand in.")
    return labels, problems, notes


def check_files(labels: list[Label]) -> tuple[list[str], list[str]]:
    """Open every labelled photo the way the app does (so a broken file is found before any quota is spent) and look for copies."""
    problems, notes = [], []
    first_seen: dict[str, str] = {}
    for label in labels:
        try:
            data = (FOLDER / label.file).read_bytes()
            vision.prepare_photo(data)
        except vision.VisionError as problem:
            problems.append(f"{label.file}: {problem}")
            continue
        except OSError as exc:
            problems.append(f"{label.file}: the file cannot be read ({exc}).")
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest in first_seen:
            notes.append(f"{label.file} is the same picture as {first_seen[digest]}: use different photos.")
        else:
            first_seen[digest] = label.file
    return problems, notes


def mix_notes(labels: list[Label]) -> list[str]:
    """Is the set of photos what the brief asks for: 10 photos, different vehicles, different severity?"""
    notes = []
    total = len(labels)
    if total < 10:
        notes.append(f"The brief asks for 10 test photos; {total} are labelled.")
    cars = sum(1 for label in labels if label.vehicle == "car")
    for kind, count in (("car", cars), ("bike", total - cars)):
        if count < 3:
            notes.append(f"Only {count} {kind} photo{'s' if count != 1 else ''}: the brief wants different vehicle types, so 3 or more of each is better.")
    bands = Counter(label.band for label in labels)
    for band in BANDS:
        if not bands[band]:
            notes.append(f"No '{band}' photo: the photos should differ in severity.")
    return notes


def describe_mix(labels: list[Label]) -> str:
    cars = sum(1 for label in labels if label.vehicle == "car")
    bands = Counter(label.band for label in labels)
    return (f"{cars} cars, {len(labels) - cars} bikes; minor {bands['minor']}, moderate {bands['moderate']}, "
            f"major {bands['major']}")


# ------------------------------------------------------------------------------------------------ asking Gemini and scoring
@dataclass
class Result:
    label: Label
    reading: vision.PhotoReading | None = None
    problem: str = ""
    seconds: float = 0.0

    @property
    def vehicle_ok(self) -> bool:
        return self.reading is not None and self.reading.vehicle == self.label.vehicle

    @property
    def type_exact(self) -> bool:
        return self.reading is not None and self.reading.damage_type == self.label.damage_type

    @property
    def type_ok(self) -> bool:
        """The damage type is the label or one of the alternates the labeller wrote down before the run."""
        return self.reading is not None and self.reading.damage_type in (self.label.damage_type, *self.label.also_ok)

    @property
    def band_exact(self) -> bool:
        return self.reading is not None and self.reading.severity == self.label.band

    @property
    def band_near(self) -> bool:
        if self.reading is None or self.reading.severity not in BANDS:
            return False
        return abs(BANDS.index(self.reading.severity) - BANDS.index(self.label.band)) <= 1

    @property
    def full_ok(self) -> bool:
        return self.type_ok and self.band_exact


def ask(path: Path) -> tuple[vision.PhotoReading | None, str, float]:
    """Gemini's reading of one photo (or the reason there is none) and the seconds it took.
    A busy or over-quota Gemini is asked again after a pause; a wrong key or model name stops the whole test."""
    data = path.read_bytes()
    for wait in (*RETRY_WAITS, None):
        begin = time.perf_counter()
        try:
            reading = vision.analyze_photo(data)
            return reading, "", time.perf_counter() - begin
        except vision.VisionError as problem:
            if problem.kind in ("key", "model"):
                raise
            if problem.kind in ("quota", "busy") and wait is not None:
                print(f"          Gemini: {problem} Waiting {wait} s, then this photo is tried again ...")
                time.sleep(wait)
                continue
            return None, str(problem), time.perf_counter() - begin
    return None, "no answer", 0.0


def describe(label: Label) -> str:
    return f"{label.vehicle}, {label.damage_type}, {label.band}"


def gemini_said(reading: vision.PhotoReading | None) -> str:
    if reading is None:
        return "-"
    score = f"{reading.severity_score} ({reading.severity})" if reading.severity_score else "no severity"
    return f"{reading.vehicle}, {reading.damage_type or 'no damage found'}, {score}"


def verdict(result: Result) -> str:
    if result.reading is None:
        return f"NO ANSWER ({result.problem})"
    marks = ["vehicle right" if result.vehicle_ok else "VEHICLE WRONG",
             "type right" if result.type_exact else "type right (your alternate)" if result.type_ok else "TYPE WRONG",
             "severity right" if result.band_exact else "severity one band off" if result.band_near else "SEVERITY WRONG"]
    return ", ".join(marks)


def run_all(labels: list[Label]) -> list[Result]:
    results = []
    for number, label in enumerate(labels, start=1):
        if number > 1:
            time.sleep(PAUSE)
        reading, problem, seconds = ask(FOLDER / label.file)
        result = Result(label, reading, problem, seconds)
        results.append(result)
        prefix = f"  [{number:>2}/{len(labels)}] {label.file:<14} "
        print(f"{prefix}you   : {describe(label)}")
        print(f"{' ' * len(prefix)}Gemini: {gemini_said(reading)}   =>  {verdict(result)}")
    return results


def tally(results: list[Result]) -> dict:
    return {"photos": len(results), "answered": sum(1 for r in results if r.reading is not None),
            "vehicle_right": sum(r.vehicle_ok for r in results), "type_strict": sum(r.type_exact for r in results),
            "type_right": sum(r.type_ok for r in results), "severity_exact": sum(r.band_exact for r in results),
            "severity_near": sum(r.band_near for r in results), "type_and_severity": sum(r.full_ok for r in results)}


def parts_score(results: list[Result]) -> tuple[int, int, int, int]:
    """(listed parts Gemini also named, parts listed, parts Gemini named, photos where parts were listed)."""
    found = listed = named = photos = 0
    for result in results:
        if not result.label.parts:
            continue
        photos += 1
        said = set(result.reading.parts) if result.reading is not None else set()
        found += len(said & set(result.label.parts))
        listed += len(result.label.parts)
        named += len(said)
    return found, listed, named, photos


# ------------------------------------------------------------------------------------------------ report, results, history
def read_history() -> list[dict]:
    if not HISTORY.exists():
        return []
    with HISTORY.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def labels_note(history: list[dict], saved: datetime) -> str:
    """A warning when labels.csv was saved after the previous run (labels may have been changed after seeing results)."""
    if not history:
        return ""
    try:
        before = datetime.strptime(history[-1]["run_time"], STAMP)
    except (KeyError, ValueError):
        return ""
    if saved > before:
        return (f"labels.csv was saved at {saved:%Y-%m-%d %H:%M}, after the previous run ({before:%Y-%m-%d %H:%M}): "
                "the earlier runs in the history were made with the older labels.")
    return ""


def add_history(row: dict) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    new = not HISTORY.exists()
    with HISTORY.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def yes(flag: bool) -> str:
    return "yes" if flag else "no"


def write_results_csv(results: list[Result]) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "source", "vehicle_label", "vehicle_gemini", "damage_type_label", "also_ok", "damage_type_gemini",
                         "severity_label", "severity_score_gemini", "severity_gemini", "parts_label", "parts_gemini", "confidence",
                         "vehicle_right", "type_strict", "type_right", "severity_exact", "severity_near", "gemini_says", "problem"])
        for r in results:
            g = r.reading
            writer.writerow([r.label.file, r.label.source, r.label.vehicle, g.vehicle if g else "", r.label.damage_type,
                             "|".join(r.label.also_ok), (g.damage_type or "") if g else "", r.label.band,
                             (g.severity_score or "") if g else "", (g.severity or "") if g else "", "|".join(r.label.parts),
                             "|".join(g.parts) if g else "", g.confidence if g else "", yes(r.vehicle_ok), yes(r.type_exact),
                             yes(r.type_ok), yes(r.band_exact), yes(r.band_near), cell(g.summary) if g else "", r.problem])


def write_report(results: list[Result], counts: dict, notes: list[str], run_number: int, started: datetime, saved: datetime) -> None:
    n = counts["photos"]
    low, high = wilson(counts["type_right"], n)
    met = n > 0 and counts["type_right"] / n >= TARGET
    times = sorted(r.seconds for r in results if r.reading is not None)
    found, listed, named, with_parts = parts_score(results)
    labels = [r.label for r in results]
    lines = [
        "# Gemini Vision photo test (Module 3)", "",
        f"Run {run_number}, started {started:{STAMP}}. Model `{config.GEMINI_VISION_MODEL}`, prompt fingerprint `{fingerprint()}`.",
        "All runs are listed in `reports/vision_eval_history.csv`; one row per photo is in `reports/vision_eval_results.csv`.", "",
        "## How the test was done", "",
        f"- {n} photos ({describe_mix(labels)}). They are ordinary photos, not rows of the claims data set.",
        "- The labeller looked at each photo and wrote down the vehicle, the damage type and the severity band BEFORE Gemini was asked. "
        f"`labels.csv` was last saved at {saved:{STAMP}}; this run started at {started:{STAMP}}.",
        "- Each photo was sent to Gemini once, with the same prompt as the claim page. Gemini was not told whether the photo shows a car or a bike.",
        "- Damage types overlap (a hit front bumper can be bumper damage, collision damage or panel damage), so the labeller could write "
        "down other acceptable types for a photo before the run (column `also_ok`). The strict row counts only the single label.",
        "- Severity band: Gemini gives a score from 1 to 10 and the app turns it into minor (1-3), moderate (4-6) or major (7-10).", "",
        "## Result", "",
        "| Check | Photos | Share |", "|---|---|---|",
        f"| Car or bike right | {counts['vehicle_right']} of {n} | {pct(counts['vehicle_right'], n)} |",
        f"| Damage type right, the single label only | {counts['type_strict']} of {n} | {pct(counts['type_strict'], n)} |",
        f"| **Damage type right (the label or the labeller's alternates)** | **{counts['type_right']} of {n}** | **{pct(counts['type_right'], n)}** |",
        f"| Severity band right | {counts['severity_exact']} of {n} | {pct(counts['severity_exact'], n)} |",
        f"| Severity band right or one band off | {counts['severity_near']} of {n} | {pct(counts['severity_near'], n)} |",
        f"| Damage type (with alternates) and severity band both right | {counts['type_and_severity']} of {n} | {pct(counts['type_and_severity'], n)} |",
        "",
        f"Brief target: at least 8 of 10 photos with the right damage type. Result: **{'MET' if met else 'NOT MET'}** "
        f"({counts['type_right']} of {n}).",
        f"With only {n} photos the real success rate could lie anywhere between about {100 * low:.0f} % and {100 * high:.0f} % "
        "(95 % interval), so this is an indication, not a guarantee.", "",
    ]
    if with_parts:
        lines.append(f"Damaged parts (the {with_parts} photos where the labeller listed parts): Gemini named {pct(found, listed)} of the listed "
                     f"parts, and {pct(found, named)} of the parts it named were on the list.")
    else:
        lines.append("Damaged parts were not labelled, so they were not scored.")
    if times:
        lines.append(f"Gemini answered in {times[len(times) // 2]:.1f} s per photo (median), the slowest took {times[-1]:.1f} s.")
    lines += ["", "## Photo by photo", "",
              "| # | Photo | Source | Label | Gemini | Type | Severity | Gemini says |", "|---|---|---|---|---|---|---|---|"]
    for number, r in enumerate(results, start=1):
        kind = "no answer" if r.reading is None else "right" if r.type_exact else "right (alternate)" if r.type_ok else "wrong"
        band = "no answer" if r.reading is None else "right" if r.band_exact else "1 band off" if r.band_near else "wrong"
        said = cell(r.reading.summary) if r.reading is not None else cell(r.problem)
        lines.append(f"| {number} | {cell(r.label.file)} | {cell(r.label.source)} | {cell(describe(r.label))} | "
                     f"{cell(gemini_said(r.reading))} | {kind} | {band} | {said} |")
    if notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    lines += ["", "## Limits of this test", "",
              "- One person labelled the photos, and the damage types overlap, so some labels are a matter of opinion.",
              "- A small number of photos, chosen by hand and not at random (see the interval above).",
              "- Gemini's answers can change a little from run to run; every run is kept in the history file.",
              "- If the prompt in `acko/vision.py` changes, the prompt fingerprint changes: a change between two runs is visible in the history.",
              ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def print_summary(results: list[Result], counts: dict, run_number: int, started: datetime, saved: datetime) -> None:
    n = counts["photos"]
    low, high = wilson(counts["type_right"], n)
    met = n > 0 and counts["type_right"] / n >= TARGET
    found, listed, named, with_parts = parts_score(results)
    times = sorted(r.seconds for r in results if r.reading is not None)
    print()
    print(f"Photos tested: {n} ({describe_mix([r.label for r in results])})")
    print(f"  car or bike right ............................... {counts['vehicle_right']:>2} of {n}")
    print(f"  damage type right (your label only) ............. {counts['type_strict']:>2} of {n}")
    print(f"  damage type right (label or your alternates) .... {counts['type_right']:>2} of {n}   <- the brief's 8 of 10")
    print(f"  severity band right ............................. {counts['severity_exact']:>2} of {n}   "
          f"(right or one band off: {counts['severity_near']} of {n})")
    print(f"  damage type and severity band both right ........ {counts['type_and_severity']:>2} of {n}")
    if with_parts:
        print(f"  damaged parts ({with_parts} photos with parts listed) ..... Gemini named {pct(found, listed)} of your parts; "
              f"{pct(found, named)} of its parts were on your list")
    print(f"Brief target (8 of 10 photos with the right damage type): {'MET' if met else 'NOT MET'}   "
          f"(with {n} photos the real rate could be anywhere from {100 * low:.0f} % to {100 * high:.0f} %)")
    if times:
        print(f"Gemini took {times[len(times) // 2]:.1f} s per photo (median), the slowest {times[-1]:.1f} s.")
    print(f"labels.csv was saved {saved:{STAMP}}; this run started {started:{STAMP}} (run {run_number}; every run is kept in {where(HISTORY)}).")
    print(f"Saved: {where(REPORT)} and {where(RESULTS)}")


# ------------------------------------------------------------------------------------------------ main
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Labelled photo test for Gemini Vision.")
    parser.add_argument("--template", action="store_true", help="write labels.csv with one empty row per photo")
    parser.add_argument("--check", action="store_true", help="check labels.csv without calling Gemini")
    args = parser.parse_args(argv)
    if args.template:
        return write_template()
    photos = list_photos()
    if not photos:
        print(f"There is no photo in {where(FOLDER)}. Copy your test photos (JPG, PNG or WEBP) into that folder first.")
        return 1
    if not LABELS.exists():
        print(f"There is no {where(LABELS)} yet. Run  python -m scripts.eval_vision --template  and fill the file in.")
        return 1
    labels, problems, notes = load_labels(photos)
    file_problems, file_notes = check_files(labels)
    problems += file_problems
    notes += file_notes + mix_notes(labels)
    if problems:
        print(f"{where(LABELS)} cannot be used yet:")
        for problem in problems:
            print(f"  - {problem}")
        print("Damage types : " + ", ".join(claims.DAMAGE_TYPES))
        print("Severity     : minor, moderate, major")
        return 1
    if not labels:
        print(f"No photo is labelled yet. Open {where(LABELS)}, look at each photo and fill in vehicle, damage_type and severity.")
        return 1
    print(f"Labels OK: {len(labels)} photos ({describe_mix(labels)}).")
    for note in notes:
        print(f"  note: {note}")
    if args.check:
        print("Next: python -m scripts.eval_vision")
        return 0
    if not config.GEMINI_API_KEY:
        print("GEMINI_API_KEY is not set in this window, so Gemini cannot be called. (It worked for the chatbot, so open the same window you used for that.)")
        return 1
    saved = datetime.fromtimestamp(LABELS.stat().st_mtime).replace(microsecond=0)      # the version of labels.csv this run uses
    started = datetime.now().replace(microsecond=0)
    print(f"\nAsking Gemini ({config.GEMINI_VISION_MODEL}) about {len(labels)} photos, one at a time. "
          "Gemini is not told whether a photo shows a car or a bike.\n")
    try:
        results = run_all(labels)
    except vision.VisionError as problem:
        print(f"\nStopped: {problem}")
        print("Nothing was saved. Fix this and run the test again.")
        return 1
    counts = tally(results)
    history = read_history()
    warnings = []
    changed = labels_note(history, saved)
    if changed:
        warnings.append(changed)
    if datetime.fromtimestamp(LABELS.stat().st_mtime).replace(microsecond=0) != saved:
        warnings.append("labels.csv was saved again while the test was running; these results used the labels as they were when the run began.")
    silent = counts["photos"] - counts["answered"]
    if silent:
        warnings.append(f"{silent} photo(s) got no answer from Gemini and count as wrong. If it was the quota, wait a few minutes and run "
                        "the test again; both runs stay in the history.")
    notes += warnings
    run_number = len(history) + 1
    write_results_csv(results)
    write_report(results, counts, notes, run_number, started, saved)
    add_history({"run_time": f"{started:{STAMP}}", "model": config.GEMINI_VISION_MODEL, "prompt_fingerprint": fingerprint(),
                 "photos": counts["photos"], "answered": counts["answered"], "vehicle_right": counts["vehicle_right"],
                 "type_strict": counts["type_strict"], "type_right": counts["type_right"],
                 "severity_exact": counts["severity_exact"], "severity_near": counts["severity_near"],
                 "type_and_severity": counts["type_and_severity"], "labels_saved": f"{saved:{STAMP}}"})
    print_summary(results, counts, run_number, started, saved)
    for warning in warnings:
        print(f"note: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
