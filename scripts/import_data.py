"""Step 3: copy the course dataset into the project.

Run from the project folder:
    python -m scripts.import_data                           (finds DataSet*.zip in Downloads/Desktop/Documents)
    python -m scripts.import_data "C:\\path\\to\\DataSet-xxxx.zip"
    python -m scripts.import_data "C:\\path\\to\\DataSet"       (an extracted folder works too)

Result:  data/raw/*.csv  (4 big CSV files)   and   data/policy_pdfs/*.pdf  (the Acko policy PDFs).
The zip or folder is only read, never changed.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from acko import config

# Columns every CSV must contain, otherwise the models of this project cannot be trained.
QUOTE_COLUMNS = ["city", "city_risk_score", "vehicle_make", "vehicle_model", "fuel_type", "manufacturing_year",
                 "idv", "ncb_percent", "policy_type", "annual_premium"]
CLAIM_COLUMNS = ["city", "city_risk_score", "vehicle_make", "vehicle_model", "idv", "policy_type",
                 "policy_age_months", "damage_type", "damage_severity_score", "num_parts_affected",
                 "claim_amount", "approval_probability", "claim_approved"]
REQUIRED_COLUMNS = {"car_quote": QUOTE_COLUMNS, "bike_quote": QUOTE_COLUMNS,
                    "car_claims": CLAIM_COLUMNS, "bike_claims": CLAIM_COLUMNS}
LABELS = {"car_quote": "car quotations", "bike_quote": "bike quotations",
          "car_claims": "car claims", "bike_claims": "bike claims"}


def find_zip() -> Path | None:
    """Look for the newest DataSet*.zip in the usual download places."""
    found: list[Path] = []
    for folder in (Path.home() / "Downloads", Path.home() / "Desktop", Path.home() / "Documents", config.ROOT):
        if folder.is_dir():
            found += [p for p in folder.glob("DataSet*.zip") if p.is_file()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def list_files(source: Path) -> dict:
    """Map file name -> function(destination) that copies that file. Works for a zip and for a folder."""
    files: dict = {}
    if source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        for member in archive.infolist():
            if not member.is_dir():
                def copy_member(destination: Path, member=member) -> None:
                    with archive.open(member) as src, open(destination, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                files.setdefault(Path(member.filename).name, copy_member)
    elif source.is_dir():
        for path in source.rglob("*"):
            if path.is_file():
                files.setdefault(path.name, lambda destination, path=path: shutil.copyfile(path, destination))
    return files


def main() -> int:
    source = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else find_zip()
    if source is None or not source.exists():
        print("FAIL  I could not find the DataSet zip.")
        print('      Run again with its path:  python -m scripts.import_data "C:\\path\\to\\DataSet-xxxx.zip"')
        return 1
    print("Reading from:", source)
    files = list_files(source)
    problems = 0

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for key, file_name in config.RAW_FILES.items():
        if file_name not in files:
            print(f"FAIL  {file_name} is not inside {source.name}")
            problems += 1
            continue
        destination = config.RAW_DIR / file_name
        files[file_name](destination)
        columns = list(pd.read_csv(destination, nrows=0).columns)
        missing = [c for c in REQUIRED_COLUMNS[key] if c not in columns]
        rows = len(pd.read_csv(destination, usecols=[0]))
        if missing:
            print(f"FAIL  {file_name}: columns missing: {', '.join(missing)}")
            problems += 1
        else:
            print(f"OK    {LABELS[key]:<16} {rows:>9,} rows   {file_name}")

    # Policy PDFs: skip files called "Copy of ..." (the course zip contains a duplicate FAQ).
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_names = sorted(n for n in files if n.lower().endswith(".pdf") and not n.lower().startswith("copy of"))
    for name in pdf_names:
        destination = config.PDF_DIR / name
        files[name](destination)
        print(f"OK    policy PDF       {len(PdfReader(destination).pages):>4} pages   {name}")
    if len(pdf_names) < 3:
        print(f"FAIL  Only {len(pdf_names)} policy PDF(s) found; the brief needs at least 3 (motor, health, FAQ).")
        problems += 1

    print("Data import finished with", "problems." if problems else "no problems.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
