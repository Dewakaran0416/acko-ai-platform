"""Build the project's Jupyter notebooks from their text source and run them, so each one is saved WITH its output.

    python -m scripts.make_notebooks quote          build and run notebooks/02_quote_model.ipynb
    python -m scripts.make_notebooks car-amount     build and run one claims notebook (car-amount, car-approval, bike-amount, bike-approval)
    python -m scripts.make_notebooks claims         build and run the four claims notebooks, one after the other
    python -m scripts.make_notebooks                build and run every notebook that has a source file

The source of a notebook is a plain text file in notebooks/source/. Cells are separated by lines that start with "# %%";
a cell that starts with "# %% [markdown]" is text (each line begins with "# "), every other cell is code.
Plain text is easy to read and to compare in Git; the .ipynb file that is built from it holds the pictures and tables.

Running a notebook never changes models/. The four claims notebooks also save their SHAP pictures in reports/shap/ (each notebook
says so). If a cell fails, the script prints the end of the error and writes nothing.
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from acko import config

NOTEBOOK_DIR = config.ROOT / "notebooks"
SOURCE_DIR = NOTEBOOK_DIR / "source"
NOTEBOOKS = {"quote": "02_quote_model",         # name on the command line -> file name
             "car-amount": "03_claims_car_amount", "car-approval": "04_claims_car_approval",
             "bike-amount": "05_claims_bike_amount", "bike-approval": "06_claims_bike_approval"}
GROUPS = {"claims": ["car-amount", "car-approval", "bike-amount", "bike-approval"]}       # one name for several notebooks
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def read_cells(source: Path) -> list:
    """Split a text file written in the '# %%' cell format into notebook cells."""
    cells, kind, lines = [], None, []

    def finish() -> None:
        text = "\n".join(lines).strip("\n")
        if kind == "markdown":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.split("\n"))
            cells.append(nbformat.v4.new_markdown_cell(text))
        elif kind == "code" and text.strip():
            cells.append(nbformat.v4.new_code_cell(text))

    for line in source.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# %%"):
            finish()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            lines.append(line)
    finish()
    return cells


def run(name: str) -> int:
    source = SOURCE_DIR / f"{NOTEBOOKS[name]}.py"
    target = NOTEBOOK_DIR / f"{NOTEBOOKS[name]}.ipynb"
    if not source.exists():
        print(f"The source file {source.relative_to(config.ROOT)} is missing.")
        return 1
    notebook = nbformat.v4.new_notebook(cells=read_cells(source))
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    code_cells = sum(1 for cell in notebook.cells if cell.cell_type == "code")
    print(f"{target.name}: {len(notebook.cells)} cells ({code_cells} with code). Running them - this takes a few minutes ...")
    started, finished = time.time(), []

    def progress(cell, cell_index, execute_reply=None, **_):
        if cell.cell_type == "code":
            finished.append(cell_index)
            print(f"  code cell {len(finished)} of {code_cells} done   ({time.time() - started:.0f} s)")

    if sys.platform == "win32":                    # the notebook kernel's network layer needs this event loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # --log-level=ERROR hides the kernel's routine notice about its local network connection (it only talks to this computer).
    client = NotebookClient(notebook, kernel_name="python3", timeout=1800, startup_timeout=180,
                            extra_arguments=["--log-level=ERROR"], on_cell_executed=progress,
                            resources={"metadata": {"path": str(config.ROOT)}})
    try:
        client.execute()
    except CellExecutionError as problem:
        print("\nA cell failed. The end of the error message:\n")
        print("\n".join(ANSI.sub("", str(problem)).strip().split("\n")[-18:]))
        return 1
    nbformat.validate(notebook)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, target)
    pictures = sum(1 for cell in notebook.cells for output in cell.get("outputs", [])
                   if "image/png" in output.get("data", {}))
    print(f"\nSaved {target.relative_to(config.ROOT)}: {target.stat().st_size / 1e6:.1f} MB, {pictures} pictures, "
          f"{code_cells} code cells ran without an error in {(time.time() - started) / 60:.1f} minutes.")
    return 0


def main(argv: list[str]) -> int:
    names = [single for name in argv for single in GROUPS.get(name, [name])] or list(NOTEBOOKS)
    unknown = [name for name in names if name not in NOTEBOOKS]
    if unknown:
        print(f"Unknown notebook: {', '.join(unknown)}. Choose from: {', '.join(list(NOTEBOOKS) + list(GROUPS))}")
        return 2
    return max(run(name) for name in names)            # every notebook is run, even if an earlier one failed


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
