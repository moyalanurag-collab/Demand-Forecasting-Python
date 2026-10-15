"""Author: Anurag Moyal

Convert the percent-format sources in src/notebook_sources/ into executed .ipynb files.

Usage:  python src/build_notebooks.py            # builds + executes every notebook
        python src/build_notebooks.py 01         # only the notebook whose source starts with 01
"""
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "notebook_sources"
OUT = ROOT / "notebooks"


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "md":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif body.strip():
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.split("\n"):
        if re.match(r"^# %%", line):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    for src in sorted(SRC.glob(f"{only}*.py")):
        nb = nbformat.v4.new_notebook(cells=parse(src.read_text(encoding="utf-8")))
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        print(f"Executing {src.name} ...")
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}).execute()
        target = OUT / (src.stem + ".ipynb")
        nbformat.write(nb, target)
        print(f"  -> {target}")


if __name__ == "__main__":
    main()
