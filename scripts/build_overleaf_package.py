"""Package the thesis for the tutors: the PDF and a zip that compiles on Overleaf.

The zip holds only what LaTeX reads to build memoria/tfg.tex, taken from the
recorder file of the last build (memoria/tfg.fls), plus the bibliography that
biber reads: the sources, the configuration, the logos and every figure and
table. Notes, the README, auxiliary files and the scripts stay out. On Overleaf:
New project, Upload project, and the main document is tfg.tex (pdfLaTeX).

Build the thesis first (latexmk in memoria/), so the recorder file is current:

    uv run python scripts/build_overleaf_package.py --out DIR
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from padel_cv.paths import REPO

THESIS = REPO / "memoria"
MAIN = "tfg"
GENERATED = {".aux", ".bbl", ".lof", ".lot", ".out", ".toc", ".xml", ".bcf", ".log"}
"""Files LaTeX writes and reads back: Overleaf creates its own."""
EXTRA = ["bibliografia.bib"]
"""Read by biber, so absent from the LaTeX recorder file."""


def thesis_inputs(recorder: Path) -> list[str]:
    """The files under memoria/ that the last build read, relative to it."""
    root = THESIS.resolve()
    inputs = set(EXTRA)
    for line in recorder.read_text().splitlines():
        if not line.startswith("INPUT "):
            continue
        path = Path(line.removeprefix("INPUT "))
        path = (path if path.is_absolute() else THESIS / path).resolve()
        if path.is_relative_to(root) and path.suffix not in GENERATED:
            inputs.add(str(path.relative_to(root)))
    return sorted(inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Directory for the PDF and the zip")
    parser.add_argument("--name", default="TFG_Jorge_Sanchez_borrador_completo")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = thesis_inputs(THESIS / f"{MAIN}.fls")
    archive = args.out / f"{args.name}_overleaf.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name in files:
            zipped.write(THESIS / name, name)
    pdf = args.out / f"{args.name}.pdf"
    shutil.copyfile(THESIS / f"{MAIN}.pdf", pdf)
    print(f"{archive} ({len(files)} ficheros)")
    print(pdf)


if __name__ == "__main__":
    main()
