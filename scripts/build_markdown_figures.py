"""Write the two example figures of section 1.3: memoria/figuras/ejemplo-*.pdf.

Each figure is a short Markdown document (an ADR and an entry of the experiment
log, both summarised for the figure) shown as Markdown looks once rendered: the
light style of GitHub, inside a box. The sources live in memoria/figuras/fuentes/;
each one is converted to HTML and printed to a vector PDF by a headless browser,
at the width of the text, so it goes into the thesis at its natural size.

Neither the Markdown parser nor Playwright is a dependency of the project, so
both are added for the run:

    uv run --with markdown-it-py --with playwright==1.61.0 \\
        python scripts/build_markdown_figures.py

On a WSL without the browser's system libraries, point LD_LIBRARY_PATH at a copy
of them (libnss3, libnspr4, libasound2) extracted in user space.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright

from padel_cv.paths import REPO

SOURCES = REPO / "memoria" / "figuras" / "fuentes"
OUT = REPO / "memoria" / "figuras"
FIGURES = {"ejemplo-adr.md": "ejemplo-adr.pdf", "ejemplo-bitacora.md": "ejemplo-bitacora.pdf"}

TEXT_WIDTH = "160mm"
"""Width of the text block (A4 minus the 2.5 cm margins): the figure is printed at
its final size, so it needs no scaling in LaTeX."""
FONT_SIZE = "8pt"
"""Smaller than the text of the thesis: the figures are examples, not content to read
closely. Segoe UI looks larger than Latin Modern at the same size, so 8 pt still
reads well. Every spacing is in rem, so it shrinks along with the letter."""
FONTS = Path("/mnt/c/Windows/Fonts")
"""Segoe UI and Consolas, the fonts GitHub uses on Windows. Without them the
browser falls back to DejaVu Sans and DejaVu Sans Mono."""

CSS = """
@font-face { font-family: "Segoe UI"; src: url("{fonts}/segoeui.ttf"); }
@font-face { font-family: "Segoe UI"; src: url("{fonts}/segoeuii.ttf"); font-style: italic; }
@font-face { font-family: "Segoe UI"; src: url("{fonts}/seguisb.ttf"); font-weight: 600; }
@font-face { font-family: "Segoe UI"; src: url("{fonts}/seguisbi.ttf");
  font-weight: 600; font-style: italic; }
@font-face { font-family: "Consolas"; src: url("{fonts}/consola.ttf"); }
@page { margin: 0; }
html { font-size: {size}; }
html, body { margin: 0; padding: 0; background: #ffffff; }
.box {
  width: {width}; box-sizing: border-box; padding: 1.5rem 1.9rem;
  border: 1px solid #d1d9e0; border-radius: 6px;
  font-family: "Segoe UI", "DejaVu Sans", sans-serif; font-size: 1rem;
  line-height: 1.5; color: #1f2328;
}
.box > :first-child { margin-top: 0; }
.box > :last-child { margin-bottom: 0; }
h1, h2 { font-weight: 600; line-height: 1.25; padding-bottom: 0.3em;
  border-bottom: 1px solid #d1d9e0; margin: 1.7rem 0 1rem; }
h1 { font-size: 1.5em; }
h2 { font-size: 1.25em; }
p, ul, table { margin: 0 0 1.05rem; }
ul { padding-left: 2em; }
li + li { margin-top: 0.25em; }
strong { font-weight: 600; }
code { font-family: "Consolas", "DejaVu Sans Mono", monospace; font-size: 88%;
  padding: 0.15em 0.4em; border-radius: 6px; background: rgba(129, 139, 152, 0.12); }
table { border-collapse: collapse; }
th, td { border: 1px solid #d1d9e0; padding: 0.4rem 1rem; }
th { font-weight: 600; }
tbody tr:nth-child(2n) { background: #f6f8fa; }
"""


def to_html(markdown: str) -> str:
    """The Markdown as a standalone page: GitHub's light style, inside a box."""
    body = MarkdownIt("commonmark").enable("table").render(markdown)
    css = (
        CSS.replace("{fonts}", FONTS.as_uri())
        .replace("{width}", TEXT_WIDTH)
        .replace("{size}", FONT_SIZE)
    )
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        f'<style>{css}</style></head><body><div class="box">{body}</div></body></html>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.emulate_media(media="print")
        for source, target in FIGURES.items():
            html = Path(tmp) / f"{Path(source).stem}.html"
            html.write_text(to_html((SOURCES / source).read_text()))
            page.goto(html.as_uri())
            page.evaluate("document.fonts.ready")
            # The page is exactly as tall as the box, so the PDF has no margin to crop.
            height = page.evaluate("document.querySelector('.box').getBoundingClientRect().height")
            path = args.out / target
            page.pdf(
                path=str(path),
                width=TEXT_WIDTH,
                height=f"{math.ceil(height) + 1}px",
                print_background=True,
                page_ranges="1",
            )
            print(path)
        browser.close()


if __name__ == "__main__":
    main()
