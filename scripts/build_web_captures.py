"""Write the screenshots of the web viewer for section 4.3 (memoria/figuras/capturas/).

Starts the viewer locally, opens it in a headless browser and saves four images:
the home page, and the page of one point whole (visor-punto) and split in two (the
video, the court and the current shot above; the timeline and the player cards
below). The video is paused at a fixed instant, so the images come out the same
every time.

Needs the points already exported (scripts/export_points.py) and the web's
dependencies (npm install in packages/web). Playwright is not a dependency of the
project, so it is added for the run:

    uv run --with playwright==1.61.0 python scripts/build_web_captures.py

On a WSL without the browser's system libraries, point LD_LIBRARY_PATH at a copy
of them (libnss3, libnspr4, libasound2) extracted in user space.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import Page, sync_playwright

from padel_cv.paths import REPO

OUT = REPO / "memoria" / "figuras" / "capturas"
WEB = REPO / "packages" / "web"
PORT = 5199

POINT = "20230927_MADRID_01"
"""The point of the second page: the same MADRID point as the players figure."""
AT_S = 6.0
"""Instant the video is paused at: J4's smash, on screen with its verdict."""
WIDTH = 1440
"""Width of the browser window, in CSS pixels: the narrower, the larger the text
looks once the page is fitted to the width of the figure."""
HEIGHT = 900
SCALE = 3
"""Device pixels per CSS pixel, so text stays sharp in print."""
QUALITY = 95
"""JPEG quality. The PDF embeds each image as it is, so its size adds to the
PDF's: lossless PNG at 4x added about 25 MB for no difference visible in print,
this adds about 5 MB."""
TIMELINE_HEADING = "Línea de tiempo"
"""Where the point's page is split: just above the timeline card."""
GAP_PX = 12


def wait_for_port(port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(("localhost", port)) == 0:
                return
        time.sleep(0.3)
    raise SystemExit(f"el visor no arrancó en el puerto {port}")


def save(image: np.ndarray, path: Path) -> None:
    """JPEG at QUALITY, or lossless if the path asks for a PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    options = [] if path.suffix == ".png" else [cv2.IMWRITE_JPEG_QUALITY, QUALITY]
    cv2.imwrite(str(path), image, options)
    print(path)


def full_page(page: Page) -> np.ndarray:
    raw = page.screenshot(full_page=True, type="png")
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def capture_home(page: Page, out: Path, ext: str) -> None:
    page.goto(f"http://localhost:{PORT}/#/")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)
    save(full_page(page), out / f"visor-portada.{ext}")


def capture_point(page: Page, out: Path, scale: int, ext: str) -> None:
    page.goto(f"http://localhost:{PORT}/#/p/{POINT}")
    page.wait_for_selector("video")
    page.wait_for_function("document.querySelector('video').readyState >= 2")
    # Paused at AT_S, without the browser's controls drawn over the frame.
    page.evaluate(
        """async (at) => {
          const video = document.querySelector('video');
          video.pause();
          video.removeAttribute('controls');
          video.currentTime = at;
          await new Promise((done) => video.addEventListener('seeked', done, { once: true }));
        }""",
        AT_S,
    )
    page.wait_for_timeout(1000)
    split_css = page.evaluate(
        """(heading) => {
          const label = [...document.querySelectorAll('*')].find((e) =>
            e.children.length === 0 && e.textContent.trim().toLowerCase().startsWith(heading));
          let card = label;
          while (card.parentElement && card.offsetHeight < 150) card = card.parentElement;
          return card.getBoundingClientRect().top + window.scrollY;
        }""",
        TIMELINE_HEADING.lower(),
    )
    image = full_page(page)
    save(image, out / f"visor-punto.{ext}")
    split = int((split_css - GAP_PX) * scale)
    save(image[:split], out / f"visor-punto-arriba.{ext}")
    save(image[split:], out / f"visor-punto-abajo.{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--scale", type=int, default=SCALE)
    parser.add_argument("--png", action="store_true", help="Lossless PNG instead of JPEG")
    args = parser.parse_args()
    ext = "png" if args.png else "jpg"

    server = subprocess.Popen(
        ["npx", "vite", "--port", str(PORT), "--strictPort"],
        cwd=WEB,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        wait_for_port(PORT)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(
                viewport={"width": args.width, "height": HEIGHT}, device_scale_factor=args.scale
            )
            capture_home(page, args.out, ext)
            capture_point(page, args.out, args.scale, ext)
            browser.close()
    finally:
        # npx starts vite as a child: stop the whole group, not just npx.
        os.killpg(server.pid, signal.SIGTERM)
        server.wait()


if __name__ == "__main__":
    main()
