"""Find the hand-marked court that applies to a given rally (ADR-0015 D2).

Courts are marked once per tournament because the camera is fixed within one —
verified over CVSPORTS by phase-correlating the court edges between rallies:
ten of the eleven tournaments move under 3 px. The exception is VALLADOLID,
where the camera rises 9.9 px from rally 04 onwards, so a tournament can also
carry per-range files (`<TOURNAMENT>@04.json` applies from rally 04 on).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import numpy.typing as npt

COURTS_DIR = Path("data/datasets/courts")
_RALLY_RE = re.compile(r"^(?P<tournament>\d{8}_[A-Z]+)_(?P<index>\d+)$")


def tournament_of(rally_stem: str) -> str:
    """ "20230528_VIGO_03" -> "20230528_VIGO"."""
    match = _RALLY_RE.match(rally_stem)
    return match.group("tournament") if match else rally_stem


def court_file_for(rally_stem: str, courts_dir: Path = COURTS_DIR) -> Path | None:
    """The court JSON that applies to this rally, or None if unmarked.

    Prefers the most specific range file: for rally 05 of a tournament with
    `X.json` and `X@04.json`, the latter wins because 5 >= 4.
    """
    match = _RALLY_RE.match(rally_stem)
    if match is None:
        return None
    tournament, index = match.group("tournament"), int(match.group("index"))

    best: tuple[int, Path] | None = None
    for path in courts_dir.glob(f"{tournament}@*.json"):
        start = int(path.stem.split("@")[1])
        if index >= start and (best is None or start > best[0]):
            best = (start, path)
    if best is not None:
        return best[1]
    base = courts_dir / f"{tournament}.json"
    return base if base.exists() else None


def load_corners(path: Path) -> npt.NDArray[np.float64]:
    """The four court corners in pixels (for the player mask)."""
    return np.array(json.loads(path.read_text())["corners_px"], dtype=np.float64)


def load_homography(path: Path) -> npt.NDArray[np.float64]:
    """The px->m homography."""
    return np.array(json.loads(path.read_text())["homography"], dtype=np.float64)
