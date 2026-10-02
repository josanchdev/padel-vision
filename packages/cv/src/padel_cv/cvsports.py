"""The CVSPORTS_Padel dataset (Decorte et al., CVPRW 2024): its names and files.

Rallies are named DATE_LOCATION_INDEX ("20230528_VIGO_03"): one tournament per
DATE_LOCATION, filmed by one fixed camera. The hit instants live in
`metadata/hits.csv` as (filename, start, end) windows in seconds.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

_RALLY_RE = re.compile(r"^(?P<tournament>\d{8}_[A-Z]+)_(?P<index>\d+)$")


def parse_rally(rally_stem: str) -> tuple[str, int] | None:
    """ "20230528_VIGO_03" -> ("20230528_VIGO", 3); None for any other name."""
    match = _RALLY_RE.match(rally_stem)
    return (match.group("tournament"), int(match.group("index"))) if match else None


def tournament_of(rally_stem: str) -> str:
    """ "20230528_VIGO_03" -> "20230528_VIGO". A clip outside the dataset's
    naming is a tournament of its own."""
    parsed = parse_rally(rally_stem)
    return parsed[0] if parsed else rally_stem


def load_hits_csv(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Rally filename -> its hits, as (start, end) windows in seconds."""
    hits: dict[str, list[tuple[float, float]]] = {}
    for row in csv.DictReader(csv_path.open()):
        hits.setdefault(row["filename"], []).append((float(row["start"]), float(row["end"])))
    return hits
