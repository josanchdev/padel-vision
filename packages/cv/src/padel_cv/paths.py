"""Where everything the project reads and writes lives (see data/README.md).

    data/raw/       third-party datasets, as downloaded
    data/labels/    our own labelling (in git)
    data/cache/     derived and regenerable
    runs/           trained models (not published)
    docs/metrics/   the evidence (in git)

Absolute paths, so the scripts work from any directory.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "data"

CVSPORTS = DATA / "raw" / "cvsports_padel"
"""CVSPORTS_Padel: `rallies/*.mp4` and `metadata/` (hits.csv, hit_assignments.xlsx)."""
CVSPORTS_RALLIES = CVSPORTS / "rallies"
CVSPORTS_HITS = CVSPORTS / "metadata" / "hits.csv"
PADELTRACKER = DATA / "raw" / "padeltracker100"
"""PadelTracker100: the two Barcelona 2022 finals and their annotations."""

SHOT_TYPE_LABELS = DATA / "labels" / "shot_types"
COURTS = DATA / "labels" / "courts"

CACHE = DATA / "cache"
RALLY_FEATURES = CACHE / "rally_features"
AUDIO_FEATURES = CACHE / "audio_features.npz"

RUNS = REPO / "runs"
METRICS = REPO / "docs" / "metrics"
FIGURES = METRICS / "figures"
WEB_POINTS = REPO / "packages" / "web" / "public" / "points"
