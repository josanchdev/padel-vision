"""Write the classifier's input figure of section 4.2.5: memoria/figuras/entrada.pdf.

A smash of a MALMO point as the classifier receives it. The window is built with
the system's own `build_window` from the cache the classifier was trained on
(data/cache/rally_features), and the hitter is chosen by the system's assignment.
Needs no video and no GPU.

    uv run python scripts/build_input_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from padel_ml.hit_assignment import assign_hit, frame_states
from padel_ml.rally_features import load_rally_features
from padel_ml.shot_type_dataset import (
    CLASS_TO_INDEX,
    SEQ_LEN,
    build_window,
    load_labels,
    window_bounds,
)
from padel_ml.thesis.input_figure import draw_input_figure, step_frames, trail

from padel_cv.cvsports import tournament_of
from padel_cv.paths import RALLY_FEATURES, REPO, SHOT_TYPE_LABELS

OUT = REPO / "memoria" / "figuras" / "entrada.pdf"

RALLY = "20231112_MALMO_02"
HIT = 615
"""A smash by J4 (near side): the arm fully up and a clean skeleton at the hit."""
REACH = 3
"""Frames either side of the hit drawn fading: enough to show a sequence, few
enough that the hit's own skeleton still reads."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reach", type=int, default=REACH, help="Frames either side of the hit drawn fading"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    features = load_rally_features(RALLY_FEATURES / f"{RALLY}.npz")
    labels = load_labels(SHOT_TYPE_LABELS / f"{RALLY}.csv")
    hits = [frame for frame, _ in labels]
    index = hits.index(HIT)
    hitter = assign_hit(HIT, frame_states(features.players, features.ball))
    if hitter is None:
        raise SystemExit(f"golpe {HIT}: sin jugador asignado")
    window = build_window(
        features.players,
        features.ball,
        hits,
        index,
        hitter,
        features.fps,
        CLASS_TO_INDEX[labels[index][1]],
        RALLY,
        tournament_of(RALLY),
    )
    if window is None:
        raise SystemExit(f"golpe {HIT}: el jugador falta en la ventana")

    frames = step_frames(*window_bounds(hits, index, features.fps), SEQ_LEN)
    steps = trail(frames, HIT, args.reach)
    hit_step = next(s for s in steps if frames[s] == HIT)
    draw_input_figure(window, steps, hit_step, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
