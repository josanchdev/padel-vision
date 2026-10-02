"""The rallies the paper annotated with who hit, as the vote sees them (ADR-0015).

`annotated_rallies` runs each VIGO rally with a per-hit player annotation
through the system's own per-frame pass (`track_rally`) and builds the vote's
input the way `resolve_shots` does. The evidence scripts then score the vote at
the ANNOTATED hit times, so the assignment step is measured alone
(`scripts/evidence_hit_assignment.py`, `scripts/experiment_assignment_window.py`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from padel_cv.court_registry import court_file_for
from padel_cv.stages.pose import PlayerPoseStage
from padel_ml.ball_infer import BallDetector
from padel_ml.ball_postprocess import postprocess_ball
from padel_ml.hit_assignment import FrameState, frame_states
from padel_ml.hit_assignment_gt import HitAssignment, load_hit_assignments
from padel_ml.rally_analysis import track_rally


@dataclass
class AnnotatedRally:
    """One rally with the paper's per-hit player annotation, run through the system."""

    name: str
    fps: float
    states: dict[int, FrameState]
    hits: list[HitAssignment]


def annotated_rallies(
    dataset_dir: Path, courts_dir: Path, pose: PlayerPoseStage, ball: BallDetector
) -> Iterator[AnnotatedRally]:
    """Every rally with a per-hit player annotation, in order, ready to vote on."""
    truth = load_hit_assignments(dataset_dir / "metadata" / "hit_assignments.xlsx")
    for rally in sorted(truth):
        video = dataset_dir / "rallies" / f"{rally}.mp4"
        if not video.exists():
            continue
        court = court_file_for(rally, courts_dir)
        if court is None:
            raise FileNotFoundError(f"{rally}: no court marked (padel-cv annotate-court)")
        tracks = track_rally(video, pose, ball, court)
        track = {b.frame_index: (b.x_px, b.y_px) for b in postprocess_ball(tracks.raw_ball)}
        yield AnnotatedRally(rally, tracks.fps, frame_states(tracks.players, track), truth[rally])
