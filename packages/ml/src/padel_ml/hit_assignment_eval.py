"""Measure hit assignment against the paper's ground truth (ADR-0015).

Runs the full backbone over the 16 VIGO rallies that ship with per-hit player
annotations and reports the same two numbers the paper does (Table 3): accuracy
for the specific player and for the team only, weighted by each rally's hit
count. The paper gets 83.70% / 86.83%.

Evaluation uses the ANNOTATED hit times, not our audio detections, so this
measures the assignment step alone — mixing in detection errors would conflate
two things. (The audio detector is measured separately, F1 0.93.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from padel_cv.pipeline import BallDetection, Frame, ImageArray, PoseDetection
from padel_cv.player_identity import PlayerIdentityTracker, court_mask_polygon, filter_players
from padel_cv.stages.pose import PlayerPoseStage
from padel_ml.ball_infer import BallDetector, BallHit
from padel_ml.ball_postprocess import postprocess_ball
from padel_ml.hit_assignment import FrameState, assign_hit, team_alternation_sweep
from padel_ml.hit_assignment_gt import HitAssignment, load_hit_assignments

# Court corner keypoint indices in the 13-point schema (near/far, left/right).
CORNER_INDICES = (0, 1, 12, 11)


@dataclass
class RallyResult:
    """Per-rally accuracy, mirroring the paper's Table 3."""

    rally: str
    player_correct: int
    team_correct: int
    total: int
    unassigned: int

    @property
    def player_accuracy(self) -> float:
        return self.player_correct / self.total if self.total else 0.0

    @property
    def team_accuracy(self) -> float:
        return self.team_correct / self.total if self.total else 0.0


@dataclass
class AssignmentEval:
    """Weighted global accuracy over all evaluated rallies."""

    rallies: list[RallyResult]

    @property
    def total_hits(self) -> int:
        return sum(r.total for r in self.rallies)

    @property
    def player_accuracy(self) -> float:
        return sum(r.player_correct for r in self.rallies) / max(self.total_hits, 1)

    @property
    def team_accuracy(self) -> float:
        return sum(r.team_correct for r in self.rallies) / max(self.total_hits, 1)

    @property
    def unassigned(self) -> int:
        return sum(r.unassigned for r in self.rallies)


def build_states(
    video: Path,
    pose_stage: PlayerPoseStage,
    ball_detector: BallDetector,
    court_corners_px: np.ndarray | None,
    fps: float,
) -> dict[int, FrameState]:
    """Run pose + ball + identity over a rally and collect per-frame state."""
    polygon = court_mask_polygon(court_corners_px) if court_corners_px is not None else None
    identity = PlayerIdentityTracker(fps=fps)
    capture = cv2.VideoCapture(str(video))
    poses_by_frame: dict[int, list[PoseDetection]] = {}
    raw_ball: list[BallHit] = []
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame = pose_stage.process(
            Frame(index=index, timestamp_s=index / fps, image=cast(ImageArray, image))
        )
        players = filter_players(frame.poses, polygon)
        identity.update(index, players)
        # keep every pose object: the identity tracker back-fills IDs on the
        # startup window once numbering is fixed, so filtering here would drop
        # poses that are about to be identified.
        poses_by_frame[index] = players
        hit = ball_detector.detect(image, index)
        if hit is not None:
            raw_ball.append(hit)
        index += 1
    capture.release()

    ball_by_frame = {
        h.frame_index: BallDetection((h.x_px, h.y_px), h.confidence)
        for h in postprocess_ball(raw_ball)
    }
    return {
        i: FrameState(
            poses=[p for p in poses_by_frame.get(i, []) if p.player_id is not None],
            ball=ball_by_frame.get(i),
        )
        for i in range(index)
    }


def evaluate_rally(
    states: dict[int, FrameState], truth: list[HitAssignment], fps: float, rally: str
) -> RallyResult:
    """Assign each annotated hit and score it against the annotation."""
    predictions: dict[int, int | None] = {}
    frame_of: dict[int, HitAssignment] = {}
    for hit in truth:
        frame = round(hit.time_s * fps)
        predictions[frame] = assign_hit(frame, states)
        frame_of[frame] = hit
    predictions = team_alternation_sweep(predictions)

    player_correct = team_correct = unassigned = 0
    for frame, predicted in predictions.items():
        expected = frame_of[frame]
        if predicted is None:
            unassigned += 1
            continue
        if predicted < 0:  # team-only recovery from the alternation sweep
            team_correct += int(-predicted == expected.team)
            continue
        player_correct += int(predicted == expected.slot)
        team_correct += int((1 if predicted <= 2 else 2) == expected.team)
    return RallyResult(
        rally=rally,
        player_correct=player_correct,
        team_correct=team_correct,
        total=len(truth),
        unassigned=unassigned,
    )


def evaluate_assignment(
    dataset_dir: Path,
    court_corners_px: np.ndarray | None = None,
    pose_model: str | None = None,
    ball_checkpoint: Path | None = None,
    device: str | None = None,
    limit: int | None = None,
) -> AssignmentEval:
    """Score the assignment backbone over every annotated rally."""
    truth_by_rally = load_hit_assignments(dataset_dir / "metadata" / "hit_assignments.xlsx")
    pose_stage = (
        PlayerPoseStage(model_name=pose_model, device=device)
        if pose_model
        else PlayerPoseStage(device=device)
    )
    results: list[RallyResult] = []
    for rally in sorted(truth_by_rally)[:limit]:
        video = dataset_dir / "rallies" / f"{rally}.mp4"
        if not video.exists():
            continue
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.release()
        ball_detector = BallDetector(
            ball_checkpoint or Path("runs/ball_full/tracknetv3.pt"), device=device
        )
        states = build_states(video, pose_stage, ball_detector, court_corners_px, fps)
        results.append(evaluate_rally(states, truth_by_rally[rally], fps, rally))
    return AssignmentEval(rallies=results)
