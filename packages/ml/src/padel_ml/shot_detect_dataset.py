"""Dataset builder for the learned SHOT DETECTOR (Modelo 1, ADR-0013).

The heuristic detector (ADR-0012) hit only 15-52% recall because a fixed
geometric rule can't cope with the ball vanishing at impact or noisy turns. Here
we build training windows so a model can LEARN "this pose-and-ball motion is a
strike" — the way a human recognises a hit without any rule.

Each sample is a fixed window of T frames around a candidate frame, carrying two
aligned channels:

- **pose**: the hitter's skeleton, normalized per frame (centred on the hips,
  scaled by torso) so the model sees pose *shape over time*, not court position.
- **ball**: the ball's (x, y) expressed in that SAME normalized frame — so "ball
  arriving at the wrist" is a position the model can read directly, invariant to
  where on court the rally is. Missing ball frames are held-filled and flagged.

Label is binary: 1 if the window is centred on a real shot's impact, 0 for
negatives sampled from rally frames far from any shot. (Per-frame localization is
the natural next step; per-window keeps this first detector simple — ADR-0013.)

Ball source is a parameter: GT here (step 1, isolates whether the signal exists /
the theoretical ceiling), swappable for our own TrackNet later (step 2, robustness
to an imperfect ball). Training data still uses OUR pose at inference (ADR-0005).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from padel_cv.clip_builder import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    normalize_skeleton,
)
from padel_ml.shot_eval import (
    ShotBlock,
    load_ball_track,
    load_shot_blocks,
)

FloatArray = npt.NDArray[np.float32]

WINDOW = 32  # frames per window (~1s at 30fps), matches the classifier's clips
_HALF = WINDOW // 2


@dataclass
class DetectWindow:
    """One training window for the shot detector."""

    pose: FloatArray  # (WINDOW, 17, 3) normalized skeleton over time
    ball: FloatArray  # (WINDOW, 3): normalized (x, y) + present-flag per frame
    label: int  # 1 = centred on a shot impact, 0 = no shot
    source_frame: int


def _hip_center_and_scale(skeleton: FloatArray) -> tuple[FloatArray, float]:
    """The same normalization anchor the skeleton uses, so the ball lands in the
    skeleton's own coordinate frame."""
    hips = skeleton[[LEFT_HIP, RIGHT_HIP], :2]
    shoulders = skeleton[[LEFT_SHOULDER, RIGHT_SHOULDER], :2]
    hip_center = hips.mean(axis=0)
    torso = float(np.linalg.norm(shoulders.mean(axis=0) - hip_center))
    return hip_center, (torso if torso > 1.0 else 1.0)


def _normalize_ball(
    ball_xy: tuple[float, float] | None, hip_center: FloatArray, scale: float
) -> tuple[float, float, float]:
    """Ball into the skeleton's normalized frame; present-flag 0 when missing."""
    if ball_xy is None:
        return 0.0, 0.0, 0.0
    x = (ball_xy[0] - hip_center[0]) / scale
    y = (ball_xy[1] - hip_center[1]) / scale
    return float(x), float(y), 1.0


def _closest_person_to_ball(
    persons: list[FloatArray], ball_xy: tuple[float, float] | None
) -> FloatArray | None:
    """The hitter: the skeleton whose nearest wrist is closest to the ball. With
    no ball, fall back to the first person (a negative window still needs a pose)."""
    if not persons:
        return None
    if ball_xy is None:
        return persons[0]
    from padel_ml.shot_eval import _L_WRI, _R_WRI

    best, best_d = persons[0], float("inf")
    for kp in persons:
        for wi in (_L_WRI, _R_WRI):
            if kp[wi, 2] <= 0.0:
                continue
            d = float(np.hypot(kp[wi, 0] - ball_xy[0], kp[wi, 1] - ball_xy[1]))
            if d < best_d:
                best_d, best = d, kp
    return best


def _window_around(
    frame: int,
    persons_by_frame: dict[int, list[FloatArray]],
    ball_by_frame: dict[int, tuple[float, float]],
) -> tuple[FloatArray, FloatArray] | None:
    """Build the (pose, ball) window centred on `frame`, or None if the hitter's
    skeleton is missing for more than half of it."""
    frames = range(frame - _HALF, frame - _HALF + WINDOW)
    # Pick the hitter at the centre frame; follow that same person across the window
    # by nearest-skeleton continuity is overkill here — GT gives one dense person
    # near the ball, so we pick the ball-closest person per frame.
    raw_pose: list[FloatArray | None] = []
    raw_ball: list[tuple[float, float] | None] = []
    for f in frames:
        persons = persons_by_frame.get(f, [])
        ball = ball_by_frame.get(f)
        hitter = _closest_person_to_ball(persons, ball)
        raw_pose.append(hitter)
        raw_ball.append(ball)

    present = [p for p in raw_pose if p is not None]
    if len(present) < WINDOW // 2:
        return None

    # Hold-fill missing skeletons (forward, else backward) as the classifier does.
    filled: list[FloatArray] = []
    last: FloatArray | None = None
    for p in raw_pose:
        if p is not None:
            last = p
        filled.append(p if p is not None else (last if last is not None else present[0]))

    pose_out = np.stack([normalize_skeleton(p) for p in filled]).astype(np.float32)
    ball_out = np.zeros((WINDOW, 3), dtype=np.float32)
    for i, (p, b) in enumerate(zip(filled, raw_ball, strict=True)):
        hip_center, scale = _hip_center_and_scale(p)
        ball_out[i] = _normalize_ball(b, hip_center, scale)
    return pose_out, ball_out


def build_detect_windows(
    persons_by_frame: dict[int, list[FloatArray]],
    ball_by_frame: dict[int, tuple[float, float]],
    blocks: list[ShotBlock],
    *,
    negatives_ratio: float = 1.0,
    min_gap: int = 30,
    seed: int = 0,
) -> list[DetectWindow]:
    """Positive windows centred on each shot's impact + sampled negatives.

    Impact frame is approximated by the block centre (the classifier uses the same
    convention). Negatives are frames at least `min_gap` from any shot centre.
    """
    rng = np.random.default_rng(seed)
    shot_centres = [b.centre for b in blocks]
    windows: list[DetectWindow] = []

    for centre in shot_centres:
        w = _window_around(centre, persons_by_frame, ball_by_frame)
        if w is not None:
            windows.append(DetectWindow(w[0], w[1], 1, centre))

    n_neg = int(len(windows) * negatives_ratio)
    all_frames = sorted(persons_by_frame.keys())
    candidates = [f for f in all_frames if all(abs(f - c) > min_gap for c in shot_centres)]
    rng.shuffle(candidates)
    for f in candidates:
        if len([w for w in windows if w.label == 0]) >= n_neg:
            break
        w = _window_around(f, persons_by_frame, ball_by_frame)
        if w is not None:
            windows.append(DetectWindow(w[0], w[1], 0, f))
    return windows


@dataclass
class DetectDataset:
    """Persisted detector windows: pose (N,T,17,3), ball (N,T,3), y (N,), match (N,)."""

    pose: FloatArray
    ball: FloatArray
    labels: npt.NDArray[np.int64]
    matches: npt.NDArray[np.int64]

    def save(self, path: Path | str) -> None:
        np.savez_compressed(
            path, pose=self.pose, ball=self.ball, labels=self.labels, matches=self.matches
        )


def assemble_from_match(
    pose_json: Path,
    ball_json: Path,
    shots_csv: Path,
    *,
    negatives_ratio: float = 1.0,
    seed: int = 0,
) -> list[DetectWindow]:
    """All detector windows for one match, from GT pose + GT ball (step 1)."""
    persons = _load_persons(pose_json)
    ball = {s.frame_index: (s.x_px, s.y_px) for s in load_ball_track(ball_json)}
    blocks = load_shot_blocks(shots_csv)
    return build_detect_windows(persons, ball, blocks, negatives_ratio=negatives_ratio, seed=seed)


def _load_persons(pose_json: Path) -> dict[int, list[FloatArray]]:
    """frame -> list of (17,3) skeletons for every person in that frame (GT pose)."""
    import json

    from padel_ml.shot_eval import _frame_index

    data = json.loads(pose_json.read_text())
    frame_of = {img["id"]: _frame_index(img["file_name"]) for img in data["images"]}
    out: dict[int, list[FloatArray]] = {}
    for ann in data["annotations"]:
        kp = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        out.setdefault(frame_of[ann["image_id"]], []).append(kp)
    return out


def to_dataset(windows_per_match: list[list[DetectWindow]]) -> DetectDataset:
    """Stack windows from several matches into arrays with a match id (cross-match)."""
    pose, ball, labels, matches = [], [], [], []
    for match_id, windows in enumerate(windows_per_match):
        for w in windows:
            pose.append(w.pose)
            ball.append(w.ball)
            labels.append(w.label)
            matches.append(match_id)
    return DetectDataset(
        pose=np.stack(pose).astype(np.float32),
        ball=np.stack(ball).astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
        matches=np.array(matches, dtype=np.int64),
    )
