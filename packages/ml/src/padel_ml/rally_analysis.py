"""One rally, end to end: when, who, and what kind of shot.

The single code path behind the demo video, the web export and every
measurement: `track_rally` is the one per-frame pass (also behind the feature
cache and the hit-assignment evidence), and `resolve_shots` the one WHO + WHAT
step (also behind the whole-chain evaluation). Keeping them on one code path is
what gives the evaluation its meaning: a score measured on a copy of the
pipeline says nothing about the code that actually runs.

    WHEN   audio CRNN over the soundtrack                (ADR-0015 A)
    WHO    pose + ball, weighted vote around each hit    (ADR-0015 D)
    WHAT   BST-0 over the hitter's window + serve rule   (ADR-0016)

It works on a rally — one point — not on a whole match. The serve rule assumes
the first detected hit opens the point, as it does in every training rally, and
a full match would also carry replays and dead time between points.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
import torch

from padel_cv.bounces import BallSample, Bounce, detect_bounces
from padel_cv.court import localize_pose
from padel_cv.court_registry import load_corners, load_homography
from padel_cv.match_data import BounceRecord, MatchAnalysis, PlayerFrameRecord, ShotRecord
from padel_cv.pipeline import Frame, ImageArray, PoseDetection
from padel_cv.player_identity import PlayerIdentityTracker, court_mask_polygon, filter_players
from padel_cv.stages.pose import PlayerPoseStage
from padel_ml.audio_train import detect_hits_in_audio
from padel_ml.ball_infer import BallDetector, BallHit
from padel_ml.ball_postprocess import postprocess_ball
from padel_ml.ball_trajectory import clean_track
from padel_ml.hit_assignment import assign_hit, frame_states
from padel_ml.shot_type_dataset import CLASSES, SEQ_LEN, ShotWindow, build_window
from padel_ml.shot_type_model import ShotTypeBST
from padel_ml.shot_type_train import best_class

POSE_CONFIDENCE = 0.25
"""Detection threshold for people. The same value the training features were
extracted with (scripts/extract_rally_features.py): the classifier learned from
poses produced this way, so inference must produce them the same way."""

UNKNOWN_PLAYER = 0
"""`player_id` written to the web's JSON when no player could be tied to a hit."""

Classifier = Callable[[ShotWindow], npt.NDArray[np.float32]]
"""Window -> class probabilities, in the order of `CLASSES`."""


@dataclass(frozen=True)
class ModelPaths:
    """Where the three trained models live."""

    audio: Path
    ball: Path
    shot_type: Path

    @classmethod
    def under(cls, runs_dir: Path) -> ModelPaths:
        """The layout the training scripts write to."""
        return cls(
            audio=runs_dir / "audio" / "audio_crnn.pt",
            ball=runs_dir / "ball_full" / "tracknetv3.pt",
            shot_type=runs_dir / "shot_type" / "bst0.pt",
        )


class RallyModels:
    """Every model the analysis needs, loaded once and reused across videos."""

    def __init__(self, paths: ModelPaths, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_checkpoint = paths.audio
        self.pose = PlayerPoseStage(device=self.device, confidence=POSE_CONFIDENCE)
        self.ball = BallDetector(paths.ball, device=self.device)
        checkpoint = torch.load(paths.shot_type, map_location=self.device, weights_only=False)
        self.seq_len = int(checkpoint["seq_len"])
        self.shot_type = ShotTypeBST(seq_len=self.seq_len, n_classes=len(CLASSES))
        self.shot_type.load_state_dict(checkpoint["state_dict"])
        self.shot_type.to(self.device).eval()

    def classify(self, window: ShotWindow) -> npt.NDArray[np.float32]:
        with torch.no_grad():
            logits = self.shot_type(
                torch.from_numpy(window.pose)[None].to(self.device),
                torch.from_numpy(window.ball)[None].to(self.device),
            )
        return cast(npt.NDArray[np.float32], torch.softmax(logits, dim=1)[0].cpu().numpy())


@dataclass(frozen=True)
class Shot:
    """One detected hit: when, who and what."""

    frame_index: int
    timestamp_s: float
    player_id: int | None
    """1-4, or None when no player could be tied to the hit."""
    shot_type: str | None
    """One of `CLASSES`, or None when the hit was not classified: no hitter, the
    hitter missing from most of the window, or below the confidence threshold."""
    confidence: float
    """Probability of the chosen class (0 when not classified)."""
    probabilities: tuple[float, ...] | None
    """Raw classifier output, before the serve rule. Kept for evaluation."""


@dataclass
class RallyAnalysis:
    """Everything extracted from one rally."""

    video: Path
    fps: float
    width: int
    height: int
    n_frames: int
    court: Path | None
    shots: list[Shot]
    players: dict[int, list[PoseDetection]]
    """Identified on-court players per frame, with court positions when the court is known."""
    ball: dict[int, tuple[float, float]]
    """Post-processed ball track: what the vote and the classifier read."""
    ball_smoothed: dict[int, tuple[float, float]]
    """Physics-cleaned ball track: what gets drawn."""
    bounces: list[Bounce]

    def to_match_data(self) -> MatchAnalysis:
        """The versioned JSON contract the web consumes (ADR-0010)."""
        data = MatchAnalysis(source_video=str(self.video), fps=self.fps)
        for frame in sorted(self.players):
            for pose in self.players[frame]:
                x_m, y_m = pose.court_position_m or (None, None)
                data.players.append(
                    PlayerFrameRecord(
                        frame_index=frame,
                        timestamp_s=frame / self.fps,
                        player_id=cast(int, pose.player_id),
                        track_id=pose.track_id,
                        court_x_m=x_m,
                        court_y_m=y_m,
                        on_court=pose.on_court,
                    )
                )
        data.shots = [
            ShotRecord(
                frame_index=shot.frame_index,
                timestamp_s=shot.timestamp_s,
                player_id=shot.player_id or UNKNOWN_PLAYER,
                label=shot.shot_type or "Unclassified",
                confidence=shot.confidence,
            )
            for shot in self.shots
        ]
        data.bounces = [BounceRecord(b.frame_index, b.x_px, b.y_px) for b in self.bounces]
        return data


def resolve_shots(
    hit_frames: list[int],
    players: dict[int, list[PoseDetection]],
    ball: dict[int, tuple[float, float]],
    fps: float,
    classify: Classifier,
    seq_len: int = SEQ_LEN,
    min_confidence: float = 0.0,
) -> list[Shot]:
    """WHO and WHAT for every detected hit, given the per-frame detections.

    Pure — the classifier comes in as a function — so it can be tested without
    a GPU, a video or trained weights.
    """
    states = frame_states(players, ball)

    shots: list[Shot] = []
    for position, frame in enumerate(hit_frames):
        hitter = assign_hit(frame, states)
        shot_type: str | None = None
        confidence = 0.0
        probabilities: tuple[float, ...] | None = None
        if hitter is not None:
            window = build_window(
                players, ball, hit_frames, position, hitter, fps, 0, "", "", seq_len
            )
            if window is not None:
                raw = classify(window)
                # Only the rally's opening hit may be a serve (ADR-0016).
                best = best_class(raw, may_serve=position == 0)
                confidence = float(raw[best])
                probabilities = tuple(float(p) for p in raw)
                if confidence >= min_confidence:
                    shot_type = CLASSES[best]
        shots.append(Shot(frame, frame / fps, hitter, shot_type, confidence, probabilities))
    return shots


@dataclass
class RallyTracks:
    """People, who is who, and the ball over a whole rally."""

    fps: float
    width: int
    height: int
    n_frames: int
    players: dict[int, list[PoseDetection]]
    """Identified on-court players per frame, with the detector's own boxes."""
    raw_ball: list[BallHit]
    """Every ball detection, before post-processing."""


def track_rally(
    video: Path,
    pose: PlayerPoseStage,
    ball: BallDetector,
    court: Path | None,
    on_progress: Callable[[float], None] | None = None,
) -> RallyTracks:
    """The per-frame pass over a rally: pose, the J1-J4 identity and the ball.

    The one loop behind the system, the feature cache and the hit-assignment
    evidence. It starts from clean detectors: the pose tracker and the ball
    buffer would otherwise carry the previous video into this one.
    """
    pose.reset()
    ball.reset()
    polygon = court_mask_polygon(load_corners(court)) if court is not None else None
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    identity = PlayerIdentityTracker(fps=fps)
    candidates: dict[int, list[PoseDetection]] = {}
    raw_ball: list[BallHit] = []
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frame = pose.process(
                Frame(index=index, timestamp_s=index / fps, image=cast(ImageArray, image))
            )
            on_court = filter_players(frame.poses, polygon)
            identity.update(index, on_court)
            # Keep the pose objects and read their IDs only once the video is
            # done: the identity tracker back-fills IDs over its start-up window,
            # which is exactly where every serve lives.
            candidates[index] = on_court
            hit = ball.detect(image, index)
            if hit is not None:
                raw_ball.append(hit)
            index += 1
            if on_progress is not None and total > 0 and index % 25 == 0:
                on_progress(min(index / total, 1.0))
    finally:
        capture.release()

    players = {i: [p for p in poses if p.player_id is not None] for i, poses in candidates.items()}
    return RallyTracks(fps, width, height, index, players, raw_ball)


def analyze_rally(
    video: Path,
    models: RallyModels,
    court: Path | None,
    min_confidence: float = 0.0,
    on_progress: Callable[[float], None] | None = None,
) -> RallyAnalysis:
    """Run the whole chain over one rally.

    `court` has no default on purpose. Without it the crowd is not filtered and
    a spectator can win the vote for a hit, so running without a court has to be
    a visible decision of the caller, never a silent fallback.
    """
    hit_times = detect_hits_in_audio(video, models.audio_checkpoint, device=models.device)
    tracks = track_rally(video, models.pose, models.ball, court, on_progress)
    fps, players = tracks.fps, tracks.players
    if court is not None:
        homography = load_homography(court)
        for poses in players.values():
            for pose in poses:
                localize_pose(homography, pose)

    # Two ball tracks on purpose. The vote reads the plain post-processed one:
    # measured against the paper's ground truth the parabolic smoothing costs it
    # about a point, because a fit spanning a hit rounds off the very moment the
    # vote depends on. Drawing uses the smoothed one, whose jerk drops from 23 px
    # to under 7 and reads as a trajectory rather than a scatter of dots.
    ball = {b.frame_index: (b.x_px, b.y_px) for b in postprocess_ball(tracks.raw_ball)}
    ball_smoothed = {p.frame_index: (p.x_px, p.y_px) for p in clean_track(tracks.raw_ball)}

    hit_frames = sorted({round(t * fps) for t in hit_times})
    shots = resolve_shots(
        hit_frames, players, ball, fps, models.classify, models.seq_len, min_confidence
    )
    # Bounces on the plain track too, for the same reason as the vote: a fit
    # across the bounce rounds off the valley the detector looks for.
    bounces = detect_bounces(
        [BallSample(f, x, y) for f, (x, y) in sorted(ball.items())], set(hit_frames)
    )
    if on_progress is not None:
        on_progress(1.0)
    return RallyAnalysis(
        video=video,
        fps=fps,
        width=tracks.width,
        height=tracks.height,
        n_frames=tracks.n_frames,
        court=court,
        shots=shots,
        players=players,
        ball=ball,
        ball_smoothed=ball_smoothed,
        bounces=bounces,
    )
