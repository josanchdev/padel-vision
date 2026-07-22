"""Pipeline stages."""

from padel_cv.stages.court_detection import CourtDetectionStage
from padel_cv.stages.gt_court import GroundTruthCourtStage
from padel_cv.stages.identity import PlayerIdentityStage
from padel_cv.stages.pose import PlayerPoseStage
from padel_cv.stages.shots import DummyShotStage

__all__ = [
    "CourtDetectionStage",
    "DummyShotStage",
    "GroundTruthCourtStage",
    "PlayerIdentityStage",
    "PlayerPoseStage",
]
