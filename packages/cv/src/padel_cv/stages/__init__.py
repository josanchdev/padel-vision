"""Pipeline stages."""

from padel_cv.stages.court_detection import CourtDetectionStage
from padel_cv.stages.gt_court import GroundTruthCourtStage
from padel_cv.stages.identity import PlayerIdentityStage
from padel_cv.stages.pose import PlayerPoseStage

__all__ = [
    "CourtDetectionStage",
    "GroundTruthCourtStage",
    "PlayerIdentityStage",
    "PlayerPoseStage",
]
