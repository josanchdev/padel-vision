"""Pipeline stages."""

from padel_cv.stages.gt_court import GroundTruthCourtStage
from padel_cv.stages.pose import PlayerPoseStage

__all__ = ["GroundTruthCourtStage", "PlayerPoseStage"]
