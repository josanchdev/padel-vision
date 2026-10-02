from pathlib import Path

import numpy as np

from padel_cv.pipeline import Frame, PipelineStage, PoseDetection
from padel_cv.stages.pose import DEFAULT_TRACKER


def test_default_tracker_config_exists() -> None:
    assert Path(DEFAULT_TRACKER).is_file()


class FakePoseStage:
    """Adds one fixed detection, standing in for a real model stage."""

    def process(self, frame: Frame) -> Frame:
        frame.poses.append(
            PoseDetection(
                bbox_xyxy=(1.0, 1.0, 10.0, 10.0),
                confidence=0.9,
                keypoints=np.zeros((17, 3), dtype=np.float32),
            )
        )
        return frame


class CountingStage:
    def __init__(self) -> None:
        self.seen: list[int] = []

    def process(self, frame: Frame) -> Frame:
        self.seen.append(frame.index)
        return frame


def test_stages_satisfy_protocol() -> None:
    assert isinstance(FakePoseStage(), PipelineStage)
    assert isinstance(CountingStage(), PipelineStage)
