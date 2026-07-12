import numpy as np

from padel_cv.pipeline import Frame, Pipeline, PipelineStage, PoseDetection


def make_frame(index: int = 0) -> Frame:
    return Frame(
        index=index,
        timestamp_s=index / 30.0,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
    )


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


def test_pipeline_applies_stages_in_order() -> None:
    counter = CountingStage()
    pipeline = Pipeline([FakePoseStage(), counter])
    frame = pipeline.process_frame(make_frame(index=3))
    assert len(frame.poses) == 1
    assert frame.poses[0].confidence == 0.9
    assert counter.seen == [3]


def test_pipeline_stages_accumulate_results() -> None:
    pipeline = Pipeline([FakePoseStage(), FakePoseStage()])
    frame = pipeline.process_frame(make_frame())
    assert len(frame.poses) == 2
