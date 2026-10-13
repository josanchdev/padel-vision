import numpy as np
import torch
from padel_ml.thesis.heatmap_figure import (
    CROP_PX,
    INPUT_SIZE,
    ball_heatmap,
    crop_origin,
    decimal,
    peak_cell,
    window_origin,
)


def test_the_peak_is_the_highest_cell_as_column_and_row() -> None:
    heatmap = np.zeros((6, 8), np.float32)
    heatmap[4, 2] = 0.9
    heatmap[1, 6] = 0.6
    assert peak_cell(heatmap) == (2, 4)


def test_the_window_is_centred_on_the_peak_and_kept_inside_the_map() -> None:
    heatmap = np.zeros((20, 30), np.float32)
    assert window_origin(heatmap, (15, 10), size=9) == (11, 6)
    assert window_origin(heatmap, (1, 1), size=9) == (0, 0)
    assert window_origin(heatmap, (29, 19), size=9) == (21, 11)


def test_the_crop_is_centred_on_the_ball_and_kept_inside_the_frame() -> None:
    width, height = CROP_PX
    assert crop_origin((1080, 1920, 3), (960.0, 540.0)) == (960 - width // 2, 540 - height // 2)
    assert crop_origin((1080, 1920, 3), (10.0, 10.0)) == (0, 0)
    assert crop_origin((1080, 1920, 3), (1910.0, 1070.0)) == (1920 - width, 1080 - height)


def test_values_are_written_with_a_decimal_comma() -> None:
    assert decimal(0.9712) == "0,97"


def test_the_network_receives_three_frames_stacked_at_the_system_size() -> None:
    seen: list[tuple[int, ...]] = []

    def network(batch: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(batch.shape))
        return torch.zeros(1, 1, INPUT_SIZE[1] // 4, INPUT_SIZE[0] // 4)

    frames = [np.zeros((1080, 1920, 3), np.uint8) for _ in range(3)]
    heatmap = ball_heatmap(network, frames)
    assert seen == [(1, 9, INPUT_SIZE[1], INPUT_SIZE[0])]
    assert heatmap.shape == (INPUT_SIZE[1] // 4, INPUT_SIZE[0] // 4)
