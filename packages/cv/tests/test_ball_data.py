import json

import numpy as np

from padel_cv.ball_data import (
    BALL_CATEGORY_ID,
    load_ball_centers,
    render_heatmap,
)


def _write_coco(path, annotations, images) -> None:
    path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "Ball"}, {"id": 3, "name": "shot-event"}],
                "images": images,
                "annotations": annotations,
            }
        )
    )


def test_load_ball_centers_reads_fractional_box_center(tmp_path) -> None:
    path = tmp_path / "ball.json"
    _write_coco(
        path,
        images=[
            {"id": 1, "file_name": "frame_000000.PNG", "width": 1920, "height": 1080},
            {"id": 2, "file_name": "frame_000001.PNG", "width": 1920, "height": 1080},
        ],
        annotations=[
            # box center (104, 204) -> fractional (104/1920, 204/1080)
            {
                "id": 1,
                "image_id": 1,
                "category_id": BALL_CATEGORY_ID,
                "bbox": [100.0, 200.0, 8.0, 8.0],
                "attributes": {"occluded": False},
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": BALL_CATEGORY_ID,
                "bbox": [300.0, 400.0, 10.0, 10.0],
                "attributes": {"occluded": True},
            },
        ],
    )
    centers = load_ball_centers(path)
    assert centers[0].center_xy == (104.0 / 1920, 204.0 / 1080)
    assert centers[0].occluded is False
    assert centers[1].center_xy == (305.0 / 1920, 405.0 / 1080)
    assert centers[1].occluded is True


def test_load_ball_centers_ignores_non_ball_categories(tmp_path) -> None:
    path = tmp_path / "ball.json"
    _write_coco(
        path,
        images=[{"id": 1, "file_name": "frame_000000.PNG", "width": 1920, "height": 1080}],
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,  # shot-event, not a ball
                "bbox": [0.0, 0.0, 4.0, 4.0],
                "attributes": {},
            }
        ],
    )
    assert load_ball_centers(path) == {}


def test_render_heatmap_peaks_at_scaled_center() -> None:
    # Ball at frame center (0.5, 0.5) -> grid 48x27, peak at grid center.
    hm = render_heatmap((0.5, 0.5), grid_wh=(48, 27), sigma=2.0)
    peak_y, peak_x = np.unravel_index(int(hm.argmax()), hm.shape)
    assert (peak_x, peak_y) == (24, 13)  # 0.5*48=24, 0.5*27=13.5 -> 13
    # Peak is near 1: the true center (24.0, 13.5) falls between cells, so the
    # nearest cell is just under the Gaussian's 1.0 maximum.
    assert hm.max() > 0.95


def test_render_heatmap_none_is_all_zero() -> None:
    hm = render_heatmap(None, grid_wh=(48, 27))
    assert hm.shape == (27, 48)
    assert not hm.any()
