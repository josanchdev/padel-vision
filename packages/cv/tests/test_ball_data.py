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


def test_load_ball_centers_reads_box_center(tmp_path) -> None:
    path = tmp_path / "ball.json"
    _write_coco(
        path,
        images=[
            {"id": 1, "file_name": "frame_000000.PNG"},
            {"id": 2, "file_name": "frame_000001.PNG"},
        ],
        annotations=[
            # box [x, y, w, h] -> center (x + w/2, y + h/2) = (104, 204)
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
    assert centers[0].center_xy == (104.0, 204.0)
    assert centers[0].occluded is False
    assert centers[1].center_xy == (305.0, 405.0)
    assert centers[1].occluded is True


def test_load_ball_centers_ignores_non_ball_categories(tmp_path) -> None:
    path = tmp_path / "ball.json"
    _write_coco(
        path,
        images=[{"id": 1, "file_name": "frame_000000.PNG"}],
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
    # Frame 1920x1080 ball at its center -> grid 48x27, peak at grid center.
    hm = render_heatmap((960.0, 540.0), frame_wh=(1920, 1080), grid_wh=(48, 27), sigma=2.0)
    peak_y, peak_x = np.unravel_index(int(hm.argmax()), hm.shape)
    assert (peak_x, peak_y) == (24, 13)  # 960*48/1920=24, 540*27/1080=13.5 -> 13
    # Peak is near 1: the true center (24.0, 13.5) falls between cells, so the
    # nearest cell is just under the Gaussian's 1.0 maximum.
    assert hm.max() > 0.95


def test_render_heatmap_none_is_all_zero() -> None:
    hm = render_heatmap(None, frame_wh=(1920, 1080), grid_wh=(48, 27))
    assert hm.shape == (27, 48)
    assert not hm.any()
