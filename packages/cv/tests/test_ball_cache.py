import json

import cv2
import numpy as np

from padel_cv.ball_cache import (
    FRAME_H,
    FRAME_W,
    consolidate_to_memmap,
    extract_ball_frames_to_cache,
)


def _make_video(path, n_frames, size=(1920, 1080)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_ball_json(path, centers_by_frame) -> None:
    images, annotations = [], []
    for idx, center in centers_by_frame.items():
        images.append(
            {"id": idx + 1, "file_name": f"frame_{idx:06d}.PNG", "width": 1920, "height": 1080}
        )
        if center is not None:
            cx, cy = center
            annotations.append(
                {
                    "id": idx + 1,
                    "image_id": idx + 1,
                    "category_id": 1,
                    "bbox": [cx - 4, cy - 4, 8.0, 8.0],
                    "attributes": {"occluded": False},
                }
            )
    coco = {
        "categories": [{"id": 1, "name": "Ball"}],
        "images": images,
        "annotations": annotations,
    }
    path.write_text(json.dumps(coco))


def test_extract_caches_downscaled_frames_and_centers(tmp_path) -> None:
    video = tmp_path / "match.mp4"
    _make_video(video, n_frames=10)
    ball_json = tmp_path / "ball.json"
    # Ball present on frames 0 and 2, absent on 1.
    _make_ball_json(ball_json, {0: (960.0, 540.0), 1: None, 2: (100.0, 200.0)})
    cache = tmp_path / "cache"

    written = extract_ball_frames_to_cache(video, ball_json, cache, shard_size=4)
    assert written == 10

    shards = sorted(cache.glob("frames_*.npz"))
    assert len(shards) == 3  # 4 + 4 + 2

    first = np.load(shards[0])
    assert first["frames"].shape == (4, FRAME_H, FRAME_W, 3)
    # Centres are fractional: (960/1920, 540/1080) = (0.5, 0.5).
    np.testing.assert_allclose(first["centers"][0], [0.5, 0.5], rtol=1e-6)
    assert np.isnan(first["centers"][1]).all()  # frame 1 has no ball
    np.testing.assert_allclose(first["centers"][2], [100.0 / 1920, 200.0 / 1080], rtol=1e-6)
    # occluded array is present, False where there is no ball annotation.
    assert first["occluded"].dtype == np.bool_
    assert first["occluded"][1] == False  # noqa: E712 - explicit bool check


def test_extract_resumes_from_existing_shards(tmp_path) -> None:
    video = tmp_path / "match.mp4"
    _make_video(video, n_frames=10)
    ball_json = tmp_path / "ball.json"
    _make_ball_json(ball_json, {i: (10.0 * i, 20.0 * i) for i in range(10)})
    cache = tmp_path / "cache"

    # First pass stops after 4 frames (one shard).
    extract_ball_frames_to_cache(video, ball_json, cache, shard_size=4, max_frames=4)
    assert len(list(cache.glob("frames_*.npz"))) == 1

    # Second pass resumes and completes the rest without re-doing frame 0.
    written = extract_ball_frames_to_cache(video, ball_json, cache, shard_size=4)
    assert written == 6  # frames 4..9

    all_indices = sorted(int(i) for s in cache.glob("frames_*.npz") for i in np.load(s)["indices"])
    assert all_indices == list(range(10))


def test_consolidate_to_memmap_orders_frames(tmp_path) -> None:
    video = tmp_path / "match.mp4"
    _make_video(video, n_frames=10)
    ball_json = tmp_path / "ball.json"
    _make_ball_json(ball_json, {i: (100.0 + i, 200.0) for i in range(10)})
    cache = tmp_path / "cache"
    extract_ball_frames_to_cache(video, ball_json, cache, shard_size=4)  # 3 shards

    dat, meta = consolidate_to_memmap(cache)
    assert dat.exists() and meta.exists()
    m = np.load(meta)
    assert int(m["n"]) == 10
    # Frame indices are ascending and complete.
    assert list(m["indices"]) == list(range(10))
    # The memmap has the right shape and is readable per-frame.
    frames = np.memmap(dat, dtype=np.uint8, mode="r", shape=(10, FRAME_H, FRAME_W, 3))
    assert frames.shape == (10, FRAME_H, FRAME_W, 3)
    assert frames[5].max() >= 0  # a single frame reads without loading all
