import json

import cv2
import numpy as np
import torch
from padel_ml.ball_export import export_cvat_coco
from padel_ml.ball_infer import BallDetector, BallHit, detect_ball_in_video
from padel_ml.tracknet import TrackNetV2


def _save_ckpt(path) -> None:
    model = TrackNetV2()
    torch.save({"state_dict": model.state_dict(), "model_name": "tracknetv2"}, path)


def _make_video(path, n_frames, size=(640, 360)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i % 255, dtype=np.uint8))
    writer.release()


def test_detector_needs_full_buffer_before_detecting(tmp_path) -> None:
    ckpt = tmp_path / "ball.pt"
    _save_ckpt(ckpt)
    det = BallDetector(ckpt, min_confidence=0.0)  # accept any peak
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    assert det.detect(img, 0) is None  # buffer not full
    assert det.detect(img, 1) is None
    hit = det.detect(img, 2)  # 3rd frame completes the window
    assert hit is not None
    assert 0 <= hit.x_px <= 640 and 0 <= hit.y_px <= 360


def test_detect_over_video_returns_hits(tmp_path) -> None:
    ckpt = tmp_path / "ball.pt"
    _save_ckpt(ckpt)
    video = tmp_path / "clip.mp4"
    _make_video(video, n_frames=10)
    hits = detect_ball_in_video(video, ckpt, min_confidence=0.0)
    # First 2 frames have no full window; up to 8 detections possible.
    assert all(isinstance(h, BallHit) for h in hits)
    assert all(0 <= h.frame_index < 10 for h in hits)


def test_cvat_export_is_valid_coco(tmp_path) -> None:
    hits = [BallHit(frame_index=2, x_px=100.0, y_px=200.0, confidence=0.9)]
    out = tmp_path / "pre.json"
    export_cvat_coco(hits, tmp_path / "clip.mp4", out, n_frames=5, frame_size=(640, 360))
    coco = json.loads(out.read_text())
    assert coco["categories"][0]["name"] == "Ball"
    assert len(coco["images"]) == 5
    ann = coco["annotations"][0]
    assert ann["image_id"] == 3  # frame_index 2 -> image id 3 (1-based)
    # bbox centered on the detection: [x-6, y-6, 12, 12]
    assert ann["bbox"] == [94.0, 194.0, 12, 12]
