"""CLI: run a trained ball detector over a video (validation + pre-labeling).

    padel-ball-infer VIDEO --model CKPT [-o out.mp4] [--cvat pre.json] [--max-frames N]

Writes a verification mp4 (ball circled) and/or a CVAT COCO json of proposals.
Standalone: no court/homography needed, so it runs on any court.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from padel_ml.ball_export import export_cvat_coco, write_verification_video
from padel_ml.ball_infer import detect_ball_in_video


def main() -> None:
    parser = argparse.ArgumentParser(prog="padel-ball-infer", description=__doc__)
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("--model", type=Path, required=True, help="TrackNet checkpoint")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Verification mp4")
    parser.add_argument("--cvat", type=Path, default=None, help="CVAT COCO json of proposals")
    parser.add_argument("--conf", type=float, default=0.5, help="Min confidence")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if args.max_frames is not None:
        n_frames = min(n_frames, args.max_frames)

    if args.output is not None:
        hits = write_verification_video(
            args.video, args.model, args.output, args.conf, args.max_frames
        )
        print(f"Wrote {args.output} ({len(hits)} ball detections)")
    else:
        hits = detect_ball_in_video(args.video, args.model, args.conf, args.max_frames)
        print(f"{len(hits)} ball detections in {n_frames} frames")

    if args.cvat is not None:
        export_cvat_coco(hits, args.video, args.cvat, n_frames, size)
        print(f"Wrote CVAT proposals to {args.cvat}")

    detected = len(hits)
    print(f"detection rate: {100 * detected / max(n_frames, 1):.0f}% of frames")


if __name__ == "__main__":
    main()
