"""How long the system takes per stage, on one rally (the cost table of sistema.md).

Runs `analyze_rally` — the code the demo and the web export run — over one
CVSPORTS rally and adds up the time spent in each stage: the audio detector, the
pose model, the ball model, and everything else (decoding the video, identity,
ball clean-up, the vote, the classifier). A first pass warms the GPU up and is
not counted.

    uv run python scripts/evidence_processing_time.py [--rally 20230528_VIGO_11]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import torch
from padel_ml import rally_analysis
from padel_ml.evidence import ExperimentResult
from padel_ml.rally_analysis import ModelPaths, RallyModels, analyze_rally

from padel_cv.court_registry import court_file_for
from padel_cv.paths import CVSPORTS_RALLIES, RUNS


def timed(function: Callable[..., Any], calls: list[float]) -> Callable[..., Any]:
    """`function`, adding the duration of every call to `calls`."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = function(*args, **kwargs)
        calls.append(time.perf_counter() - start)
        return result

    return wrapper


class Timed:
    """Stands in for a detector and times one of its methods."""

    def __init__(self, inner: Any, method: str) -> None:
        self._inner, self._method, self.calls = inner, method, []

    @property
    def seconds(self) -> float:
        return sum(self.calls)

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        return timed(attribute, self.calls) if name == self._method else attribute


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rally", default="20230528_VIGO_11")
    args = parser.parse_args()

    video = CVSPORTS_RALLIES / f"{args.rally}.mp4"
    court = court_file_for(args.rally)
    models = RallyModels(ModelPaths.under(RUNS))
    analyze_rally(video, models, court)  # warm-up: CUDA kernels, model loading

    audio: list[float] = []
    rally_analysis.detect_hits_in_audio = timed(rally_analysis.detect_hits_in_audio, audio)
    pose, ball = Timed(models.pose, "detect"), Timed(models.ball, "detect")
    models.pose, models.ball = pose, ball  # type: ignore[assignment]
    start = time.perf_counter()
    analysis = analyze_rally(video, models, court)
    total = time.perf_counter() - start

    duration = analysis.n_frames / analysis.fps
    stages = {"audio": sum(audio), "pose": pose.seconds, "pelota": ball.seconds}
    stages["resto"] = total - sum(stages.values())
    for name, seconds in stages.items():
        print(f"  {name:7s} {seconds:6.1f} s  {seconds / total:6.1%}")
    print(f"  total   {total:6.1f} s  para {duration:.1f} s de vídeo ({total / duration:.2f}x)")

    result = ExperimentResult(
        name="processing_time",
        summary=(
            f"Coste de procesado: {total:.1f} s para {duration:.1f} s de vídeo "
            f"({total / duration:.2f}x tiempo real) en una {torch.cuda.get_device_name()}."
        ),
        metrics={
            "total_s": round(total, 2),
            "video_s": round(duration, 2),
            "realtime_factor": round(total / duration, 3),
            **{f"{name}_s": round(seconds, 2) for name, seconds in stages.items()},
        },
        dataset=f"{args.rally}: {analysis.n_frames} frames, {analysis.width}x{analysis.height}, "
        f"{analysis.fps:.0f} fps",
        method="analyze_rally completo, una pasada de calentamiento previa sin contar",
        params={"gpu": torch.cuda.get_device_name(), "rally": args.rally},
        notes=(
            "«resto» es decodificar el vídeo, la identidad J1-J4, la limpieza de la pelota, "
            "el voto y el clasificador. El vídeo anotado no se cuenta: es una segunda pasada "
            "opcional (rally_render)."
        ),
    )
    print(f"guardado -> {result.save()}")


if __name__ == "__main__":
    main()
