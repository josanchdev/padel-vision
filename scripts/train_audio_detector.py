"""Train the production audio hit detector on ALL of CVSPORTS (ADR-0015 A).

The evidence (`evidence_audio_detector.py`) holds rallies back to score the
detector, and the whole-chain evaluation (`evaluate_chain_cv.py`) retrains it
without each tournament in turn; every one of those models is thrown away. This
trains the one that ships: all 99 rallies, nothing held back, with the same
function as every evaluation fold (`fit_and_save`), so the model that runs is
trained exactly like the models that were measured.

    uv run python scripts/train_audio_detector.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from padel_ml.audio_train import fit_and_save

REPO = Path(__file__).resolve().parents[1]
CVSPORTS = REPO / "data" / "raw" / "padel_audio_dataset" / "CVSPORTS_Padel"
CACHE = REPO / "data" / "datasets" / "audio_cache.npz"
OUT = REPO / "runs" / "audio" / "audio_crnn.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rallies = sorted(CVSPORTS.glob("rallies/*.mp4"))
    print(f"{len(rallies)} rallies de CVSPORTS · dispositivo {device}")

    started = time.perf_counter()
    fit_and_save(CVSPORTS, args.out, cache=CACHE, seed=args.seed, device=device)
    print(f"entrenado en {time.perf_counter() - started:.0f}s")
    print(f"guardado -> {args.out}")


if __name__ == "__main__":
    main()
