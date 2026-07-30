"""The resume checkpoint must restore training state so a WSL crash mid-run
costs at most one epoch, not the whole ~10h job."""

from pathlib import Path

import torch
from padel_ml.ball_train import _load_resume, _save_resume
from padel_ml.tracknet import TrackNetV2


def _fresh() -> tuple[TrackNetV2, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    model = TrackNetV2()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    return model, opt, sched


def test_resume_roundtrip_restores_epoch_and_best(tmp_path: Path) -> None:
    model, opt, sched = _fresh()
    # Simulate having trained through epoch 7: step the scheduler and mutate weights.
    for _ in range(7):
        sched.step()
    with torch.no_grad():
        next(iter(model.parameters())).add_(1.0)  # make weights distinct
    history = {"f1": [0.1] * 7, "precision": [0.2] * 7, "recall": [0.3] * 7}
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    ckpt = tmp_path / "m.pt.ckpt"
    _save_resume(ckpt, 7, model, opt, sched, history, best_f1=0.55, best_state=best_state)
    assert ckpt.exists()
    assert not ckpt.with_suffix(ckpt.suffix + ".tmp").exists()  # temp cleaned up

    # A fresh run loads it and must continue from epoch 8 with the same state.
    model2, opt2, sched2 = _fresh()
    hist2: dict[str, list[float]] = {"f1": [], "precision": [], "recall": []}
    start_epoch, best_f1, loaded_best = _load_resume(ckpt, model2, opt2, sched2, hist2, "cpu")

    assert start_epoch == 8
    assert best_f1 == 0.55
    assert hist2["f1"] == [0.1] * 7
    # Model + scheduler state actually restored.
    p1 = next(iter(model.parameters()))
    p2 = next(iter(model2.parameters()))
    assert torch.allclose(p1, p2)
    assert sched2.last_epoch == sched.last_epoch
    assert set(loaded_best.keys()) == set(model.state_dict().keys())


def test_save_resume_is_atomic_overwrite(tmp_path: Path) -> None:
    model, opt, sched = _fresh()
    ckpt = tmp_path / "m.pt.ckpt"
    hist: dict[str, list[float]] = {"f1": [], "precision": [], "recall": []}
    _save_resume(ckpt, 1, model, opt, sched, hist, 0.1, model.state_dict())
    _save_resume(ckpt, 2, model, opt, sched, hist, 0.2, model.state_dict())  # overwrite
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["epoch"] == 2  # second write won, no leftover temp
    assert not ckpt.with_suffix(ckpt.suffix + ".tmp").exists()
