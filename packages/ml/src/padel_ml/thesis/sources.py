"""Read-only access to the evidence JSON of `docs/metrics/`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from padel_cv.paths import METRICS


class Evidence:
    """`evidence("chain_cv")` returns that file parsed, read once."""

    def __init__(self, directory: Path = METRICS) -> None:
        self.directory = directory
        self._cache: dict[str, Any] = {}

    def __call__(self, name: str) -> Any:
        if name not in self._cache:
            self._cache[name] = json.loads((self.directory / f"{name}.json").read_text())
        return self._cache[name]

    def interval(self, key: str) -> tuple[float, float, float]:
        """(value, low, high) of a 95 % interval in confidence_intervals.json."""
        metrics = self("confidence_intervals")["metrics"]
        return (
            float(metrics[f"{key}.valor"]),
            float(metrics[f"{key}.bajo"]),
            float(metrics[f"{key}.alto"]),
        )
