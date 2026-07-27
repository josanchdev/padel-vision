"""Domain models shared by the API and the worker."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MatchStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


def _now() -> datetime:
    return datetime.now(UTC)


class Match(BaseModel):
    """A submitted video and the state of its analysis job."""

    id: str
    filename: str
    status: MatchStatus = MatchStatus.PENDING
    progress: float = 0.0  # 0..1 fraction of frames processed
    shots_detected: int = 0
    duration_s: float | None = None  # analysed video length, for the dashboard card
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def touch(self) -> None:
        self.updated_at = _now()
