"""Runtime configuration, read from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PADEL_", env_file=".env", extra="ignore")

    redis_url: RedisDsn = RedisDsn("redis://localhost:6379")
    storage_dir: Path = Path("storage")
    court_model: Path | None = None
    """Trained court detector; falls back to GT-free processing when unset."""
    shot_model: Path | None = None
    """Trained PoseConv3D shot classifier; falls back to the wrist-speed dummy."""
    max_upload_mb: int = 8192

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.storage_dir / "results"

    @property
    def data_dir(self) -> Path:
        """Structured match data (JSON/CSV) — the queryable product (ADR-0010)."""
        return self.storage_dir / "data"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
