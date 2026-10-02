from pathlib import Path

from padel_cv.pose import DEFAULT_TRACKER


def test_default_tracker_config_exists() -> None:
    assert Path(DEFAULT_TRACKER).is_file()
