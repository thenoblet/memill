from __future__ import annotations

from pathlib import Path

import pytest

from yt2mp3.config import (
    CbrQuality,
    Settings,
    VbrQuality,
    plan_concurrency,
)


def test_vbr_emits_lame_quality_flag() -> None:
    assert VbrQuality(0).ffmpeg_args() == ("-q:a", "0")


def test_cbr_emits_bitrate_flag() -> None:
    assert CbrQuality("320k").ffmpeg_args() == ("-b:a", "320k")


@pytest.mark.parametrize("level", [-1, 10])
def test_vbr_rejects_levels_outside_lame_range(level: int) -> None:
    with pytest.raises(ValueError, match=r"0\.\.9"):
        VbrQuality(level)


def test_cbr_rejects_a_bitrate_that_is_not_a_rate() -> None:
    with pytest.raises(ValueError, match="bitrate"):
        CbrQuality("very loud")


@pytest.mark.parametrize(
    ("jobs", "expected"),
    [(None, (4, 2)), (1, (1, 8)), (2, (2, 4)), (8, (8, 2))],
)
def test_concurrency_keeps_the_socket_budget(
    jobs: int | None, expected: tuple[int, int]
) -> None:
    assert plan_concurrency(jobs, cpu_count=8) == expected


def test_concurrency_rejects_a_non_positive_job_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        plan_concurrency(0, cpu_count=8)


def test_settings_are_immutable() -> None:
    settings = Settings(
        destination=Path("/tmp/out"),
        staging_root=Path("/tmp/stage"),
        quality=VbrQuality(0),
        jobs=4,
        fragments=2,
    )
    with pytest.raises(AttributeError):
        settings.jobs = 9  # type: ignore[misc]
