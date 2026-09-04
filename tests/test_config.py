from __future__ import annotations

from pathlib import Path

import pytest

from memill.config import (
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
    [(None, (4, 2)), (1, (1, 8)), (2, (2, 4)), (8, (8, 1)), (16, (16, 1))],
)
def test_concurrency_keeps_the_socket_budget(
    jobs: int | None, expected: tuple[int, int]
) -> None:
    assert plan_concurrency(jobs, cpu_count=8) == expected


def test_concurrency_rejects_a_non_positive_job_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        plan_concurrency(0, cpu_count=8)


@pytest.mark.parametrize("jobs", range(1, 17))
def test_concurrency_never_multiplies_beyond_the_budget(jobs: int) -> None:
    resolved, fragments = plan_concurrency(jobs, cpu_count=8)
    assert resolved == jobs
    # At or under the budget the product must fit inside it; beyond the budget
    # the user's explicit -j is the ceiling, never a multiple of it.
    assert resolved * fragments <= max(8, jobs)


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


@pytest.mark.parametrize("bitrate", ["512k", "0k", "7k", "321k", "9999k"])
def test_cbr_rejects_a_rate_libmp3lame_cannot_encode(bitrate: str) -> None:
    """MP3 is 8-320 kbps. Anything else matches the shape and fails in ffmpeg.

    Left to the encoder, every track in the run downloads in full first, so the
    range has to be refused where it costs a usage message instead.
    """
    with pytest.raises(ValueError, match="bitrate must be between 8k and 320k"):
        CbrQuality(bitrate)


@pytest.mark.parametrize("bitrate", ["8k", "32k", "128k", "320k"])
def test_cbr_accepts_the_range_libmp3lame_does_encode(bitrate: str) -> None:
    assert CbrQuality(bitrate).ffmpeg_args() == ("-b:a", bitrate)
