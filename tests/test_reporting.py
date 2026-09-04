from __future__ import annotations

import io

from yt2mp3.reporting import (
    PHASE_ENCODE,
    PlainReporter,
    ProgressReporter,
    RichReporter,
    select_reporter,
)


def test_plain_reporter_logs_one_line_per_state_change() -> None:
    stream = io.StringIO()
    reporter = PlainReporter(stream)
    reporter.batch_started(2)
    reporter.track_started("a", "Some Mix")
    reporter.track_phase("a", PHASE_ENCODE)
    reporter.track_finished("a", "done")
    reporter.close()

    lines = stream.getvalue().splitlines()
    assert any("2" in line for line in lines)
    assert any("Some Mix" in line for line in lines)
    assert any(PHASE_ENCODE in line for line in lines)
    assert any("done" in line for line in lines)


def test_plain_reporter_does_not_emit_a_line_per_progress_tick() -> None:
    stream = io.StringIO()
    reporter = PlainReporter(stream)
    reporter.track_started("a", "Mix")
    before = len(stream.getvalue().splitlines())
    for step in range(100):
        reporter.track_progress("a", step / 100)
    assert len(stream.getvalue().splitlines()) == before


def test_a_non_tty_stream_selects_the_plain_reporter() -> None:
    assert isinstance(select_reporter(stream=io.StringIO()), PlainReporter)


def test_force_plain_overrides_a_tty() -> None:
    class FakeTty(io.StringIO):
        def isatty(self) -> bool:
            return True

    result = select_reporter(stream=FakeTty(), force_plain=True)
    assert isinstance(result, PlainReporter)
    assert isinstance(select_reporter(stream=FakeTty()), RichReporter)


def test_both_reporters_satisfy_the_protocol() -> None:
    assert isinstance(PlainReporter(io.StringIO()), ProgressReporter)
    rich_reporter = RichReporter.for_stream(io.StringIO())
    try:
        assert isinstance(rich_reporter, ProgressReporter)
    finally:
        rich_reporter.close()
