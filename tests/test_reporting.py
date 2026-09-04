from __future__ import annotations

import io

from yt2mp3.reporting import (
    PHASE_DOWNLOAD,
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


def test_plain_reporter_exact_lines_and_content() -> None:
    """PlainReporter must emit exactly one line per state change."""
    stream = io.StringIO()
    reporter = PlainReporter(stream)
    reporter.batch_started(1)
    reporter.track_started("track1", "Track One")
    reporter.track_phase("track1", PHASE_DOWNLOAD)
    reporter.track_finished("track1", "done")
    reporter.close()

    lines = stream.getvalue().splitlines()
    # Four state changes: batch_started, track_started, track_phase, track_finished
    assert len(lines) == 4
    assert "queued 1 track(s)" in lines[0]
    assert "start  Track One" in lines[1]
    assert PHASE_DOWNLOAD in lines[2]
    assert "Track One" in lines[2]
    assert "done" in lines[3]
    assert "Track One" in lines[3]


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

    rich_reporter = select_reporter(stream=FakeTty())
    try:
        assert isinstance(rich_reporter, RichReporter)
    finally:
        rich_reporter.close()


def test_both_reporters_satisfy_the_protocol() -> None:
    assert isinstance(PlainReporter(io.StringIO()), ProgressReporter)
    rich_reporter = RichReporter.for_stream(io.StringIO())
    try:
        assert isinstance(rich_reporter, ProgressReporter)
    finally:
        rich_reporter.close()


def test_rich_reporter_drives_all_methods_and_tracks_tasks() -> None:
    """RichReporter must track all tasks and update batch progress correctly."""
    stream = io.StringIO()
    rich_reporter = RichReporter.for_stream(stream)
    try:
        # Batch starts with 2 tracks
        rich_reporter.batch_started(2)
        assert rich_reporter._overall is not None
        assert rich_reporter._total == 2
        assert rich_reporter._done == 0
        assert len(rich_reporter._progress.tasks) == 1
        batch_task = rich_reporter._progress.tasks[rich_reporter._overall]
        assert batch_task.total == 2

        # First track starts
        rich_reporter.track_started("t1", "Track One")
        assert len(rich_reporter._progress.tasks) == 2
        assert rich_reporter._labels["t1"] == "Track One"

        # First track enters download phase
        rich_reporter.track_phase("t1", PHASE_DOWNLOAD)
        t1_task = rich_reporter._progress.tasks[rich_reporter._tasks["t1"]]
        assert "↓" in t1_task.fields["label"]
        assert "Track One" in t1_task.fields["label"]

        # First track progresses
        rich_reporter.track_progress("t1", 0.5)
        t1_task = rich_reporter._progress.tasks[rich_reporter._tasks["t1"]]
        assert t1_task.completed == 0.5

        # First track finishes
        rich_reporter.track_finished("t1", "done")
        assert "t1" not in rich_reporter._tasks
        assert rich_reporter._done == 1
        batch_task = rich_reporter._progress.tasks[rich_reporter._overall]
        # Batch label should reflect progress
        assert "1/2" in batch_task.fields["label"]

        # Second track starts and finishes
        rich_reporter.track_started("t2", "Track Two")
        assert len(rich_reporter._progress.tasks) == 2
        rich_reporter.track_finished("t2", "done")
        assert rich_reporter._done == 2
        batch_task = rich_reporter._progress.tasks[rich_reporter._overall]
        assert "2/2" in batch_task.fields["label"]

        # Unknown key must not advance batch counter or batch progress
        batch_task = rich_reporter._progress.tasks[rich_reporter._overall]
        batch_completed_before = batch_task.completed
        initial_done = rich_reporter._done
        rich_reporter.track_finished("unknown", "error")
        assert rich_reporter._done == initial_done
        batch_task = rich_reporter._progress.tasks[rich_reporter._overall]
        assert batch_task.completed == batch_completed_before
    finally:
        rich_reporter.close()


def test_rich_reporter_context_manager() -> None:
    """RichReporter must support context manager protocol."""
    stream = io.StringIO()
    with RichReporter.for_stream(stream) as reporter:
        assert isinstance(reporter, RichReporter)
        reporter.batch_started(1)
