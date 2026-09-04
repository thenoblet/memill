"""Progress presentation.

The pipeline depends on the ``ProgressReporter`` protocol, never on rich. That
keeps the orchestration testable with a list-appending fake, and means a
non-interactive run degrades to plain lines instead of emitting terminal
control codes into a log file.

Warning: RichReporter replaces sys.stdout and sys.stderr process-wide while
running. Use the context manager or call close() to restore them.
"""

from __future__ import annotations

import threading
from typing import Protocol, TextIO, runtime_checkable

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

PHASE_DOWNLOAD = "downloading"
PHASE_ENCODE = "encoding"


@runtime_checkable
class ProgressReporter(Protocol):
    """What the pipeline is allowed to say about its progress."""

    def batch_started(self, total: int) -> None: ...
    def track_started(self, key: str, label: str) -> None: ...
    def track_phase(self, key: str, phase: str) -> None: ...
    def track_progress(self, key: str, fraction: float) -> None: ...
    def track_finished(
        self, key: str, status: str, detail: str | None = None
    ) -> None: ...
    def close(self) -> None: ...


class PlainReporter:
    """One line per state change. Safe to redirect into a file."""

    __slots__ = ("_labels", "_stream")

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._labels: dict[str, str] = {}

    def __enter__(self) -> PlainReporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _write(self, message: str) -> None:
        self._stream.write(f"{message}\n")
        self._stream.flush()

    def batch_started(self, total: int) -> None:
        self._write(f"queued {total} track(s)")

    def track_started(self, key: str, label: str) -> None:
        self._labels[key] = label
        self._write(f"start  {label}")

    def track_phase(self, key: str, phase: str) -> None:
        self._write(f"{phase:<12} {self._labels.get(key, key)}")

    def track_progress(self, key: str, fraction: float) -> None:
        """Deliberately silent: a line per tick would be thousands of lines."""

    def track_finished(self, key: str, status: str, detail: str | None = None) -> None:
        suffix = f" - {detail}" if detail else ""
        self._write(f"{status:<12} {self._labels.get(key, key)}{suffix}")

    def close(self) -> None:
        self._stream.flush()


class RichReporter:
    """A live bar per in-flight track, plus one for the batch.

    Warning: this reporter replaces sys.stdout and sys.stderr process-wide.
    Always use the context manager or call close() to restore them.
    """

    __slots__ = (
        "_done",
        "_labels",
        "_lock",
        "_overall",
        "_progress",
        "_tasks",
        "_total",
    )

    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskID] = {}
        self._labels: dict[str, str] = {}
        self._overall: TaskID | None = None
        self._total = 0
        self._done = 0
        self._progress.start()

    def __enter__(self) -> RichReporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @classmethod
    def for_stream(cls, stream: TextIO) -> RichReporter:
        """Create a reporter for a stream.

        Warning: the returned reporter replaces sys.stdout and sys.stderr
        process-wide. Use it as a context manager or call close() when done
        to restore them.
        """
        return cls(
            Progress(
                TextColumn("[bold blue]{task.fields[label]}", justify="left"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=Console(file=stream),
                transient=False,
            )
        )

    def batch_started(self, total: int) -> None:
        self._total = total
        self._overall = self._progress.add_task(
            "batch", total=total, label=f"0/{total} tracks"
        )

    def track_started(self, key: str, label: str) -> None:
        self._labels[key] = label
        self._tasks[key] = self._progress.add_task(key, total=1.0, label=label)

    def track_phase(self, key: str, phase: str) -> None:
        task = self._tasks.get(key)
        if task is not None:
            marker = "↓" if phase == PHASE_DOWNLOAD else "♪"
            # Keep the title. A bar reading only "downloading" never says which
            # track is downloading, which is the one thing the user wants.
            title = self._labels.get(key, key)
            self._progress.update(task, completed=0.0, label=f"{marker} {title}")

    def track_progress(self, key: str, fraction: float) -> None:
        task = self._tasks.get(key)
        if task is not None:
            self._progress.update(task, completed=fraction)

    def track_finished(self, key: str, status: str, detail: str | None = None) -> None:
        task = self._tasks.pop(key, None)
        if task is None:
            # Unknown or duplicate key: advancing here would over-count the batch.
            return
        self._labels.pop(key, None)
        self._progress.remove_task(task)
        # `+= 1` is a read-modify-write, and the pool calls this from several
        # threads at once: two finishes can read the same value and one write
        # is lost, so a 40-track run ends its label reading "38/40 tracks".
        # rich's own Progress is lock-guarded, which is why the bar stays
        # right and only this counter is wrong -- and why the update is held
        # inside the lock too, so the label cannot go backwards when two
        # threads reach it out of order.
        with self._lock:
            self._done += 1
            if self._overall is not None:
                self._progress.update(
                    self._overall, advance=1, label=f"{self._done}/{self._total} tracks"
                )

    def close(self) -> None:
        self._progress.stop()


def select_reporter(*, stream: TextIO, force_plain: bool = False) -> ProgressReporter:
    """Rich for a terminal, plain lines for anything being piped or logged."""
    if force_plain or not stream.isatty():
        return PlainReporter(stream)
    return RichReporter.for_stream(stream)
