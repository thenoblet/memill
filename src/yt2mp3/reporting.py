"""Progress presentation.

The pipeline depends on the ``ProgressReporter`` protocol, never on rich. That
keeps the orchestration testable with a list-appending fake, and means a
non-interactive run degrades to plain lines instead of emitting terminal
control codes into a log file.
"""

from __future__ import annotations

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
    """A live bar per in-flight track, plus one for the batch."""

    __slots__ = ("_overall", "_progress", "_tasks")

    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._tasks: dict[str, TaskID] = {}
        self._overall: TaskID | None = None
        self._progress.start()

    @classmethod
    def for_stream(cls, stream: TextIO) -> RichReporter:
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
        self._overall = self._progress.add_task(
            "batch", total=total, label=f"0/{total} tracks"
        )

    def track_started(self, key: str, label: str) -> None:
        self._tasks[key] = self._progress.add_task(key, total=1.0, label=label)

    def track_phase(self, key: str, phase: str) -> None:
        task = self._tasks.get(key)
        if task is not None:
            marker = "↓" if phase == PHASE_DOWNLOAD else "♪"
            self._progress.update(task, completed=0.0, label=f"{marker} {phase}")

    def track_progress(self, key: str, fraction: float) -> None:
        task = self._tasks.get(key)
        if task is not None:
            self._progress.update(task, completed=fraction)

    def track_finished(self, key: str, status: str, detail: str | None = None) -> None:
        task = self._tasks.pop(key, None)
        if task is not None:
            self._progress.update(task, completed=1.0, label=status)
            self._progress.remove_task(task)
        if self._overall is not None:
            self._progress.advance(self._overall)

    def close(self) -> None:
        self._progress.stop()


def select_reporter(*, stream: TextIO, force_plain: bool = False) -> ProgressReporter:
    """Rich for a terminal, plain lines for anything being piped or logged."""
    if force_plain or not stream.isatty():
        return PlainReporter(stream)
    return RichReporter.for_stream(stream)
