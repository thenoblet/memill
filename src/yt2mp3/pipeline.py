"""Per-track state machine and the thread pool that drives it."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from yt2mp3.config import Settings
from yt2mp3.encoder import build_encode_command, run_encode
from yt2mp3.errors import Yt2Mp3Error
from yt2mp3.naming import infer_tags, output_stem, stem_budget
from yt2mp3.reporting import PHASE_DOWNLOAD, PHASE_ENCODE, ProgressReporter
from yt2mp3.source import SourceMedia, TrackRef
from yt2mp3.transfer import Archive, publish

STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


class EncodeFn(Protocol):
    """The encoder call, injected so the pipeline is testable without ffmpeg.

    Spelled out rather than left as ``Callable[..., None]``: the ellipsis
    form accepts any arguments at all, so a misspelled keyword at the one
    call site below would type-check cleanly and fail only at runtime.
    """

    def __call__(
        self,
        argv: Sequence[str],
        *,
        duration: float | None,
        on_progress: Callable[[float], None] | None = None,
    ) -> None: ...


class SupportsFetch(Protocol):
    """The slice of ``Downloader`` the pipeline actually uses."""

    def fetch(
        self,
        ref: TrackRef,
        staging: Path,
        on_progress: Callable[[float], None] | None = None,
    ) -> SourceMedia: ...


class StemRegistry:
    """Guarantees no two tracks in one batch claim the same filename.

    Titles are not unique: a playlist can repeat a track, and several
    decoration-only titles all sanitize to "untitled". Without this the
    second publish silently overwrites the first, both report done, and the
    archive ensures a resume never recovers the lost audio.
    """

    __slots__ = ("_claimed", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claimed: set[str] = set()

    def claim(self, stem: str) -> str:
        # Case-folded: the destination is an NTFS mount, where two stems
        # differing only in case are the same file.
        with self._lock:
            candidate, suffix = stem, 2
            while candidate.casefold() in self._claimed:
                candidate = f"{stem} ({suffix})"
                suffix += 1
            self._claimed.add(candidate.casefold())
            return candidate


@dataclass(frozen=True, slots=True)
class TrackOutcome:
    ref: TrackRef
    status: str
    path: Path | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BatchResult:
    outcomes: tuple[TrackOutcome, ...] = ()

    @property
    def failed(self) -> tuple[TrackOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status == STATUS_FAILED)

    @property
    def completed(self) -> tuple[TrackOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status == STATUS_DONE)

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


_TRAVERSAL = frozenset({".", ".."})


@contextmanager
def staging_dir(root: Path, key: str) -> Iterator[Path]:
    """A per-track scratch directory on ext4, removed however we leave it.

    The key is refused rather than rewritten if it could name anything but a
    direct child of ``root``. This directory is handed to ``shutil.rmtree``,
    and yt-dlp covers 1800+ extractors, so "the id is always safe" is an
    assumption about someone else's code guarding an irreversible delete.
    """
    if not key or "/" in key or "\\" in key or key in _TRAVERSAL:
        raise Yt2Mp3Error(f"unsafe staging key: {key!r}")
    path = root / key
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _duration_of(ref: TrackRef, info: Mapping[str, Any]) -> float | None:
    if ref.duration:
        return ref.duration
    value = info.get("duration")
    return float(value) if isinstance(value, (int, float)) else None


def process_track(
    ref: TrackRef,
    *,
    settings: Settings,
    downloader: SupportsFetch,
    reporter: ProgressReporter,
    archive: Archive,
    encode: EncodeFn = run_encode,
    stems: StemRegistry | None = None,
) -> TrackOutcome:
    """Download, encode, publish and record one track.

    Every deliberate failure is caught and returned as an outcome: one bad track
    in a hundred-track playlist must not take the other ninety-nine with it.
    """
    # Announced before the archive check, not after: a reporter that has
    # never heard of a key cannot advance the batch bar for it, so a fully
    # archived resume would sit at 0/N for its entire run.
    reporter.track_started(ref.video_id, ref.title)

    if ref.video_id in archive:
        reporter.track_finished(ref.video_id, STATUS_SKIPPED, "already in library")
        return TrackOutcome(ref, STATUS_SKIPPED)

    # A lone track cannot collide with anything, so a direct caller that has
    # no batch to share gets a private registry rather than an error.
    registry = stems if stems is not None else StemRegistry()

    def report(fraction: float) -> None:
        reporter.track_progress(ref.video_id, fraction)

    try:
        with staging_dir(settings.staging_root, ref.video_id) as staging:
            reporter.track_phase(ref.video_id, PHASE_DOWNLOAD)
            media = downloader.fetch(ref, staging, report)

            tags = infer_tags(media.info, clean=settings.clean_titles)
            stem = registry.claim(
                output_stem(tags, max_length=stem_budget(settings.destination))
            )
            encoded = staging / f"{stem}.mp3"

            reporter.track_phase(ref.video_id, PHASE_ENCODE)
            encode(
                build_encode_command(
                    audio=media.audio,
                    output=encoded,
                    quality=settings.quality,
                    tags=tags,
                    cover=media.cover,
                    normalize=settings.normalize,
                ),
                duration=_duration_of(ref, media.info),
                on_progress=report,
            )

            published = publish(encoded, settings.destination, encoded.name)
            if settings.keep_source:
                publish(
                    media.audio, settings.destination, f"{stem}{media.audio.suffix}"
                )

        # Only now is the track genuinely on disk under its final name.
        archive.add(ref.video_id)
        reporter.track_finished(ref.video_id, STATUS_DONE)
        return TrackOutcome(ref, STATUS_DONE, path=published)
    except Yt2Mp3Error as exc:
        reporter.track_finished(ref.video_id, STATUS_FAILED, str(exc))
        return TrackOutcome(ref, STATUS_FAILED, error=str(exc))


def run_batch(
    refs: Sequence[TrackRef],
    *,
    settings: Settings,
    downloader: SupportsFetch,
    reporter: ProgressReporter,
    archive: Archive,
    encode: EncodeFn = run_encode,
) -> BatchResult:
    """Process every ref, at most ``settings.jobs`` at a time."""
    reporter.batch_started(len(refs))
    if not refs:
        return BatchResult()

    results: list[TrackOutcome | None] = [None] * len(refs)
    stems = StemRegistry()
    with ThreadPoolExecutor(max_workers=settings.jobs) as pool:
        futures = {
            pool.submit(
                process_track,
                ref,
                settings=settings,
                downloader=downloader,
                reporter=reporter,
                archive=archive,
                encode=encode,
                stems=stems,
            ): index
            for index, ref in enumerate(refs)
        }
        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        except BaseException:
            # Only a run-level abort gets here -- a cancelled download, an
            # unreadable cookie store, Ctrl-C -- because process_track
            # returns its own failures. Every ref was submitted up front, so
            # without this the remaining 199 tracks of a 200-track playlist
            # would still download before anyone saw the exception.
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    # Input order, not completion order: the summary should read like the queue.
    return BatchResult(tuple(outcome for outcome in results if outcome is not None))
