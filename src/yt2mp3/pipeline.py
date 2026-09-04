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
from yt2mp3.errors import TransferError, Yt2Mp3Error
from yt2mp3.naming import KEPT_SOURCE_EXTENSION, infer_tags, output_stem, stem_budget
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

    def claim(self, stem: str, *, max_length: int) -> str:
        """Return a stem no other track in this batch holds, within
        ``max_length``.

        The marker is not simply appended: the base is re-truncated to make
        room for it, because the caller's budget already accounts for the
        destination path and the ``.part`` suffix, and overshooting it is
        exactly the failure the budget exists to prevent.

        Case-folded: the destination is an NTFS mount, where two stems
        differing only in case are the same file.
        """
        with self._lock:
            candidate, suffix = stem, 2
            while candidate.casefold() in self._claimed:
                marker = f" ({suffix})"
                head = stem[: max(1, max_length - len(marker))].rstrip(". ")
                candidate = f"{head}{marker}"
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
    """A per-track scratch directory on ext4, removed on success only.

    Kept on failure, deliberately. yt-dlp writes a ``.part`` file here and
    resumes from it on the next attempt, and ``key`` is the video id, so the
    directory a retry is handed is the same one the failed run left behind.
    Removing it on the way out of a failure -- which a ``finally`` would --
    throws away the only thing that makes the retry cheap: a network blip at
    81% of a three-hour mix cost the whole 81%.

    The trade: every failed or interrupted track leaves a directory behind,
    roughly the size of one downloaded track each. Each is reclaimed by the
    next successful run of the same id, and all of them by ``make clean``,
    which wipes ``~/.cache/yt2mp3``.

    The key is refused rather than rewritten if it could name anything but a
    direct child of ``root``. This directory is handed to ``shutil.rmtree``,
    and yt-dlp covers 1800+ extractors, so "the id is always safe" is an
    assumption about someone else's code guarding an irreversible delete.
    """
    if not key or "/" in key or "\\" in key or key in _TRAVERSAL:
        raise Yt2Mp3Error(f"unsafe staging key: {key!r}")
    path = root / key
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # A read-only staging root, a full disk or a bad --cache path are
        # expected failures, not bugs. A bare OSError here would fly past the
        # pipeline's Yt2Mp3Error handler and abort the whole batch -- the same
        # defect Archive.add already guards against, one module over.
        raise TransferError(
            f"could not create the staging directory {path}: {exc}"
        ) from exc
    try:
        yield path
    except BaseException:
        # BaseException, not Exception: Ctrl-C mid-download is the case that
        # most wants a resumable .part file. Both clauses do the same thing
        # here -- the removal lives in the else -- but naming it says the
        # interrupted run was considered, not overlooked.
        raise
    else:
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
            # One budget covers both published files, so it has to be sized for
            # the longer suffix. --keep-source publishes "<stem><source suffix>"
            # alongside "<stem>.mp3", and YouTube audio arrives as .webm/.opus
            # -- one character more than the .mp3 the budget would otherwise
            # assume, which is enough to cross the 260-char NTFS limit.
            budget = stem_budget(
                settings.destination,
                extension=(
                    KEPT_SOURCE_EXTENSION if settings.keep_source else ".mp3"
                ),
            )
            stem = registry.claim(
                output_stem(tags, max_length=budget), max_length=budget
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


def _unique_by_id(refs: Sequence[TrackRef]) -> list[TrackRef]:
    """Drop repeated video ids, keeping the first occurrence and input order.

    ``staging_dir`` is keyed on the video id, so two refs naming the same video
    share one scratch directory: at the default ``jobs > 1`` both clear the
    archive check before either records, and whichever finishes first rmtrees
    the other's working files mid-download. Reachable from ``yt2mp3 URL URL``,
    a ``--from-file`` list with a repeat, or a playlist containing the same
    video twice -- and none of ``collect_urls``, ``expand`` or the archive
    deduplicates before this point.
    """
    seen: set[str] = set()
    unique: list[TrackRef] = []
    for ref in refs:
        if ref.video_id not in seen:
            seen.add(ref.video_id)
            unique.append(ref)
    return unique


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
    queue = _unique_by_id(refs)
    # The deduplicated count, not the given one: a batch bar told there are
    # four tracks when three will ever finish stops one short of full.
    reporter.batch_started(len(queue))
    if not queue:
        return BatchResult()

    results: list[TrackOutcome | None] = [None] * len(queue)
    stems = StemRegistry()
    # Not `with ThreadPoolExecutor(...)`: its __exit__ calls shutdown(wait=True),
    # which negates the wait=False below and holds a Ctrl-C until every in-flight
    # fetch has finished -- minutes on an hour-long mix, with the bars still
    # animating and nothing said. The explicit finally lets the exception
    # surface at once.
    pool = ThreadPoolExecutor(max_workers=settings.jobs)
    try:
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
            for index, ref in enumerate(queue)
        }
        # No `except BaseException: raise` clause is needed to reach the
        # cancellation: a finally runs for KeyboardInterrupt too, which the
        # narrower `except Exception` never would. Only a run-level abort can
        # get out of here at all -- a cancelled download, an unreadable cookie
        # store, Ctrl-C -- because process_track returns its own failures.
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    finally:
        # cancel_futures drops everything still queued, so the remaining 199
        # tracks of a 200-track playlist never start. wait=False means we do
        # not block on the handful already running -- they are not killed:
        # each finishes its download, publishes and records its track, and
        # concurrent.futures' own atexit hook joins the threads before the
        # interpreter exits. On the success path every future is already done,
        # so this cancels nothing and returns immediately.
        pool.shutdown(wait=False, cancel_futures=True)

    # Input order, not completion order: the summary should read like the queue.
    return BatchResult(tuple(outcome for outcome in results if outcome is not None))
