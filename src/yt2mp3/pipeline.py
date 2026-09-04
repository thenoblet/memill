"""Per-track state machine and the thread pool that drives it."""

from __future__ import annotations

import shutil
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

EncodeFn = Callable[..., None]


class SupportsFetch(Protocol):
    """The slice of ``Downloader`` the pipeline actually uses."""

    def fetch(
        self,
        ref: TrackRef,
        staging: Path,
        on_progress: Callable[[float], None] | None = None,
    ) -> SourceMedia: ...


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


@contextmanager
def staging_dir(root: Path, key: str) -> Iterator[Path]:
    """A per-track scratch directory on ext4, removed however we leave it."""
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
) -> TrackOutcome:
    """Download, encode, publish and record one track.

    Every deliberate failure is caught and returned as an outcome: one bad track
    in a hundred-track playlist must not take the other ninety-nine with it.
    """
    if ref.video_id in archive:
        reporter.track_finished(ref.video_id, STATUS_SKIPPED, "already in library")
        return TrackOutcome(ref, STATUS_SKIPPED)

    reporter.track_started(ref.video_id, ref.title)

    def report(fraction: float) -> None:
        reporter.track_progress(ref.video_id, fraction)

    try:
        with staging_dir(settings.staging_root, ref.video_id) as staging:
            reporter.track_phase(ref.video_id, PHASE_DOWNLOAD)
            media = downloader.fetch(ref, staging, report)

            tags = infer_tags(media.info, clean=settings.clean_titles)
            stem = output_stem(tags, max_length=stem_budget(settings.destination))
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
            ): index
            for index, ref in enumerate(refs)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    # Input order, not completion order: the summary should read like the queue.
    return BatchResult(tuple(outcome for outcome in results if outcome is not None))
