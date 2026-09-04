from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from yt_dlp.utils import DownloadCancelled

from memill import pipeline
from memill.config import Settings, VbrQuality
from memill.encoder import LOUDNORM, run_encode
from memill.errors import DownloadError, TransferError, Yt2Mp3Error
from memill.naming import (
    DEFAULT_MAX_STEM,
    MIN_STEM,
    WINDOWS_PATH_LIMIT,
    stem_budget,
)
from memill.pipeline import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    BatchResult,
    StemRegistry,
    TrackOutcome,
    run_batch,
    staging_dir,
)
from memill.source import SourceMedia, TrackRef
from memill.transfer import Archive

REF = TrackRef(video_id="aaa", url="https://x/aaa", title="Some Mix", duration=12.0)


class FakeDownloader:
    """Writes plausible staging files instead of touching the network.

    It also records what it was handed -- the staging directory, and how many
    fetches were in flight at once -- because those are properties of the
    orchestration that nothing else can observe.
    """

    def __init__(
        self,
        *,
        fail: bool = False,
        fail_ids: tuple[str, ...] = (),
        cancel_ids: tuple[str, ...] = (),
        cancel_error: type[BaseException] = DownloadCancelled,
        info_extra: dict[str, Any] | None = None,
        cover: bool = False,
        delay: dict[str, float] | None = None,
        suffix: str = ".opus",
    ) -> None:
        self.fail = fail
        self.fail_ids = set(fail_ids)
        self.cancel_ids = set(cancel_ids)
        self.cancel_error = cancel_error
        self.info_extra = dict(info_extra or {})
        self.cover = cover
        self.delay = dict(delay or {})
        # What YouTube served. The pipeline budgets the filename against the
        # real suffix, so a test can vary it.
        self.suffix = suffix
        self.calls: list[TrackRef] = []
        self.staging_seen: dict[str, Path] = {}
        self.staging_was_dir: list[bool] = []
        # What was already in the directory on arrival -- the only way to see
        # that a retry really is handed the partial file it should resume.
        self.staging_contents: list[list[str]] = []
        self.max_concurrent = 0
        self._live = 0
        self._lock = threading.Lock()

    def fetch(
        self, ref: TrackRef, staging: Path, on_progress: Any = None
    ) -> SourceMedia:
        with self._lock:
            self.calls.append(ref)
            self.staging_seen[ref.video_id] = staging
            self.staging_was_dir.append(staging.is_dir())
            self.staging_contents.append(
                sorted(path.name for path in staging.iterdir())
                if staging.is_dir()
                else []
            )
            self._live += 1
            self.max_concurrent = max(self.max_concurrent, self._live)
        try:
            time.sleep(self.delay.get(ref.video_id, 0.0))
            if ref.video_id in self.cancel_ids:
                # DownloadCancelled is yt-dlp's own, which source.py
                # deliberately lets through unwrapped so it aborts the run
                # rather than one track. KeyboardInterrupt is Ctrl-C, and is
                # not an Exception -- the case the handler's BaseException
                # clause exists for.
                raise self.cancel_error("user cancelled")
            if self.fail or ref.video_id in self.fail_ids:
                raise DownloadError("network went away")
            audio = staging / f"{ref.video_id}{self.suffix}"
            audio.write_bytes(b"opus")
            cover: Path | None = None
            if self.cover:
                cover = staging / f"{ref.video_id}.jpg"
                cover.write_bytes(b"jpeg")
            if on_progress:
                on_progress(1.0)
            info = {"id": ref.video_id, "title": ref.title, **self.info_extra}
            return SourceMedia(audio=audio, cover=cover, info=info)
        finally:
            with self._lock:
                self._live -= 1


def fake_encode(
    argv: list[str], *, duration: float | None, on_progress: Any = None
) -> None:
    Path(argv[-1]).write_bytes(b"mp3-bytes")
    if on_progress:
        on_progress(1.0)


class RecordingEncoder:
    """``fake_encode`` that also keeps the argv and duration it was given."""

    def __init__(self) -> None:
        self.argv: list[list[str]] = []
        self.durations: list[float | None] = []

    def __call__(
        self, argv: list[str], *, duration: float | None, on_progress: Any = None
    ) -> None:
        self.argv.append(list(argv))
        self.durations.append(duration)
        fake_encode(argv, duration=duration, on_progress=on_progress)


class RecordingReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, ...]] = []
        self.progress: list[tuple[str, float]] = []

    def batch_started(self, total: int) -> None:
        self.events.append(("batch", str(total)))

    def track_started(self, key: str, label: str) -> None:
        self.events.append(("start", key, label))

    def track_phase(self, key: str, phase: str) -> None:
        self.events.append(("phase", key, phase))

    def track_progress(self, key: str, fraction: float) -> None:
        self.events.append(("progress", key))
        self.progress.append((key, fraction))

    def track_finished(self, key: str, status: str, detail: str | None = None) -> None:
        self.events.append(("finish", key, status))

    def close(self) -> None:
        self.events.append(("close",))


class ObservingArchive(Archive):
    """Records whether ``watch`` was on disk at the moment ``add`` was called.

    This is the only way to tell "the archive was written after publishing"
    apart from "the archive was written at all".
    """

    def __init__(self, path: Path, watch: Path) -> None:
        super().__init__(path)
        self._watch = watch
        self.existed_on_add: list[bool] = []

    def add(self, key: str) -> None:
        self.existed_on_add.append(self._watch.exists())
        super().add(key)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "destination": tmp_path / "library",
        "staging_root": tmp_path / "stage",
        "quality": VbrQuality(0),
        "jobs": 1,
        "fragments": 8,
    }
    return Settings(**{**base, **overrides})


def finish_order(reporter: RecordingReporter) -> list[str]:
    return [event[1] for event in reporter.events if event[0] == "finish"]


def test_a_successful_track_is_published_and_recorded(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    archive = Archive(tmp_path / "archive.txt")
    result = run_batch(
        [REF],
        settings=settings,
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=archive,
        encode=fake_encode,
    )
    assert result.exit_code == 0
    assert result.outcomes[0].status == STATUS_DONE
    assert (settings.destination / "Some Mix.mp3").read_bytes() == b"mp3-bytes"
    assert "aaa" in archive
    # The outcome must carry the published path, not the staging one.
    assert result.outcomes[0].path == settings.destination / "Some Mix.mp3"
    assert result.outcomes[0].error is None
    # keep_source is off by default, so the opus must not be published.
    assert not (settings.destination / "Some Mix.opus").exists()


def test_an_archived_track_is_skipped_without_downloading(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive.txt")
    archive.add("aaa")
    downloader = FakeDownloader()
    reporter = RecordingReporter()
    settings = make_settings(tmp_path)
    result = run_batch(
        [REF],
        settings=settings,
        downloader=downloader,
        reporter=reporter,
        archive=archive,
        encode=fake_encode,
    )
    assert result.outcomes[0].status == STATUS_SKIPPED
    assert downloader.calls == []
    assert result.exit_code == 0
    # A skipped track must still be announced, or a reporter that keys its bar
    # on track_started can never advance for it and a fully archived resume
    # sits at 0/N for the whole run. Nothing may be downloaded or staged.
    assert reporter.events == [
        ("batch", "1"),
        ("start", "aaa", "Some Mix"),
        ("finish", "aaa", STATUS_SKIPPED),
    ]
    assert not settings.staging_root.exists()


def test_a_failing_track_does_not_stop_the_batch(tmp_path: Path) -> None:
    other = TrackRef("bbb", "https://x/bbb", "Other Mix", 5.0)
    result = run_batch(
        [REF, other],
        settings=make_settings(tmp_path, jobs=2),
        downloader=FakeDownloader(fail=True),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert [outcome.status for outcome in result.outcomes] == [
        STATUS_FAILED,
        STATUS_FAILED,
    ]
    assert result.exit_code == 1
    assert "network went away" in (result.outcomes[0].error or "")


def test_one_failure_does_not_prevent_the_others_from_succeeding(
    tmp_path: Path,
) -> None:
    refs = [TrackRef(f"id{i}", f"https://x/{i}", f"Track {i}", 1.0) for i in range(3)]
    settings = make_settings(tmp_path, jobs=3)
    archive = Archive(tmp_path / "a.txt")
    result = run_batch(
        refs,
        settings=settings,
        downloader=FakeDownloader(fail_ids=("id1",)),
        reporter=RecordingReporter(),
        archive=archive,
        encode=fake_encode,
    )
    assert [outcome.status for outcome in result.outcomes] == [
        STATUS_DONE,
        STATUS_FAILED,
        STATUS_DONE,
    ]
    assert (settings.destination / "Track 0.mp3").read_bytes() == b"mp3-bytes"
    assert (settings.destination / "Track 2.mp3").read_bytes() == b"mp3-bytes"
    assert not (settings.destination / "Track 1.mp3").exists()
    assert "id0" in archive
    assert "id2" in archive
    assert "id1" not in archive
    assert result.completed == (result.outcomes[0], result.outcomes[2])
    assert result.failed == (result.outcomes[1],)
    assert result.exit_code == 1


def test_a_failed_track_keeps_its_staging_directory_for_the_retry(
    tmp_path: Path,
) -> None:
    """The whole point of the directory surviving: the next run resumes.

    yt-dlp writes a ``.part`` file into staging and continues from it, and
    ``staging_dir`` is keyed on the video id, so the retry is handed the very
    directory the failed attempt left behind. Removing it on failure -- what a
    ``finally`` does -- is what turned a network blip at 81% of a three-hour
    mix into a download starting again at 0%.

    Real resumption needs yt-dlp and a network, so what is pinned here is the
    retention contract underneath it: kept on failure, the same path for the
    same id, the partial file still there when the retry arrives, and gone
    once the retry succeeds.
    """
    settings = make_settings(tmp_path)
    staging = settings.staging_root / "aaa"
    failed = FakeDownloader(fail=True)
    run_batch(
        [REF],
        settings=settings,
        downloader=failed,
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    # The directory has to have existed for its retention to mean anything.
    assert failed.staging_was_dir == [True]
    assert failed.staging_seen["aaa"] == staging
    assert staging.is_dir()

    (staging / "aaa.opus.part").write_bytes(b"the first 81 per cent")

    retried = FakeDownloader()
    run_batch(
        [REF],
        settings=settings,
        downloader=retried,
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    # Same directory, and the partial download still in it when yt-dlp -- for
    # which this fake stands in -- was handed it.
    assert retried.staging_seen["aaa"] == staging
    assert retried.staging_contents == [["aaa.opus.part"]]
    # Reclaimed by the run that succeeded, so the kept directory is not a leak.
    assert not staging.exists()


def test_staging_is_removed_after_a_successful_track(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    downloader = FakeDownloader()
    run_batch(
        [REF],
        settings=settings,
        downloader=downloader,
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert downloader.staging_was_dir == [True]
    assert not (settings.staging_root / "aaa").exists()
    assert list(settings.staging_root.iterdir()) == []


def test_outcomes_keep_input_order_not_completion_order(tmp_path: Path) -> None:
    refs = [TrackRef(f"id{i}", f"https://x/{i}", f"Track {i}", 1.0) for i in range(5)]
    # Every worker starts at once and the first ref finishes last, so completion
    # order is the exact reverse of input order. With an instant fake the two
    # coincide and the assertion below would hold even for a broken pipeline.
    delay = {f"id{i}": 0.05 * (len(refs) - i) for i in range(len(refs))}
    reporter = RecordingReporter()
    result = run_batch(
        refs,
        settings=make_settings(tmp_path, jobs=len(refs)),
        downloader=FakeDownloader(delay=delay),
        reporter=reporter,
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    expected = [ref.video_id for ref in refs]
    assert finish_order(reporter) == list(reversed(expected))  # the premise
    assert [outcome.ref.video_id for outcome in result.outcomes] == expected
    assert [outcome.status for outcome in result.outcomes] == [STATUS_DONE] * 5


def test_the_reporter_sees_both_phases(tmp_path: Path) -> None:
    reporter = RecordingReporter()
    run_batch(
        [REF],
        settings=make_settings(tmp_path),
        downloader=FakeDownloader(),
        reporter=reporter,
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    phases = [event[2] for event in reporter.events if event[0] == "phase"]
    assert phases == ["downloading", "encoding"]


def test_the_reporter_is_told_the_total_and_the_final_status(tmp_path: Path) -> None:
    reporter = RecordingReporter()
    run_batch(
        [REF],
        settings=make_settings(tmp_path),
        downloader=FakeDownloader(),
        reporter=reporter,
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert reporter.events[0] == ("batch", "1")
    assert reporter.events[1] == ("start", "aaa", "Some Mix")
    assert reporter.events[-1] == ("finish", "aaa", STATUS_DONE)


def test_progress_from_both_phases_reaches_the_reporter(tmp_path: Path) -> None:
    reporter = RecordingReporter()
    run_batch(
        [REF],
        settings=make_settings(tmp_path),
        downloader=FakeDownloader(),
        reporter=reporter,
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    # One tick from the downloader, one from the encoder: dropping the callback
    # from either call site leaves only one.
    assert reporter.progress == [("aaa", 1.0), ("aaa", 1.0)]


def test_the_archive_is_written_only_after_the_file_is_published(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    archive = ObservingArchive(
        tmp_path / "a.txt", settings.destination / "Some Mix.mp3"
    )
    result = run_batch(
        [REF],
        settings=settings,
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=archive,
        encode=fake_encode,
    )
    assert result.outcomes[0].status == STATUS_DONE
    assert archive.existed_on_add == [True]


def test_a_failed_publish_leaves_the_track_out_of_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(source: Path, destination_dir: Path, filename: str) -> Path:
        raise TransferError("read-only filesystem")

    monkeypatch.setattr(pipeline, "publish", boom)
    settings = make_settings(tmp_path)
    archive = Archive(tmp_path / "a.txt")
    result = run_batch(
        [REF],
        settings=settings,
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=archive,
        encode=fake_encode,
    )
    assert result.outcomes[0].status == STATUS_FAILED
    assert "read-only filesystem" in (result.outcomes[0].error or "")
    assert "aaa" not in archive
    # Kept, like every other failure: the downloaded audio is still in there
    # and a retry should not have to fetch it again.
    assert (settings.staging_root / "aaa").is_dir()


def test_an_empty_batch_is_announced_and_returns_no_outcomes(tmp_path: Path) -> None:
    reporter = RecordingReporter()
    result = run_batch(
        [],
        settings=make_settings(tmp_path),
        downloader=FakeDownloader(),
        reporter=reporter,
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert result.outcomes == ()
    assert result.exit_code == 0
    assert reporter.events == [("batch", "0")]


def test_at_most_settings_jobs_tracks_run_at_once(tmp_path: Path) -> None:
    refs = [TrackRef(f"id{i}", f"https://x/{i}", f"Track {i}", 1.0) for i in range(6)]
    downloader = FakeDownloader(delay={ref.video_id: 0.05 for ref in refs})
    run_batch(
        refs,
        settings=make_settings(tmp_path, jobs=2),
        downloader=downloader,
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert downloader.max_concurrent == 2


def test_the_filename_is_budgeted_against_the_destination_length(
    tmp_path: Path,
) -> None:
    destination = tmp_path / ("d" * 120)
    settings = make_settings(tmp_path, destination=destination)
    budget = stem_budget(destination)
    assert MIN_STEM <= budget < DEFAULT_MAX_STEM  # the premise
    ref = TrackRef("aaa", "https://x/aaa", "T" * 200, 1.0)
    result = run_batch(
        [ref],
        settings=settings,
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert result.outcomes[0].status == STATUS_DONE
    assert [path.name for path in destination.iterdir()] == [f"{'T' * budget}.mp3"]


def test_a_destination_too_long_for_a_filename_fails_only_that_track(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path, destination=tmp_path / ("d" * 240))
    result = run_batch(
        [REF],
        settings=settings,
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    # stem_budget raises TransferError, which is a Yt2Mp3Error and so must be
    # caught here rather than escaping and killing the batch.
    assert result.outcomes[0].status == STATUS_FAILED
    assert "too long" in (result.outcomes[0].error or "")
    # A failure inside the staging block keeps the directory, here as anywhere.
    assert (settings.staging_root / "aaa").is_dir()


def test_clean_titles_can_be_switched_off(tmp_path: Path) -> None:
    ref = TrackRef("aaa", "https://x/aaa", "Artist - Song (Official Video)", 1.0)
    settings = make_settings(tmp_path, clean_titles=False)
    run_batch(
        [ref],
        settings=settings,
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    # Cleaning would have produced "Artist - Song.mp3" instead.
    assert (settings.destination / "Artist - Song (Official Video).mp3").exists()
    assert not (settings.destination / "Artist - Song.mp3").exists()


def test_keep_source_publishes_the_original_audio_too(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, keep_source=True)
    run_batch(
        [REF],
        settings=settings,
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert (settings.destination / "Some Mix.mp3").read_bytes() == b"mp3-bytes"
    assert (settings.destination / "Some Mix.opus").read_bytes() == b"opus"


def test_the_encode_command_is_built_from_the_settings_and_the_media(
    tmp_path: Path,
) -> None:
    settings = make_settings(
        tmp_path, quality=VbrQuality(4), normalize=True, jobs=1
    )
    encoder = RecordingEncoder()
    downloader = FakeDownloader(cover=True, info_extra={"artist": "Someone"})
    result = run_batch(
        [REF],
        settings=settings,
        downloader=downloader,
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=encoder,
    )
    assert result.outcomes[0].status == STATUS_DONE
    argv = encoder.argv[0]
    assert argv[argv.index("-q:a") + 1] == "4"  # settings.quality, not a default
    assert LOUDNORM in argv  # settings.normalize
    assert str(tmp_path / "stage" / "aaa" / "aaa.jpg") in argv  # media.cover
    assert "title=Some Mix" in argv
    assert "artist=Someone" in argv
    # Encoding happens in staging; only publish moves it to the library.
    assert argv[-1] == str(tmp_path / "stage" / "aaa" / "Someone - Some Mix.mp3")


def test_the_encoder_is_told_the_duration_from_the_ref_or_the_info(
    tmp_path: Path,
) -> None:
    encoder = RecordingEncoder()
    known = make_settings(tmp_path / "known")
    run_batch(
        [REF],
        settings=known,
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=encoder,
    )
    assert encoder.durations == [12.0]

    unknown = make_settings(tmp_path / "unknown")
    run_batch(
        [TrackRef("bbb", "https://x/bbb", "No Duration", None)],
        settings=unknown,
        downloader=FakeDownloader(info_extra={"duration": 7.5}),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "b.txt"),
        encode=encoder,
    )
    assert encoder.durations == [12.0, 7.5]


def test_batch_result_separates_completed_from_failed() -> None:
    done = TrackOutcome(REF, STATUS_DONE, path=Path("Some Mix.mp3"))
    skipped = TrackOutcome(REF, STATUS_SKIPPED)
    failed = TrackOutcome(REF, STATUS_FAILED, error="boom")
    result = BatchResult((done, skipped, failed))
    assert result.completed == (done,)
    assert result.failed == (failed,)
    assert result.exit_code == 1
    assert BatchResult((done, skipped)).exit_code == 0
    assert BatchResult().outcomes == ()


def test_two_tracks_with_the_same_title_do_not_overwrite_each_other(
    tmp_path: Path,
) -> None:
    """The silent-data-loss case: same stem, one file, both reported done."""
    refs = [
        TrackRef("aaa", "https://x/aaa", "Duplicate Title", 1.0),
        TrackRef("bbb", "https://x/bbb", "Duplicate Title", 1.0),
    ]
    settings = make_settings(tmp_path, jobs=1)
    archive = Archive(tmp_path / "a.txt")

    class DistinctBytes(FakeDownloader):
        def fetch(
            self, ref: TrackRef, staging: Path, on_progress: Any = None
        ) -> SourceMedia:
            media = super().fetch(ref, staging, on_progress)
            media.audio.write_bytes(ref.video_id.encode())
            return media

    def encode_the_source(
        argv: list[str], *, duration: float | None, on_progress: Any = None
    ) -> None:
        # Carry the source bytes through, so a lost track is visible as lost
        # bytes rather than as two identical files.
        Path(argv[-1]).write_bytes(Path(argv[argv.index("-i") + 1]).read_bytes())
        if on_progress:
            on_progress(1.0)

    result = run_batch(
        refs,
        settings=settings,
        downloader=DistinctBytes(),
        reporter=RecordingReporter(),
        archive=archive,
        encode=encode_the_source,
    )
    assert [outcome.status for outcome in result.outcomes] == [STATUS_DONE] * 2
    assert (settings.destination / "Duplicate Title.mp3").read_bytes() == b"aaa"
    assert (settings.destination / "Duplicate Title (2).mp3").read_bytes() == b"bbb"
    assert sorted(path.name for path in settings.destination.iterdir()) == [
        "Duplicate Title (2).mp3",
        "Duplicate Title.mp3",
    ]
    assert "aaa" in archive
    assert "bbb" in archive


def test_stems_differing_only_in_case_are_treated_as_a_collision(
    tmp_path: Path,
) -> None:
    # The destination is an NTFS mount, where "Track" and "TRACK" are one file.
    refs = [
        TrackRef("aaa", "https://x/aaa", "Track", 1.0),
        TrackRef("bbb", "https://x/bbb", "TRACK", 1.0),
    ]
    settings = make_settings(tmp_path, jobs=1)
    result = run_batch(
        refs,
        settings=settings,
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert [outcome.status for outcome in result.outcomes] == [STATUS_DONE] * 2
    assert sorted(path.name for path in settings.destination.iterdir()) == [
        "TRACK (2).mp3",
        "Track.mp3",
    ]


def test_the_stem_registry_hands_out_one_name_per_caller() -> None:
    registry = StemRegistry()
    assert registry.claim("Song", max_length=150) == "Song"
    assert registry.claim("Song", max_length=150) == "Song (2)"
    assert registry.claim("Song", max_length=150) == "Song (3)"
    assert registry.claim("song", max_length=150) == "song (4)"  # case-folded
    assert registry.claim("Other", max_length=150) == "Other"


def test_the_stem_registry_makes_room_for_the_marker() -> None:
    """The marker replaces the tail; it never extends past the budget."""
    registry = StemRegistry()
    assert registry.claim("A" * 20, max_length=20) == "A" * 20
    assert registry.claim("A" * 20, max_length=20) == f"{'A' * 16} (2)"
    assert registry.claim("A" * 20, max_length=20) == f"{'A' * 16} (3)"
    # A wider marker eats more of the base, not more of the budget.
    registry = StemRegistry()
    for _ in range(10):
        assert len(registry.claim("B" * 30, max_length=30)) == 30
    assert registry.claim("B" * 30, max_length=30) == f"{'B' * 25} (11)"


@pytest.mark.parametrize("abort", [DownloadCancelled, KeyboardInterrupt])
def test_a_run_level_abort_stops_the_queue(
    tmp_path: Path, abort: type[BaseException]
) -> None:
    refs = [TrackRef("bad", "https://x/bad", "Bad", 1.0)] + [
        TrackRef(f"id{i}", f"https://x/{i}", f"Track {i}", 1.0) for i in range(1, 8)
    ]
    settings = make_settings(tmp_path, jobs=1)
    # The survivors are slow, so the cancel lands long before a second one
    # could finish; with max_workers=1 at most one other can be in flight.
    downloader = FakeDownloader(
        cancel_ids=("bad",),
        cancel_error=abort,
        delay={f"id{i}": 0.2 for i in range(1, 8)},
    )
    with pytest.raises(abort):
        run_batch(
            refs,
            settings=settings,
            downloader=downloader,
            reporter=RecordingReporter(),
            archive=Archive(tmp_path / "a.txt"),
            encode=fake_encode,
        )
    started = {call.video_id for call in downloader.calls}
    assert "bad" in started
    assert len(started) <= 2  # the abort, plus at most one already in flight
    assert not any(
        (settings.destination / f"Track {i}.mp3").exists() for i in range(3, 8)
    )


def test_a_staging_key_that_could_escape_the_root_is_refused(tmp_path: Path) -> None:
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keep.txt").write_text("do not delete")
    root = tmp_path / "stage"
    root.mkdir()

    for key in ("../precious", "..", ".", "a/b", "a\\b", ""):
        with pytest.raises(Yt2Mp3Error), staging_dir(root, key):
            pass  # pragma: no cover - the body must never run

    # rmtree is irreversible, so the guard has to hold before anything is made.
    assert (precious / "keep.txt").read_text() == "do not delete"
    assert list(root.iterdir()) == []


def test_a_good_staging_key_still_works(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    with staging_dir(root, "dQw4w9WgXcQ") as staging:
        assert staging == root / "dQw4w9WgXcQ"
        assert staging.is_dir()
    assert not staging.exists()


def test_staging_survives_an_interrupt_so_the_next_run_can_resume(
    tmp_path: Path,
) -> None:
    """Ctrl-C is the case that most wants a resumable ``.part`` file.

    ``KeyboardInterrupt`` is not an ``Exception``, so a handler narrower than
    ``BaseException`` would let the directory be swept away by exactly the
    interruption the user expects to pick up from.
    """
    root = tmp_path / "stage"
    with pytest.raises(KeyboardInterrupt), staging_dir(root, "aaa") as staging:
        (staging / "aaa.opus.part").write_bytes(b"half a mix")
        raise KeyboardInterrupt

    assert (root / "aaa" / "aaa.opus.part").read_bytes() == b"half a mix"


def test_staging_is_kept_when_the_body_raises_and_the_error_still_escapes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage"
    with pytest.raises(DownloadError, match="network went away"), staging_dir(
        root, "aaa"
    ):
        raise DownloadError("network went away")

    assert (root / "aaa").is_dir()


def test_an_unwritable_archive_fails_the_track_instead_of_the_run(
    tmp_path: Path,
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    locked = tmp_path / "locked"
    locked.mkdir()
    settings = make_settings(tmp_path)
    archive = Archive(locked / "archive.txt")
    locked.chmod(0o500)
    try:
        result = run_batch(
            [REF],
            settings=settings,
            downloader=FakeDownloader(),
            reporter=RecordingReporter(),
            archive=archive,
            encode=fake_encode,
        )
    finally:
        locked.chmod(0o700)
    # A bare PermissionError here would fly past the Yt2Mp3Error handler and
    # abort a batch whose files had all been published.
    assert result.outcomes[0].status == STATUS_FAILED
    assert "archive" in (result.outcomes[0].error or "")
    assert (settings.destination / "Some Mix.mp3").exists()


def test_a_long_duplicate_title_still_fits_the_destination_path(
    tmp_path: Path,
) -> None:
    """The marker must be budgeted, not appended after the budget is spent."""
    # Size the destination so stem_budget lands just above MIN_STEM: that is the
    # regime where four appended characters cross WINDOWS_PATH_LIMIT.
    target_budget = 40
    pad = WINDOWS_PATH_LIMIT - 11 - target_budget - len(str(tmp_path)) - 1
    destination = tmp_path / ("d" * pad)
    settings = make_settings(tmp_path, destination=destination, jobs=1)
    budget = stem_budget(destination)
    assert budget == target_budget  # the premise: near the boundary, not far

    title = "L" * 200
    refs = [
        TrackRef("aaa", "https://x/aaa", title, 1.0),
        TrackRef("bbb", "https://x/bbb", title, 1.0),
    ]
    result = run_batch(
        refs,
        settings=settings,
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert [outcome.status for outcome in result.outcomes] == [STATUS_DONE] * 2

    names = sorted(path.name for path in destination.iterdir())
    assert len(names) == 2
    for name in names:
        stem = name[: -len(".mp3")]
        # publish writes "<stem>.mp3.part" before revealing the final name, so
        # that is the longest path this file ever occupies.
        assert (
            len(str(destination)) + 1 + len(stem) + len(".mp3.part")
            < WINDOWS_PATH_LIMIT
        )
        assert len(stem) <= budget
    # Exactly one carries the marker, and it was truncated to make room for it
    # rather than grown past the budget.
    marked = [name for name in names if name.endswith(" (2).mp3")]
    assert len(marked) == 1
    assert len(marked[0]) == budget + len(".mp3")


class GatedDownloader(FakeDownloader):
    """``FakeDownloader`` whose named track blocks until the test releases it.

    An event rather than a plain sleep, because the assertion is about how
    long ``run_batch`` takes to raise *while a fetch is still running*. A
    fixed sleep would either be too short to be evidence, or would leave a
    worker thread writing into a ``tmp_path`` pytest is about to delete.
    """

    def __init__(self, gated_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._gated_id = gated_id
        self.entered = threading.Event()
        self.released = threading.Event()

    def fetch(
        self, ref: TrackRef, staging: Path, on_progress: Any = None
    ) -> SourceMedia:
        if ref.video_id == self._gated_id:
            self.entered.set()
            # The brief's "one in-flight track taking ~5s": long enough that a
            # shutdown which waits for it is unmistakable against a 1s bound.
            self.released.wait(timeout=5.0)
        else:
            # Every other track waits for the gated one to be genuinely in
            # flight before it aborts. Without this the two are only racing:
            # a failure path quick enough to finish the batch before the other
            # worker has even called fetch leaves nothing in flight for the
            # shutdown to wait on, and the test measures nothing. Staging
            # directories are no longer rmtree'd on the way out of a failure,
            # which took exactly enough work out of that path to make the race
            # lose about one run in seven.
            self.entered.wait(timeout=5.0)
        return super().fetch(ref, staging, on_progress)


def test_an_abort_does_not_wait_for_the_track_already_running(
    tmp_path: Path,
) -> None:
    """Ctrl-C must surface at once, not once the in-flight fetch has finished.

    ``with ThreadPoolExecutor(...)`` calls ``shutdown(wait=True)`` on the way
    out, which silently negates the ``wait=False`` of the cancellation inside
    it. Measured with six-second fetches: SIGINT at t=1.00s, run_batch raised
    at t=6.01s. On an hour-long mix that is minutes of animating bars and no
    message. The abort test above never asserted elapsed time, which is
    exactly why this stayed invisible.

    The second half is the other half of the contract: cancelling must not
    abandon the track already downloading. It still publishes and is still
    recorded, so a resume does not re-fetch it.
    """
    settings = make_settings(tmp_path, jobs=2)
    downloader = GatedDownloader(
        "slow", cancel_ids=("bad",), cancel_error=KeyboardInterrupt
    )
    refs = [
        TrackRef("slow", "https://x/slow", "Slow Track", 1.0),
        TrackRef("bad", "https://x/bad", "Bad", 1.0),
    ]
    archive = Archive(tmp_path / "a.txt")
    start = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            run_batch(
                refs,
                settings=settings,
                downloader=downloader,
                reporter=RecordingReporter(),
                archive=archive,
                encode=fake_encode,
            )
        elapsed = time.monotonic() - start
        assert downloader.entered.is_set(), "the gated fetch never started"
        assert elapsed < 1.0, f"the abort waited {elapsed:.2f}s for the in-flight fetch"
    finally:
        downloader.released.set()

    published = settings.destination / "Slow Track.mp3"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not published.exists():
        time.sleep(0.01)
    # The worker was not killed: concurrent.futures joins it at interpreter
    # shutdown, and it finishes its track properly on the way out.
    assert published.read_bytes() == b"mp3-bytes"
    while time.monotonic() < deadline and "slow" not in archive:
        time.sleep(0.01)
    assert "slow" in archive


def test_a_repeated_video_id_is_processed_once(tmp_path: Path) -> None:
    """Two refs naming the same video would share one staging directory.

    ``staging_dir`` is keyed on the video id, so at the default ``jobs > 1``
    both refs clear the archive check before either records, and whichever
    finishes first rmtrees the other's working files mid-download. Nothing
    upstream deduplicates: ``memill URL URL``, a ``--from-file`` list with a
    repeat and a playlist holding the same video twice all reach here intact.
    """
    settings = make_settings(tmp_path, jobs=2)
    # Same id, different spelling and title -- the pair a playlist and a direct
    # URL produce. The first occurrence wins, so the first title is the file.
    refs = [
        TrackRef("aaa", "https://x/aaa", "Some Mix", 12.0),
        TrackRef("aaa", "https://youtu.be/aaa", "Some Mix [Official Audio]", 12.0),
    ]
    downloader = FakeDownloader()
    reporter = RecordingReporter()
    result = run_batch(
        refs,
        settings=settings,
        downloader=downloader,
        reporter=reporter,
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert [outcome.status for outcome in result.outcomes] == [STATUS_DONE]
    assert result.outcomes[0].ref is refs[0]
    assert result.exit_code == 0
    assert [call.video_id for call in downloader.calls] == ["aaa"]
    assert [path.name for path in settings.destination.iterdir()] == ["Some Mix.mp3"]
    # The batch bar is told the deduplicated count, or it stops one short.
    assert ("batch", "1") in reporter.events


def test_keep_source_still_fits_the_destination_path(tmp_path: Path) -> None:
    """``--keep-source`` publishes a second file under a longer suffix.

    The stem budget reserves ``.mp3`` (4 characters), but the kept source is
    whatever YouTube served -- ``.webm`` or ``.opus``, 5 characters. At a
    binding budget that one extra character puts the ``.part`` form at 261
    with the NUL, over the limit, after the download has been paid for.

    The suffix pinned here is the one the downloader really produced, because
    that is the one the pipeline now budgets for: the download has already
    happened by the time the budget is taken, so no conservative guess is
    involved.
    """
    target_budget = 40
    source_extension = ".opus"  # what FakeDownloader writes, as YouTube does
    # `used` inside stem_budget: destination + "/" + suffix + ".part" + NUL.
    reserved = 1 + len(source_extension) + len(".part") + 1
    pad = WINDOWS_PATH_LIMIT - reserved - target_budget - len(str(tmp_path)) - 1
    destination = tmp_path / ("d" * pad)
    settings = make_settings(
        tmp_path, destination=destination, jobs=1, keep_source=True
    )
    # The premise: at this destination the two budgets genuinely differ, and
    # that single character is what crosses the limit.
    assert stem_budget(destination, extension=source_extension) == target_budget
    assert stem_budget(destination) == target_budget + 1

    result = run_batch(
        [TrackRef("aaa", "https://x/aaa", "L" * 200, 1.0)],
        settings=settings,
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert result.outcomes[0].status == STATUS_DONE

    names = sorted(path.name for path in destination.iterdir())
    assert [Path(name).suffix for name in names] == [".mp3", ".opus"]
    for name in names:
        # publish writes "<name>.part" before revealing it, so that is the
        # longest path this file ever occupies. `<` not `<=`: MAX_PATH counts
        # the NUL terminator that this expression does not.
        assert (
            len(str(destination)) + 1 + len(name) + len(".part") < WINDOWS_PATH_LIMIT
        )


def test_an_unwritable_staging_root_fails_the_track_instead_of_the_run(
    tmp_path: Path,
) -> None:
    """``staging_dir``'s mkdir must not raise outside the Yt2Mp3Error contract.

    A bare OSError here escapes ``process_track``'s handler and aborts the
    whole batch -- the identical defect ``Archive.add`` already guards, one
    module over.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    root = tmp_path / "stage"
    root.mkdir()
    settings = make_settings(tmp_path, staging_root=root)
    root.chmod(0o500)
    try:
        result = run_batch(
            [REF],
            settings=settings,
            downloader=FakeDownloader(),
            reporter=RecordingReporter(),
            archive=Archive(tmp_path / "a.txt"),
            encode=fake_encode,
        )
    finally:
        root.chmod(0o700)
    assert result.outcomes[0].status == STATUS_FAILED
    assert "staging" in (result.outcomes[0].error or "")
    assert result.exit_code == 1


def test_an_uncreatable_destination_fails_each_track_not_the_batch(
    tmp_path: Path,
) -> None:
    """``publish`` must not let a bare ``OSError`` out of its ``mkdir``.

    The destination mount going missing, or ``-o`` into a directory the user
    cannot write, raises ``PermissionError``. Outside ``Yt2Mp3Error`` it
    escapes ``process_track``, comes back out of ``future.result()`` and
    cancels every track still queued -- so two tracks return no outcomes at
    all instead of two failures.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    locked = tmp_path / "locked"
    locked.mkdir()
    settings = make_settings(tmp_path, destination=locked / "library", jobs=1)
    locked.chmod(0o500)
    refs = [
        TrackRef("aaa", "https://x/aaa", "One", 1.0),
        TrackRef("bbb", "https://x/bbb", "Two", 1.0),
    ]
    try:
        result = run_batch(
            refs,
            settings=settings,
            downloader=FakeDownloader(),
            reporter=RecordingReporter(),
            archive=Archive(tmp_path / "a.txt"),
            encode=fake_encode,
        )
    finally:
        locked.chmod(0o700)

    assert [outcome.status for outcome in result.outcomes] == [STATUS_FAILED] * 2
    assert all("could not publish" in (o.error or "") for o in result.outcomes)


def test_ffmpeg_vanishing_mid_run_fails_each_track_not_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``require_ffmpeg`` closes the startup window, not the per-track one.

    Run with the real ``run_encode``, because the defect is inside it: a
    ``FileNotFoundError`` out of ``Popen`` is not a ``Yt2Mp3Error``, so it
    escapes the whole batch instead of failing one track.
    """

    def refuse(*_: object, **__: object) -> None:
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    refs = [
        TrackRef("aaa", "https://x/aaa", "One", 1.0),
        TrackRef("bbb", "https://x/bbb", "Two", 1.0),
    ]
    result = run_batch(
        refs,
        settings=make_settings(tmp_path, jobs=1),
        downloader=FakeDownloader(),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=run_encode,
    )

    assert [outcome.status for outcome in result.outcomes] == [STATUS_FAILED] * 2
    assert all("could not run ffmpeg" in (o.error or "") for o in result.outcomes)


def test_a_claimed_stem_is_never_a_name_windows_would_trim() -> None:
    """``claim(".hack", max_length=3)`` used to return the bare marker " (2)".

    The head truncated to ".", ``rstrip(". ")`` emptied it, and what was left
    was a filename beginning with a space -- which Windows Explorer silently
    trims, so the second track lands on the first one's name.
    """
    registry = StemRegistry()
    first = registry.claim(".hack", max_length=3)
    second = registry.claim(".hack", max_length=3)
    third = registry.claim(".hack", max_length=3)

    for name in (first, second, third):
        assert name == name.strip(), f"{name!r} would be trimmed by Windows"
        assert not name.endswith("."), f"{name!r} ends in a dot"
        assert name, "an empty stem is not a filename"
    assert len({first, second, third}) == 3


def test_claiming_a_marker_never_yields_a_reserved_or_empty_head() -> None:
    """The head is re-sanitised, not merely re-sliced."""
    registry = StemRegistry()
    registry.claim("CON", max_length=8)
    assert registry.claim("CON", max_length=8) == "_CON (2)"

    empty = StemRegistry()
    empty.claim("...", max_length=8)
    assert empty.claim("...", max_length=8).strip() != ""


def test_keep_source_budgets_the_real_suffix_not_a_guess(tmp_path: Path) -> None:
    """The budget is taken AFTER the download, so the suffix is known.

    It used to assume the longest plausible one (``.webm``, five characters),
    which costs a character of title on every ``--keep-source`` run whose
    source is shorter -- and at a binding budget that character is the title's
    last one.
    """
    target_budget = 40
    source_extension = ".m4a"  # four characters, like the .mp3 beside it
    reserved = 1 + len(".mp3") + len(".part") + 1
    pad = WINDOWS_PATH_LIMIT - reserved - target_budget - len(str(tmp_path)) - 1
    destination = tmp_path / ("d" * pad)
    settings = make_settings(
        tmp_path, destination=destination, jobs=1, keep_source=True
    )
    # The premise: guessing ".webm" here would cost exactly one character.
    assert stem_budget(destination, extension=source_extension) == target_budget
    assert stem_budget(destination, extension=".webm") == target_budget - 1

    result = run_batch(
        [TrackRef("aaa", "https://x/aaa", "L" * 200, 1.0)],
        settings=settings,
        downloader=FakeDownloader(suffix=source_extension),
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    assert result.outcomes[0].status == STATUS_DONE

    names = sorted(path.name for path in destination.iterdir())
    assert [Path(name).suffix for name in names] == [".m4a", ".mp3"]
    for name in names:
        assert len(Path(name).stem) == target_budget
        assert (
            len(str(destination)) + 1 + len(name) + len(".part") < WINDOWS_PATH_LIMIT
        )
