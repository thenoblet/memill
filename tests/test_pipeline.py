from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from yt2mp3 import pipeline
from yt2mp3.config import Settings, VbrQuality
from yt2mp3.encoder import LOUDNORM
from yt2mp3.errors import DownloadError, TransferError
from yt2mp3.naming import DEFAULT_MAX_STEM, MIN_STEM, stem_budget
from yt2mp3.pipeline import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    BatchResult,
    TrackOutcome,
    run_batch,
)
from yt2mp3.source import SourceMedia, TrackRef
from yt2mp3.transfer import Archive

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
        info_extra: dict[str, Any] | None = None,
        cover: bool = False,
        delay: dict[str, float] | None = None,
    ) -> None:
        self.fail = fail
        self.fail_ids = set(fail_ids)
        self.info_extra = dict(info_extra or {})
        self.cover = cover
        self.delay = dict(delay or {})
        self.calls: list[TrackRef] = []
        self.staging_seen: dict[str, Path] = {}
        self.staging_was_dir: list[bool] = []
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
            self._live += 1
            self.max_concurrent = max(self.max_concurrent, self._live)
        try:
            time.sleep(self.delay.get(ref.video_id, 0.0))
            if self.fail or ref.video_id in self.fail_ids:
                raise DownloadError("network went away")
            audio = staging / f"{ref.video_id}.opus"
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
    # The skip must be decided before the track is announced as started, and no
    # staging directory may be created for a track we never touch.
    assert ("start", "aaa", "Some Mix") not in reporter.events
    assert ("finish", "aaa", STATUS_SKIPPED) in reporter.events
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


def test_staging_is_removed_even_when_the_track_fails(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    downloader = FakeDownloader(fail=True)
    run_batch(
        [REF],
        settings=settings,
        downloader=downloader,
        reporter=RecordingReporter(),
        archive=Archive(tmp_path / "a.txt"),
        encode=fake_encode,
    )
    # The directory has to have existed for its removal to mean anything.
    assert downloader.staging_was_dir == [True]
    assert downloader.staging_seen["aaa"] == settings.staging_root / "aaa"
    assert not (settings.staging_root / "aaa").exists()


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
    assert not (settings.staging_root / "aaa").exists()


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
    assert not (settings.staging_root / "aaa").exists()


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
