from __future__ import annotations

import argparse
import dataclasses
import io
import os
import sys
import threading
from pathlib import Path
from typing import TextIO

import pytest

from memill import cli
from memill.cli import build_parser, collect_urls, render_summary, settings_from_args
from memill.config import CbrQuality, VbrQuality
from memill.errors import DependencyError, DownloadError, TransferError
from memill.pipeline import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    BatchResult,
    TrackOutcome,
)
from memill.source import TrackRef
from memill.transfer import ARCHIVE_FILENAME

REF = TrackRef("aaa", "https://x/aaa", "Some Mix", 10.0)


def parse(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def fake_downloader(
    refs: list[TrackRef],
    fetches: list[object],
    *,
    expand_error: Exception | None = None,
) -> type:
    """A stand-in for ``cli.Downloader`` that records every fetch attempted."""

    class _Downloader:
        def __init__(self, *_: object, **__: object) -> None: ...

        def expand(self, urls: list[str]) -> list[TrackRef]:
            if expand_error is not None:
                raise expand_error
            return list(refs)

        def fetch(self, *args: object, **__: object) -> None:
            fetches.append(args)
            raise AssertionError("this run must not download")

    return _Downloader


class RecordingReporter:
    """Counts close() calls; otherwise does nothing the pipeline can notice."""

    def __init__(self) -> None:
        self.closed = 0
        # One entry per select_reporter call, so a test can tell "never
        # opened" apart from "opened and leaked" -- close() counts cannot.
        self.opened: list[bool] = []

    def batch_started(self, total: int) -> None: ...
    def track_started(self, key: str, label: str) -> None: ...
    def track_phase(self, key: str, phase: str) -> None: ...
    def track_progress(self, key: str, fraction: float) -> None: ...
    def track_finished(
        self, key: str, status: str, detail: str | None = None
    ) -> None: ...

    def close(self) -> None:
        self.closed += 1


class SwallowingReporter(RecordingReporter):
    """Mimics RichReporter: replaces sys.stderr while live, restores on close.

    This is the behaviour that makes ordering load-bearing -- anything printed
    before close() lands in ``sink`` and the user never sees it.
    """

    def __init__(self, real: TextIO) -> None:
        super().__init__()
        self.real = real
        self.sink = io.StringIO()
        sys.stderr = self.sink

    def close(self) -> None:
        super().close()
        sys.stderr = self.real


def outcome(
    status: str, title: str = "A Track", error: str | None = None
) -> TrackOutcome:
    ref = TrackRef(title, f"https://x/{title}", title, None)
    return TrackOutcome(ref, status, error=error)


def install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    refs: list[TrackRef] | None = None,
    fetches: list[object] | None = None,
    expand_error: Exception | None = None,
) -> RecordingReporter:
    """Wire main() up to fakes: no ffmpeg, no network, no live display."""
    reporter = RecordingReporter()
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(
        cli,
        "Downloader",
        fake_downloader(
            [REF] if refs is None else refs,
            [] if fetches is None else fetches,
            expand_error=expand_error,
        ),
    )

    def select(*, stream: TextIO, force_plain: bool = False) -> RecordingReporter:
        reporter.opened.append(force_plain)
        return reporter

    monkeypatch.setattr(cli, "select_reporter", select)
    return reporter


# --- argument parsing and Settings construction -----------------------------


def test_default_quality_is_vbr_zero() -> None:
    settings = settings_from_args(parse("https://x/1"), cpu_count=8)
    assert settings.quality == VbrQuality(0)


def test_bitrate_switches_to_constant_bitrate() -> None:
    settings = settings_from_args(parse("https://x/1", "-b", "320k"), cpu_count=8)
    assert settings.quality == CbrQuality("320k")


def test_explicit_quality_level_is_honoured() -> None:
    settings = settings_from_args(parse("https://x/1", "-q", "5"), cpu_count=8)
    assert settings.quality == VbrQuality(5)


def test_quality_and_bitrate_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse("https://x/1", "-q", "2", "-b", "320k")


def test_the_default_destination_is_portable_not_personal() -> None:
    """A published tool cannot assume anyone's mount points.

    This replaced an assertion pinning one machine's Windows mount. Shipping
    that default meant every other user's first run wrote somewhere that does
    not exist on their system.
    """
    settings = settings_from_args(parse("https://x/1"), cpu_count=8)
    assert settings.destination == Path.home() / "Music"


def test_the_output_env_var_overrides_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMILL_OUTPUT", "/tmp/some/library")
    settings = settings_from_args(parse("https://x/1"), cpu_count=8)
    assert settings.destination == Path("/tmp/some/library")


def test_an_explicit_output_still_beats_the_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MEMILL_OUTPUT", "/tmp/ignored")
    args = parse("https://x/1", "-o", str(tmp_path))
    settings = settings_from_args(args, cpu_count=8)
    assert settings.destination == tmp_path



def test_output_flag_overrides_the_destination() -> None:
    settings = settings_from_args(parse("https://x/1", "-o", "/tmp/music"), cpu_count=8)
    assert settings.destination == Path("/tmp/music")


def test_a_quoted_tilde_destination_is_expanded() -> None:
    """The shell expands an unquoted ``~`` and leaves a quoted one alone.

    ``-o "~/music"`` therefore reaches argparse as a literal tilde, and
    without expansion the run creates a directory actually named ``~`` in the
    working directory and files the library inside it.
    """
    settings = settings_from_args(parse("https://x/1", "-o", "~/music"), cpu_count=8)
    assert settings.destination == Path.home() / "music"
    assert "~" not in str(settings.destination)


def test_jobs_and_fragments_hold_the_connection_budget() -> None:
    settings = settings_from_args(parse("https://x/1", "-j", "2"), cpu_count=8)
    assert (settings.jobs, settings.fragments) == (2, 4)


def test_positive_flags_default_to_off_and_features_to_on() -> None:
    settings = settings_from_args(parse("https://x/1"), cpu_count=8)
    assert settings.embed_cover
    assert settings.clean_titles
    assert settings.use_archive
    assert not settings.keep_source
    assert not settings.normalize
    assert not settings.dry_run
    assert settings.cookies_from_browser is None


def test_negative_flags_turn_features_off() -> None:
    args = parse(
        "https://x/1", "--no-cover", "--raw-title", "--no-archive", "--keep-source"
    )
    settings = settings_from_args(args, cpu_count=8)
    assert not settings.embed_cover
    assert not settings.clean_titles
    assert not settings.use_archive
    assert settings.keep_source


def test_normalize_and_cookies_reach_the_settings() -> None:
    args = parse("https://x/1", "--normalize", "--cookies-from-browser", "firefox")
    settings = settings_from_args(args, cpu_count=8)
    assert settings.normalize
    assert settings.cookies_from_browser == "firefox"


def usage_error(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    """Assert argv is refused at the parser: usage line, exit 2, no traceback."""
    with pytest.raises(SystemExit) as caught:
        parse(*argv)
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: memill")
    return err


def test_a_quality_level_above_nine_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "-q/--quality" in usage_error(capsys, "https://x/1", "-q", "12")


def test_a_negative_quality_level_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "-q/--quality" in usage_error(capsys, "https://x/1", "-q", "-1")


def test_the_quality_range_boundaries_are_still_accepted() -> None:
    assert settings_from_args(parse("https://x/1", "-q", "0"), cpu_count=8).quality == (
        VbrQuality(0)
    )
    assert settings_from_args(parse("https://x/1", "-q", "9"), cpu_count=8).quality == (
        VbrQuality(9)
    )


def test_a_bitrate_without_its_k_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "-b/--bitrate" in usage_error(capsys, "https://x/1", "-b", "320")


def test_an_empty_bitrate_is_a_usage_error_not_a_silent_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`if args.bitrate` is a truthiness test, so "" would fall through to VBR."""
    assert "-b/--bitrate" in usage_error(capsys, "https://x/1", "-b", "")


def test_an_empty_bitrate_in_a_namespace_raises_rather_than_selecting_vbr() -> None:
    """settings_from_args is public: a caller can reach it without the parser."""
    args = parse("https://x/1")
    args.bitrate = ""
    with pytest.raises(ValueError, match="320k"):
        settings_from_args(args, cpu_count=8)


def test_zero_jobs_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert "-j/--jobs" in usage_error(capsys, "https://x/1", "-j", "0")


def test_a_missing_url_file_is_a_usage_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    missing = tmp_path / "nope" / "urls.txt"
    assert "no such file" in usage_error(capsys, "-", "--from-file", str(missing))


def test_a_fifo_is_accepted_and_readable_as_a_url_file(tmp_path: Path) -> None:
    """`--from-file <(...)` and /dev/stdin are exists() but not is_file()."""
    fifo = tmp_path / "urls.fifo"
    os.mkfifo(fifo)
    writer = threading.Thread(
        target=lambda: fifo.write_text("https://x/2\n", encoding="utf-8"),
        daemon=True,
    )
    writer.start()
    args = parse("--from-file", str(fifo))
    assert collect_urls(args, io.StringIO()) == ["https://x/2"]
    writer.join(timeout=5)


def test_a_directory_as_a_url_file_says_it_is_a_directory(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    err = usage_error(capsys, "-", "--from-file", str(tmp_path))
    assert "is a directory" in err


def test_one_job_is_accepted_because_config_owns_the_floor() -> None:
    """_job_count defers to plan_concurrency, so the two floors cannot drift."""
    settings = settings_from_args(parse("https://x/1", "-j", "1"), cpu_count=8)
    assert (settings.jobs, settings.fragments) == (1, 8)


# --- URL collection ---------------------------------------------------------


def test_urls_come_from_arguments_a_file_and_stdin(tmp_path: Path) -> None:
    listing = tmp_path / "urls.txt"
    listing.write_text("https://x/2\n\n# a comment\nhttps://x/3\n", encoding="utf-8")
    args = parse("https://x/1", "-", "--from-file", str(listing))
    urls = collect_urls(args, io.StringIO("https://x/4\n"))
    assert urls == ["https://x/1", "https://x/2", "https://x/3", "https://x/4"]


def test_stdin_is_read_only_when_a_dash_is_given() -> None:
    args = parse("https://x/1")
    assert collect_urls(args, io.StringIO("https://x/9\n")) == ["https://x/1"]


def test_a_bare_dash_reads_stdin_and_is_not_itself_a_url() -> None:
    urls = collect_urls(parse("-"), io.StringIO("https://x/9\n"))
    assert urls == ["https://x/9"]


def test_file_and_stdin_lines_are_stripped_of_blanks_and_comments(
    tmp_path: Path,
) -> None:
    listing = tmp_path / "urls.txt"
    listing.write_text("  https://x/2  \n\n#c\n", encoding="utf-8")
    args = parse("-", "--from-file", str(listing))
    urls = collect_urls(args, io.StringIO("\n # skip\n  https://x/3\n"))
    assert urls == ["https://x/2", "https://x/3"]


# --- summary ----------------------------------------------------------------


def test_summary_counts_each_status_separately() -> None:
    result = BatchResult(
        (
            outcome(STATUS_DONE, "one"),
            outcome(STATUS_DONE, "two"),
            outcome(STATUS_SKIPPED, "three"),
            outcome(STATUS_FAILED, "four", error="boom"),
        )
    )
    stream = io.StringIO()
    render_summary(result, stream)
    assert "2 done, 1 skipped, 1 failed" in stream.getvalue()


def test_summary_names_every_failure_and_its_reason() -> None:
    result = BatchResult(
        (outcome(STATUS_FAILED, "one", error="404"), outcome(STATUS_DONE, "two"))
    )
    stream = io.StringIO()
    render_summary(result, stream)
    text = stream.getvalue()
    assert "  failed: one - 404\n" in text
    assert "two" not in text.split("\n")[-2]


def test_an_unrecognised_status_fails_loudly_rather_than_vanishing() -> None:
    """A status added later must not be counted into a key nothing prints."""
    with pytest.raises(KeyError):
        render_summary(BatchResult((outcome("cancelled"),)), io.StringIO())


def test_summary_of_an_empty_batch_is_all_zeroes() -> None:
    stream = io.StringIO()
    render_summary(BatchResult(), stream)
    assert "0 done, 0 skipped, 0 failed" in stream.getvalue()


# --- main: exit codes -------------------------------------------------------


def test_missing_ffmpeg_exits_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom() -> None:
        raise DependencyError("ffmpeg is not on PATH")

    monkeypatch.setattr(cli, "require_ffmpeg", boom)
    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 2


def test_a_clean_run_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch)
    monkeypatch.setattr(
        cli, "run_batch", lambda *a, **k: BatchResult((outcome(STATUS_DONE),))
    )
    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 0


def test_a_failed_track_exits_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_batch",
        lambda *a, **k: BatchResult((outcome(STATUS_DONE), outcome(STATUS_FAILED))),
    )
    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 1


def test_an_interrupt_exits_one_hundred_and_thirty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch)

    def boom(*_: object, **__: object) -> BatchResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_batch", boom)
    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 130


def test_no_urls_at_all_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    with pytest.raises(SystemExit) as caught:
        cli.main(["-o", str(tmp_path)])
    assert caught.value.code == 2


def test_an_empty_expansion_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    install(monkeypatch, refs=[])
    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 1
    assert "nothing to download" in capsys.readouterr().err


def test_a_failed_expansion_reports_one_line_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    install(monkeypatch, expand_error=DownloadError("could not resolve https://x/aaa"))
    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 1
    assert capsys.readouterr().err == "error: could not resolve https://x/aaa\n"


def test_a_foreign_expansion_error_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    install(monkeypatch, expand_error=RuntimeError("could not read cookies"))
    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 1
    assert "could not read cookies" in capsys.readouterr().err


# --- main: run-level aborts out of run_batch --------------------------------


def test_a_run_level_abort_reports_one_line_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A mistyped --cookies-from-browser reaches main as a foreign exception."""
    install(monkeypatch)

    def boom(*_: object, **__: object) -> BatchResult:
        raise RuntimeError("could not load cookies from fierfox")

    monkeypatch.setattr(cli, "run_batch", boom)
    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 1
    assert capsys.readouterr().err == (
        "error: unexpected RuntimeError: could not load cookies from fierfox\n"
    )


def test_our_own_error_out_of_run_batch_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    install(monkeypatch)

    def boom(*_: object, **__: object) -> BatchResult:
        raise DownloadError("the run was cancelled")

    monkeypatch.setattr(cli, "run_batch", boom)
    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 1
    assert capsys.readouterr().err == "error: the run was cancelled\n"


def test_a_run_level_abort_still_closes_the_reporter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reporter = install(monkeypatch)

    def boom(*_: object, **__: object) -> BatchResult:
        raise RuntimeError("nope")

    monkeypatch.setattr(cli, "run_batch", boom)
    cli.main(["https://x/aaa", "-o", str(tmp_path)])
    assert reporter.closed == 1


def test_an_unreadable_archive_reports_one_line_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The archive lives in the library on an NTFS mount; bad bytes happen."""
    reporter = install(monkeypatch)
    (tmp_path / ARCHIVE_FILENAME).write_bytes(b"\xff\xfe not utf-8\n")
    monkeypatch.setattr(cli, "run_batch", lambda *a, **k: BatchResult())

    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"error: could not read the archive {tmp_path}")
    # Not `closed == 0`, which cannot tell "never opened" from "opened and
    # leaked": under a real RichReporter the leaked variant returns 1 with the
    # live display still installed and sys.stderr still a proxy.
    assert reporter.opened == []


def test_an_expected_failure_out_of_run_batch_is_not_labelled_a_bug(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """errors.py's contract: a Yt2Mp3Error was anticipated, anything else is a bug."""
    install(monkeypatch)

    def boom(*_: object, **__: object) -> BatchResult:
        raise TransferError("read-only mount")

    monkeypatch.setattr(cli, "run_batch", boom)
    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err == "error: read-only mount\n"


# --- main: nothing may be printed before the reporter is closed -------------


def swallowing(
    monkeypatch: pytest.MonkeyPatch, made: list[SwallowingReporter]
) -> None:
    """Replace select_reporter with one that hides stderr until it is closed."""
    # Registers an undo, so a failing assertion cannot leave stderr swapped out.
    monkeypatch.setattr(sys, "stderr", sys.stderr)

    def select(*, stream: TextIO, force_plain: bool = False) -> SwallowingReporter:
        reporter = SwallowingReporter(stream)
        made.append(reporter)
        return reporter

    monkeypatch.setattr(cli, "select_reporter", select)


def test_the_interrupt_message_reaches_the_user_not_the_live_display(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    install(monkeypatch)
    made: list[SwallowingReporter] = []
    swallowing(monkeypatch, made)

    def boom(*_: object, **__: object) -> BatchResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_batch", boom)

    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 130
    assert made[0].closed == 1
    assert made[0].sink.getvalue() == ""
    assert "interrupted" in capsys.readouterr().err


def test_the_abort_message_reaches_the_user_not_the_live_display(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    install(monkeypatch)
    made: list[SwallowingReporter] = []
    swallowing(monkeypatch, made)

    def boom(*_: object, **__: object) -> BatchResult:
        raise RuntimeError("cookie store unreadable")

    monkeypatch.setattr(cli, "run_batch", boom)

    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 1
    assert made[0].sink.getvalue() == ""
    assert "cookie store unreadable" in capsys.readouterr().err


def test_the_summary_reaches_the_user_not_the_live_display(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    install(monkeypatch)
    made: list[SwallowingReporter] = []
    swallowing(monkeypatch, made)
    monkeypatch.setattr(
        cli, "run_batch", lambda *a, **k: BatchResult((outcome(STATUS_DONE),))
    )

    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 0
    assert made[0].sink.getvalue() == ""
    assert "1 done, 0 skipped, 0 failed" in capsys.readouterr().err


# --- main: the dry run ------------------------------------------------------


def test_dry_run_lists_tracks_and_downloads_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fetches: list[object] = []
    install(monkeypatch, fetches=fetches)
    assert cli.main(["https://x/aaa", "--dry-run", "-o", str(tmp_path)]) == 0
    assert "Some Mix" in capsys.readouterr().out
    assert fetches == []


def test_dry_run_never_starts_a_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started: list[object] = []
    reporter = install(monkeypatch)
    monkeypatch.setattr(cli, "run_batch", lambda *a, **k: started.append(a))
    assert cli.main(["https://x/aaa", "--dry-run", "-o", str(tmp_path)]) == 0
    assert started == []
    assert reporter.closed == 0


def test_the_dry_run_branch_reads_the_settings_field_not_the_namespace(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """--dry-run is absent, yet settings.dry_run is on: the branch must fire."""
    fetches: list[object] = []
    install(monkeypatch, fetches=fetches)
    real = cli.settings_from_args

    def forced(args: argparse.Namespace, *, cpu_count: int) -> object:
        return dataclasses.replace(real(args, cpu_count=cpu_count), dry_run=True)

    monkeypatch.setattr(cli, "settings_from_args", forced)
    assert cli.main(["https://x/aaa", "-o", str(tmp_path)]) == 0
    assert "Some Mix" in capsys.readouterr().out
    assert fetches == []


# --- main: wiring -----------------------------------------------------------


def test_the_plain_flag_forces_the_plain_reporter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "Downloader", fake_downloader([REF], []))
    monkeypatch.setattr(cli, "run_batch", lambda *a, **k: BatchResult())

    def select(*, stream: TextIO, force_plain: bool = False) -> RecordingReporter:
        seen.append(force_plain)
        return RecordingReporter()

    monkeypatch.setattr(cli, "select_reporter", select)
    cli.main(["https://x/aaa", "-o", str(tmp_path), "--plain"])
    cli.main(["https://x/aaa", "-o", str(tmp_path)])
    assert seen == [True, False]


def test_run_batch_receives_the_settings_and_the_expanded_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    install(monkeypatch, refs=[REF])

    def record(refs: object, **kwargs: object) -> BatchResult:
        captured["refs"] = refs
        captured.update(kwargs)
        return BatchResult()

    monkeypatch.setattr(cli, "run_batch", record)
    cli.main(["https://x/aaa", "-o", str(tmp_path), "-j", "3"])
    assert captured["refs"] == [REF]
    settings = captured["settings"]
    assert isinstance(settings, cli.Settings)
    assert (settings.jobs, settings.destination) == (3, tmp_path)


def test_the_reporter_is_closed_exactly_once_on_a_clean_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reporter = install(monkeypatch)
    monkeypatch.setattr(cli, "run_batch", lambda *a, **k: BatchResult())
    cli.main(["https://x/aaa", "-o", str(tmp_path)])
    assert reporter.closed == 1
