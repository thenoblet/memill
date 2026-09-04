from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from typing import Any

import pytest
from yt_dlp import YoutubeDL
from yt_dlp.cookies import CookieLoadError
from yt_dlp.utils import DownloadCancelled, ExtractorError, YoutubeDLError
from yt_dlp.utils import DownloadError as YtDlpDownloadError

from memill import source
from memill.config import Settings, VbrQuality
from memill.errors import DownloadError
from memill.source import Downloader, TrackRef


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "destination": tmp_path / "library",
        "staging_root": tmp_path / "stage",
        "quality": VbrQuality(0),
        "jobs": 1,
        "fragments": 8,
    }
    return Settings(**{**base, **overrides})


def fake_factory(info: dict[str, Any], captured: list[dict[str, Any]]):
    """Return a ydl_factory that records its options and yields fixed info."""

    class FakeYDL:
        def extract_info(self, url: str, download: bool = False) -> dict[str, Any]:
            return info

    @contextmanager
    def factory(opts: dict[str, Any]) -> Iterator[FakeYDL]:
        captured.append(opts)
        yield FakeYDL()

    return factory


def test_expand_flattens_a_playlist(tmp_path: Path) -> None:
    info = {
        "entries": [
            {"id": "aaa", "title": "One", "duration": 120, "url": "https://x/aaa"},
            {"id": "bbb", "title": "Two", "duration": None, "url": "https://x/bbb"},
        ]
    }
    downloader = Downloader(make_settings(tmp_path), ydl_factory=fake_factory(info, []))
    refs = downloader.expand(["https://x/playlist"])
    assert [ref.video_id for ref in refs] == ["aaa", "bbb"]
    assert refs[0].duration == 120.0


def test_expand_handles_a_single_video(tmp_path: Path) -> None:
    info = {"id": "aaa", "title": "One", "duration": 10, "webpage_url": "https://x/aaa"}
    downloader = Downloader(make_settings(tmp_path), ydl_factory=fake_factory(info, []))
    assert downloader.expand(["https://x/aaa"]) == [
        TrackRef(video_id="aaa", url="https://x/aaa", title="One", duration=10.0)
    ]


def test_expand_skips_unavailable_entries(tmp_path: Path) -> None:
    info = {"entries": [None, {"id": "bbb", "title": "Two", "url": "https://x/bbb"}]}
    downloader = Downloader(make_settings(tmp_path), ydl_factory=fake_factory(info, []))
    assert [ref.video_id for ref in downloader.expand(["https://x/p"])] == ["bbb"]


def test_expand_skips_playlist_entries(tmp_path: Path) -> None:
    """A channel URL's top-level entries are playlist tabs, not videos.

    Under ``extract_flat="in_playlist"`` yt-dlp returns one entry per tab --
    Videos, Shorts, Live -- each carrying an id and a title, so an unfiltered
    ``_ref_from_entry`` turns every tab into a TrackRef whose fetch then dies
    with "yt-dlp did not report a downloaded file", a message that never
    mentions the channel URL that caused it.

    Written from yt-dlp's ``_type`` contract rather than from an observed
    response: the seam this suite fakes sits above the network, so the real
    shape of a channel extraction cannot be proven here.
    """
    info = {
        "entries": [
            {"_type": "playlist", "id": "UCabc-videos", "title": "Chan - Videos"},
            {"_type": "url", "id": "bbb", "title": "Two", "url": "https://x/bbb"},
        ]
    }
    downloader = Downloader(make_settings(tmp_path), ydl_factory=fake_factory(info, []))
    assert [ref.video_id for ref in downloader.expand(["https://x/chan"])] == ["bbb"]


def test_fetch_returns_the_path_yt_dlp_reported(tmp_path: Path) -> None:
    staging = tmp_path / "stage" / "aaa"
    staging.mkdir(parents=True)
    audio = staging / "aaa.opus"
    audio.write_bytes(b"x")
    info = {"id": "aaa", "requested_downloads": [{"filepath": str(audio)}]}
    downloader = Downloader(make_settings(tmp_path), ydl_factory=fake_factory(info, []))
    media = downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", 10.0), staging)
    assert media.audio == audio
    assert media.cover is None
    assert media.info["id"] == "aaa"


def test_fetch_finds_the_thumbnail_next_to_the_audio(tmp_path: Path) -> None:
    staging = tmp_path / "stage" / "aaa"
    staging.mkdir(parents=True)
    (staging / "aaa.opus").write_bytes(b"x")
    (staging / "aaa.webp").write_bytes(b"img")
    info = {
        "id": "aaa",
        "requested_downloads": [{"filepath": str(staging / "aaa.opus")}],
    }
    downloader = Downloader(make_settings(tmp_path), ydl_factory=fake_factory(info, []))
    media = downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", 10.0), staging)
    assert media.cover == staging / "aaa.webp"


def test_fetch_without_a_reported_file_is_an_error(tmp_path: Path) -> None:
    downloader = Downloader(
        make_settings(tmp_path), ydl_factory=fake_factory({"id": "aaa"}, [])
    )
    with pytest.raises(DownloadError, match="did not report"):
        downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", None), tmp_path)


def test_download_options_prefer_audio_only_and_carry_the_fragment_budget(
    tmp_path: Path,
) -> None:
    captured: list[dict[str, Any]] = []
    staging = tmp_path / "stage" / "aaa"
    staging.mkdir(parents=True)
    (staging / "aaa.opus").write_bytes(b"x")
    info = {
        "id": "aaa",
        "requested_downloads": [{"filepath": str(staging / "aaa.opus")}],
    }
    downloader = Downloader(
        make_settings(tmp_path, fragments=4), ydl_factory=fake_factory(info, captured)
    )
    downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", None), staging)
    opts = captured[-1]
    # "bestaudio" first, so YouTube's audio-only stream is what is fetched and
    # the video track -- far larger than the audio it carries -- is never
    # downloaded. The "/best" fallback is muxed, and is deliberate: a source
    # publishing no audio-only format at all would otherwise fail outright
    # rather than cost a bigger download. Both halves are documented in the
    # Downloader docstring and the README, so both are pinned here.
    assert opts["format"] == "bestaudio/best"
    assert opts["concurrent_fragment_downloads"] == 4
    assert opts["noplaylist"] is True
    # ffmpeg is ours: yt-dlp must hand us the raw stream, unprocessed.
    assert opts["postprocessors"] == []
    assert opts["outtmpl"] == {"default": str(staging / "%(id)s.%(ext)s")}
    # Silence is not cosmetic: yt-dlp's own progress bar would be drawn
    # straight over the Rich live display.
    assert opts["quiet"] is True
    assert opts["noprogress"] is True
    # Without this yt-dlp never writes a thumbnail, so _find_cover finds
    # nothing and no cover art is ever embedded -- silently.
    assert opts["writethumbnail"] is True
    assert opts["http_chunk_size"] == 10 << 20
    assert opts["no_warnings"] is True
    assert opts["retries"] == 5
    assert opts["fragment_retries"] == 5
    assert opts["socket_timeout"] == 20


# The keys a real YoutubeDL writes into the options dict it is handed.
YT_DLP_INJECTED_KEYS = (
    "compat_opts",
    "forceprint",
    "http_headers",
    "js_runtimes",
    "outtmpl",
    "print_to_file",
    "remote_components",
)


class SpySession:
    """A fake yt-dlp session that records how the adapter drove it.

    Deliberately a hand-written context manager rather than a
    ``@contextmanager`` generator: a generator's ``finally`` also runs when the
    interpreter collects it, so ``closed`` would go True even for an adapter
    that entered the session and never exited it -- and this fake exists
    precisely to catch that.
    """

    def __init__(
        self, opts: dict[str, Any], info: Any, calls: list[dict[str, Any]]
    ) -> None:
        self.opts = opts
        # A real YoutubeDL writes its own defaults into the options dict it is
        # handed, so the fake does too -- otherwise no test could tell a fresh
        # copy from a copy of the previous session's polluted one. ``received``
        # is the snapshot taken before that happens.
        self.received = dict(opts)
        for key in YT_DLP_INJECTED_KEYS:
            opts.setdefault(key, "<default written in by yt-dlp>")
        self.closed = False
        self._info = info
        self._calls = calls

    def __enter__(self) -> SpySession:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.closed = True
        return False

    def extract_info(self, url: str, download: bool = False) -> Any:
        self._calls.append({"url": url, "download": download})
        return self._info(url) if callable(self._info) else self._info


class SpyFactory:
    """A ydl_factory that keeps every session it hands out, open or closed."""

    def __init__(self, info: Any) -> None:
        self._info = info
        self.calls: list[dict[str, Any]] = []
        self.sessions: list[SpySession] = []

    def __call__(self, opts: dict[str, Any]) -> SpySession:
        session = SpySession(opts, self._info, self.calls)
        self.sessions.append(session)
        return session


def without_logger(opts: dict[str, Any]) -> dict[str, Any]:
    """The options minus the recorder, which is a fresh object every time."""
    return {key: value for key, value in opts.items() if key != "logger"}


def staged_audio(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """A staging directory holding one downloaded file, plus yt-dlp's report."""
    staging = tmp_path / "stage" / "aaa"
    staging.mkdir(parents=True)
    audio = staging / "aaa.opus"
    audio.write_bytes(b"x")
    return staging, {"id": "aaa", "requested_downloads": [{"filepath": str(audio)}]}


def test_expand_drops_entries_without_a_usable_id(tmp_path: Path) -> None:
    info = {
        "entries": [
            {"title": "Deleted video"},
            {"id": "", "title": "Private video"},
            {"id": None, "title": "Unavailable"},
            {"id": "bbb", "title": "Two", "url": "https://x/bbb"},
        ]
    }
    downloader = Downloader(make_settings(tmp_path), ydl_factory=fake_factory(info, []))
    assert [ref.video_id for ref in downloader.expand(["https://x/p"])] == ["bbb"]


def test_expand_extracts_flat_and_downloads_nothing(tmp_path: Path) -> None:
    def one_video(url: str) -> dict[str, Any]:
        return {"id": url.rsplit("/", 1)[-1], "title": "One", "webpage_url": url}

    spy = SpyFactory(one_video)
    downloader = Downloader(make_settings(tmp_path), ydl_factory=spy)

    refs = downloader.expand(["https://x/a", "https://x/b"])

    # Every URL's tracks accumulate; the last one does not replace the rest.
    assert [ref.video_id for ref in refs] == ["a", "b"]
    assert [call["url"] for call in spy.calls] == ["https://x/a", "https://x/b"]
    assert [call["download"] for call in spy.calls] == [False, False]
    assert [session.closed for session in spy.sessions] == [True, True]

    first = spy.sessions[0].received
    assert first["extract_flat"] == "in_playlist"
    assert first["skip_download"] is True
    # Silence matters at this call site too: yt-dlp's own output would be drawn
    # over the Rich live display while a playlist is being expanded.
    assert first["quiet"] is True
    assert first["no_warnings"] is True

    # Each session gets a pristine copy of the options -- not the same dict,
    # and not a copy of the previous session's, which yt-dlp has by then
    # written its own defaults into.
    assert spy.sessions[0].opts is not spy.sessions[1].opts
    assert without_logger(spy.sessions[1].received) == without_logger(first)
    assert not set(spy.sessions[1].received) & set(YT_DLP_INJECTED_KEYS)
    # The recorder is the one option that must NOT be shared between the two:
    # it holds the last complaint yt-dlp made, and the second URL's failure
    # must not be reported against the first.
    assert first["logger"] is not spy.sessions[1].received["logger"]


def test_expand_prefers_the_canonical_url_and_synthesizes_a_missing_one(
    tmp_path: Path,
) -> None:
    info = {
        "entries": [
            {
                "id": "aaa",
                "title": "One",
                "webpage_url": "https://x/canonical",
                "url": "https://x/media-fragment",
            },
            {"id": "bbb"},
        ]
    }
    downloader = Downloader(make_settings(tmp_path), ydl_factory=fake_factory(info, []))

    first, second = downloader.expand(["https://x/p"])

    assert first.url == "https://x/canonical"
    assert second.url == "https://www.youtube.com/watch?v=bbb"
    assert second.title == "bbb"


def test_expand_coerces_a_duration_to_a_float_or_none(tmp_path: Path) -> None:
    info = {
        "entries": [
            {"id": "aaa", "title": "One", "duration": 213},
            {"id": "bbb", "title": "Two", "duration": "3:33"},
        ]
    }
    downloader = Downloader(make_settings(tmp_path), ydl_factory=fake_factory(info, []))

    first, second = downloader.expand(["https://x/p"])

    assert isinstance(first.duration, float)
    assert first.duration == 213.0
    assert second.duration is None


def test_fetch_downloads_and_closes_the_session(tmp_path: Path) -> None:
    staging, info = staged_audio(tmp_path)
    spy = SpyFactory(info)
    downloader = Downloader(make_settings(tmp_path), ydl_factory=spy)

    downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", None), staging)

    assert spy.calls == [{"url": "https://x/aaa", "download": True}]
    assert spy.sessions[0].closed is True


def test_fetch_finds_a_jpeg_thumbnail_too(tmp_path: Path) -> None:
    staging, info = staged_audio(tmp_path)
    (staging / "aaa.jpg").write_bytes(b"img")
    downloader = Downloader(make_settings(tmp_path), ydl_factory=fake_factory(info, []))

    media = downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", None), staging)

    assert media.cover == staging / "aaa.jpg"


def test_fetch_ignores_a_thumbnail_when_cover_embedding_is_off(tmp_path: Path) -> None:
    staging, info = staged_audio(tmp_path)
    (staging / "aaa.webp").write_bytes(b"img")
    spy = SpyFactory(info)
    downloader = Downloader(
        make_settings(tmp_path, embed_cover=False), ydl_factory=spy
    )

    media = downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", None), staging)

    assert media.cover is None
    assert spy.sessions[0].opts["writethumbnail"] is False


def test_fetch_without_metadata_is_an_error(tmp_path: Path) -> None:
    downloader = Downloader(make_settings(tmp_path), ydl_factory=SpyFactory(None))
    with pytest.raises(DownloadError, match="no metadata"):
        downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", None), tmp_path)


def test_browser_cookies_are_passed_only_when_configured(tmp_path: Path) -> None:
    staging, info = staged_audio(tmp_path)
    ref = TrackRef("aaa", "https://x/aaa", "One", None)

    plain = SpyFactory(info)
    Downloader(make_settings(tmp_path), ydl_factory=plain).fetch(ref, staging)
    assert "cookiesfrombrowser" not in plain.sessions[0].opts

    with_cookies = SpyFactory(info)
    Downloader(
        make_settings(tmp_path, cookies_from_browser="firefox"),
        ydl_factory=with_cookies,
    ).fetch(ref, staging)
    assert with_cookies.sessions[0].opts["cookiesfrombrowser"] == ("firefox",)


def test_expand_passes_browser_cookies_only_when_configured(tmp_path: Path) -> None:
    """Resolution runs before any download, so the cookies must reach it too.

    An age-restricted or members-only URL fails at expansion otherwise, which
    is exactly the content ``--cookies-from-browser`` exists to unlock.
    """
    info = {"id": "aaa", "title": "One", "webpage_url": "https://x/aaa"}

    plain = SpyFactory(info)
    Downloader(make_settings(tmp_path), ydl_factory=plain).expand(["https://x/aaa"])
    assert "cookiesfrombrowser" not in plain.sessions[0].received

    with_cookies = SpyFactory(info)
    Downloader(
        make_settings(tmp_path, cookies_from_browser="firefox"),
        ydl_factory=with_cookies,
    ).expand(["https://x/aaa"])
    assert with_cookies.sessions[0].received["cookiesfrombrowser"] == ("firefox",)


def test_the_progress_hook_reports_the_fraction_downloaded(tmp_path: Path) -> None:
    staging, info = staged_audio(tmp_path)
    spy = SpyFactory(info)
    seen: list[float] = []
    downloader = Downloader(make_settings(tmp_path), ydl_factory=spy)

    downloader.fetch(
        TrackRef("aaa", "https://x/aaa", "One", None), staging, on_progress=seen.append
    )
    hook = spy.sessions[0].opts["progress_hooks"][0]
    hook({"status": "downloading", "downloaded_bytes": 25, "total_bytes": 100})
    hook({"status": "downloading", "downloaded_bytes": 5, "total_bytes_estimate": 10})
    hook({"status": "downloading", "downloaded_bytes": 120, "total_bytes": 100})
    hook({"status": "finished", "downloaded_bytes": 100, "total_bytes": 100})
    hook({"status": "downloading", "downloaded_bytes": 1, "total_bytes": 0})
    hook({"status": "downloading"})

    assert seen == [0.25, 0.5, 1.0]


def test_no_progress_hook_is_installed_without_a_callback(tmp_path: Path) -> None:
    staging, info = staged_audio(tmp_path)
    spy = SpyFactory(info)

    Downloader(make_settings(tmp_path), ydl_factory=spy).fetch(
        TrackRef("aaa", "https://x/aaa", "One", None), staging
    )

    assert "progress_hooks" not in spy.sessions[0].opts


def test_the_default_factory_drives_youtube_dl_as_a_context_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No factory supplied: the real seam must build and close a YoutubeDL."""
    built: list[FakeYoutubeDL] = []

    class FakeYoutubeDL:
        def __init__(self, opts: dict[str, Any]) -> None:
            self.opts = opts
            self.closed = False
            built.append(self)

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, *exc: object) -> bool:
            self.closed = True
            return False

        def extract_info(self, url: str, download: bool = False) -> dict[str, Any]:
            return {"id": "aaa", "title": "One", "webpage_url": url}

    monkeypatch.setattr(source, "YoutubeDL", FakeYoutubeDL)

    refs = Downloader(make_settings(tmp_path)).expand(["https://x/aaa"])

    assert [ref.video_id for ref in refs] == ["aaa"]
    assert built[0].opts["extract_flat"] == "in_playlist"
    assert built[0].closed is True


def exploding_factory(error: BaseException):
    """A ydl_factory whose session raises the given error when driven."""

    class ExplodingSession:
        def __enter__(self) -> ExplodingSession:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def extract_info(self, url: str, download: bool = False) -> Any:
            raise error

    def factory(opts: dict[str, Any]) -> ExplodingSession:
        return ExplodingSession()

    return factory


def test_a_yt_dlp_failure_in_expand_becomes_our_download_error(tmp_path: Path) -> None:
    """yt-dlp's DownloadError shares our name but is not our exception.

    Unwrapped it sails past the pipeline's ``Yt2Mp3Error`` handler, so one
    private video would kill the whole batch instead of failing one track.
    """
    boom = YtDlpDownloadError("Private video")
    assert not isinstance(boom, DownloadError)
    downloader = Downloader(
        make_settings(tmp_path), ydl_factory=exploding_factory(boom)
    )

    with pytest.raises(DownloadError) as excinfo:
        downloader.expand(["https://x/p"])

    assert "https://x/p" in str(excinfo.value)
    assert excinfo.value.__cause__ is boom


def test_a_yt_dlp_failure_in_fetch_becomes_our_download_error(tmp_path: Path) -> None:
    boom = YtDlpDownloadError("Video unavailable")
    downloader = Downloader(
        make_settings(tmp_path), ydl_factory=exploding_factory(boom)
    )

    with pytest.raises(DownloadError) as excinfo:
        downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", None), tmp_path)

    assert "https://x/aaa" in str(excinfo.value)
    assert excinfo.value.__cause__ is boom


def test_every_yt_dlp_error_is_translated_not_just_download_error(
    tmp_path: Path,
) -> None:
    """``YoutubeDLError`` is the base: extractor and geo failures count too."""
    boom = ExtractorError("Unable to extract player response")
    downloader = Downloader(
        make_settings(tmp_path), ydl_factory=exploding_factory(boom)
    )

    with pytest.raises(DownloadError):
        downloader.expand(["https://x/p"])


def test_a_run_level_abort_is_not_demoted_to_a_per_track_failure(
    tmp_path: Path,
) -> None:
    """CookieLoadError and DownloadCancelled must escape both call sites whole.

    They are ``YoutubeDLError`` subclasses, so the general handler would wrap
    them; yt-dlp re-raises them unconverted for the same reason we must. One
    unreadable cookie store should stop the run once with a clear message, not
    fail every track in the playlist with the same one.
    """
    for boom in (CookieLoadError("could not read the firefox cookie store"),
                 DownloadCancelled("interrupted")):
        assert isinstance(boom, YoutubeDLError)
        downloader = Downloader(
            make_settings(tmp_path), ydl_factory=exploding_factory(boom)
        )

        with pytest.raises(type(boom)) as expanded:
            downloader.expand(["https://x/p"])
        assert expanded.value is boom

        with pytest.raises(type(boom)) as fetched:
            downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", None), tmp_path)
        assert fetched.value is boom


# The exact shape of yt-dlp's last retry failure: FileDownloader.report_retry
# calls report_error("\r[download] Got error: ..."), and report_error prefixes
# "ERROR:" before handing it to trouble.
RETRY_TEXT = "\r[download] Got error: simulated failure. Giving up after 5 retries"
RETRY_FAILURE = f"ERROR: {RETRY_TEXT}"


def download_opts(tmp_path: Path) -> dict[str, Any]:
    """The options ``fetch`` really sends, captured from a completed fetch."""
    captured: list[dict[str, Any]] = []
    staging, info = staged_audio(tmp_path)
    Downloader(make_settings(tmp_path), ydl_factory=fake_factory(info, captured)).fetch(
        TrackRef("aaa", "https://x/aaa", "One", None), staging
    )
    return captured[-1]


def test_yt_dlps_final_error_is_intercepted_instead_of_reaching_stderr(
    tmp_path: Path,
) -> None:
    """``report_error`` consults neither ``quiet`` nor ``no_warnings``.

    So yt-dlp's last retry failure is written to the real stderr regardless,
    and the downloader prefixes that text with a carriage return: it lands at
    column 0 and paints over the line the Rich display had just drawn there.
    Passing a logger in the options is what diverts it.

    Driven through a real ``YoutubeDL``, built from the options ``fetch``
    actually sends and given no URL, because the claim is about yt-dlp's
    behaviour rather than ours -- and a fake would only ever confirm itself.
    """
    opts = download_opts(tmp_path)

    # The control first: without a logger the text really does escape, so the
    # silence below is evidence and not an empty stream by luck.
    escaped = io.StringIO()
    with (
        redirect_stderr(escaped),
        YoutubeDL(without_logger(opts)) as ydl,
        pytest.raises(YtDlpDownloadError),
    ):
        ydl.report_error(RETRY_TEXT)
    assert RETRY_TEXT in escaped.getvalue()

    quiet = io.StringIO()
    with (
        redirect_stderr(quiet),
        YoutubeDL(dict(opts)) as ydl,
        pytest.raises(YtDlpDownloadError),
    ):
        ydl.report_error(RETRY_TEXT)
    assert quiet.getvalue() == ""
    # It was not merely swallowed: the recorder has the reason to hand back.
    assert "simulated failure" in opts["logger"].message


def test_the_recorder_keeps_the_complaints_and_drops_the_chatter(
    tmp_path: Path,
) -> None:
    """``to_screen`` routes every progress line to ``debug``; only the
    complaints are worth retaining, and only the most recent one."""
    recorder = download_opts(tmp_path)["logger"]

    recorder.debug("[download]  51.0% of  412.00MiB at 1.20MiB/s")
    recorder.info("[info] Downloading 1 format(s)")
    assert recorder.message == ""

    recorder.warning("nsig extraction failed")
    assert recorder.message == "nsig extraction failed"
    recorder.error("ERROR: unable to download video data")
    assert recorder.message == "ERROR: unable to download video data"


def logging_then_exploding_factory(logged: str, error: BaseException):
    """A ydl_factory whose session complains through the logger, then raises.

    That is yt-dlp's own order: ``trouble`` writes the message out -- into our
    recorder, once one is installed -- and only afterwards raises.
    """

    class LoggingSession:
        def __init__(self, opts: dict[str, Any]) -> None:
            self._opts = opts

        def __enter__(self) -> LoggingSession:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def extract_info(self, url: str, download: bool = False) -> Any:
            self._opts["logger"].error(logged)
            raise error

    return LoggingSession


def test_a_bare_yt_dlp_error_is_explained_by_what_it_logged(tmp_path: Path) -> None:
    """The exception says nothing, so the recorder has to say it instead.

    Otherwise the user is told "could not download <url>:" and left there,
    with the actual reason having gone to a stderr they cannot read back.
    """
    downloader = Downloader(
        make_settings(tmp_path),
        ydl_factory=logging_then_exploding_factory(
            RETRY_FAILURE, YtDlpDownloadError("")
        ),
    )

    with pytest.raises(DownloadError) as excinfo:
        downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", None), tmp_path)

    assert str(excinfo.value) == (
        "could not download https://x/aaa: [download] Got error: "
        "simulated failure. Giving up after 5 retries"
    )


def test_a_bare_error_from_expand_is_explained_the_same_way(tmp_path: Path) -> None:
    downloader = Downloader(
        make_settings(tmp_path),
        ydl_factory=logging_then_exploding_factory(
            "ERROR: Video unavailable", YtDlpDownloadError("")
        ),
    )

    with pytest.raises(DownloadError) as excinfo:
        downloader.expand(["https://x/p"])

    assert str(excinfo.value) == "could not resolve https://x/p: Video unavailable"


def test_our_message_never_carries_yt_dlps_carriage_return(tmp_path: Path) -> None:
    """The reason usually IS in the exception -- with both prefixes attached.

    Carried through verbatim, our own sentence sends the cursor back to column
    0 and the reason overprints the explanation, which is how the user came to
    see a line ending in a bare "ERROR:". The colourless and the tty-coloured
    prefixes both have to go; ``_format_err`` decides which one at runtime.
    """
    for prefix in ("ERROR:", "\x1b[0;31mERROR:\x1b[0m"):
        downloader = Downloader(
            make_settings(tmp_path),
            ydl_factory=exploding_factory(YtDlpDownloadError(f"{prefix} {RETRY_TEXT}")),
        )

        with pytest.raises(DownloadError) as excinfo:
            downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", None), tmp_path)

        assert str(excinfo.value) == (
            "could not download https://x/aaa: [download] Got error: "
            "simulated failure. Giving up after 5 retries"
        )
        assert "\r" not in str(excinfo.value)


def test_an_error_that_says_nothing_at_all_is_still_named(tmp_path: Path) -> None:
    """Nothing in the exception and nothing logged: not a dangling colon."""
    downloader = Downloader(
        make_settings(tmp_path), ydl_factory=exploding_factory(YoutubeDLError(""))
    )

    with pytest.raises(DownloadError) as excinfo:
        downloader.fetch(TrackRef("aaa", "https://x/aaa", "One", None), tmp_path)

    assert str(excinfo.value) == "could not download https://x/aaa: YoutubeDLError"


def per_url_factory(replies: dict[str, Any]):
    """A factory answering each URL differently: an info dict, or an exception.

    ``expand`` is handed a list, and the point of these tests is what happens
    to the OTHER entries when one of them fails, which a single fixed reply
    cannot express.
    """

    class FakeYDL:
        def __init__(self, opts: dict[str, Any]) -> None:
            self.opts = opts

        def extract_info(self, url: str, download: bool = False) -> Any:
            reply = replies[url]
            if isinstance(reply, BaseException):
                raise reply
            return reply

    @contextmanager
    def factory(opts: dict[str, Any]) -> Iterator[FakeYDL]:
        yield FakeYDL(opts)

    return factory


def test_one_unresolvable_url_does_not_abort_the_others(tmp_path: Path) -> None:
    """--from-file with fifty links must not lose forty-nine to the seventh.

    ``process_track``'s own docstring already states the rule for tracks: one
    bad track in a hundred must not take the other ninety-nine with it.
    Resolution had the opposite behaviour, and it runs before any of it.
    """
    downloader = Downloader(
        make_settings(tmp_path),
        ydl_factory=per_url_factory(
            {
                "https://x/good1": {"id": "aaa", "title": "One"},
                "https://x/private": YtDlpDownloadError("Private video"),
                "https://x/good2": {"id": "bbb", "title": "Two"},
            }
        ),
    )

    refs = downloader.expand(
        ["https://x/good1", "https://x/private", "https://x/good2"]
    )

    assert [ref.video_id for ref in refs] == ["aaa", "bbb"]


def test_each_skipped_url_is_reported_with_its_reason(tmp_path: Path) -> None:
    """Skipping silently would leave the user with no idea what was dropped."""
    downloader = Downloader(
        make_settings(tmp_path),
        ydl_factory=per_url_factory(
            {
                "https://x/private": YtDlpDownloadError("ERROR: Private video"),
                "https://x/good": {"id": "aaa", "title": "One"},
            }
        ),
    )
    seen: list[tuple[str, str]] = []

    refs = downloader.expand(
        ["https://x/private", "https://x/good"],
        on_error=lambda url, reason: seen.append((url, reason)),
    )

    assert [ref.video_id for ref in refs] == ["aaa"]
    assert seen == [("https://x/private", "Private video")]


def test_every_url_failing_is_still_an_error_not_an_empty_run(
    tmp_path: Path,
) -> None:
    """Nothing resolved is a failure, not a silent success with no tracks."""
    downloader = Downloader(
        make_settings(tmp_path),
        ydl_factory=per_url_factory(
            {
                "https://x/a": YtDlpDownloadError("Private video"),
                "https://x/b": YtDlpDownloadError("Video unavailable"),
            }
        ),
    )

    with pytest.raises(DownloadError) as excinfo:
        downloader.expand(["https://x/a", "https://x/b"])

    # Both are named: the user cannot fix a list they cannot see.
    assert str(excinfo.value) == (
        "could not resolve https://x/a: Private video; "
        "could not resolve https://x/b: Video unavailable"
    )


def test_a_run_level_abort_still_stops_resolution_at_the_first_url(
    tmp_path: Path,
) -> None:
    """Skipping must not swallow the aborts that are meant to stop the run.

    An unreadable cookie store would otherwise be reported once per URL and
    then raised anyway, having tried all fifty.
    """
    replies: dict[str, Any] = {
        "https://x/a": CookieLoadError("could not read the firefox cookie store"),
        "https://x/b": {"id": "bbb", "title": "Two"},
    }
    downloader = Downloader(
        make_settings(tmp_path), ydl_factory=per_url_factory(replies)
    )
    seen: list[tuple[str, str]] = []

    with pytest.raises(CookieLoadError):
        downloader.expand(
            ["https://x/a", "https://x/b"],
            on_error=lambda url, reason: seen.append((url, reason)),
        )

    assert seen == []
