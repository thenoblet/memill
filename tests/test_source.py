from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from yt2mp3 import source
from yt2mp3.config import Settings, VbrQuality
from yt2mp3.errors import DownloadError
from yt2mp3.source import Downloader, TrackRef


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


def test_download_options_carry_audio_only_format_and_the_fragment_budget(
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
    assert opts["format"] == "bestaudio/best"
    assert opts["concurrent_fragment_downloads"] == 4
    assert opts["noplaylist"] is True
    # ffmpeg is ours: yt-dlp must hand us the raw stream, unprocessed.
    assert opts["postprocessors"] == []
    assert opts["outtmpl"] == {"default": str(staging / "%(id)s.%(ext)s")}


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
        return self._info


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
    spy = SpyFactory({"id": "aaa", "title": "One", "webpage_url": "https://x/aaa"})
    downloader = Downloader(make_settings(tmp_path), ydl_factory=spy)

    downloader.expand(["https://x/a", "https://x/b"])

    assert [call["url"] for call in spy.calls] == ["https://x/a", "https://x/b"]
    assert [call["download"] for call in spy.calls] == [False, False]
    assert spy.sessions[0].opts["extract_flat"] == "in_playlist"
    assert spy.sessions[0].opts["skip_download"] is True
    assert [session.closed for session in spy.sessions] == [True, True]


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
