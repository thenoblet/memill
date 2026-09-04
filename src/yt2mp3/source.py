"""yt-dlp adapter: playlist expansion and per-track retrieval.

The ``ydl_factory`` seam exists so the whole module can be exercised without a
network call. Nothing else in the package imports yt_dlp.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL

from yt2mp3.config import Settings
from yt2mp3.errors import DownloadError

YdlFactory = Callable[[dict[str, Any]], AbstractContextManager[Any]]

_COVER_SUFFIXES = (".jpg", ".jpeg", ".webp", ".png")
_HTTP_CHUNK_SIZE = 10 << 20  # 10 MiB; blunts YouTube's per-connection throttling.


@dataclass(frozen=True, slots=True)
class TrackRef:
    """One track we intend to fetch, known before anything is downloaded."""

    video_id: str
    url: str
    title: str
    duration: float | None


@dataclass(frozen=True, slots=True)
class SourceMedia:
    """What arrived on disk for one track."""

    audio: Path
    cover: Path | None
    info: Mapping[str, Any]


@contextmanager
def _default_factory(opts: dict[str, Any]) -> Iterator[YoutubeDL]:
    with YoutubeDL(opts) as ydl:
        yield ydl


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _ref_from_entry(entry: Mapping[str, Any]) -> TrackRef | None:
    video_id = entry.get("id")
    if not isinstance(video_id, str) or not video_id:
        return None
    url = entry.get("webpage_url") or entry.get("url")
    return TrackRef(
        video_id=video_id,
        url=str(url) if url else f"https://www.youtube.com/watch?v={video_id}",
        title=str(entry.get("title") or video_id),
        duration=_as_float(entry.get("duration")),
    )


class Downloader:
    """Retrieves audio-only media, never video."""

    __slots__ = ("_factory", "_settings")

    def __init__(
        self, settings: Settings, *, ydl_factory: YdlFactory | None = None
    ) -> None:
        self._settings = settings
        self._factory: YdlFactory = ydl_factory or _default_factory

    def expand(self, urls: Sequence[str]) -> list[TrackRef]:
        """Resolve URLs to individual tracks without downloading anything.

        Flat extraction is cheap and gives an accurate total up front, which is
        what makes the batch progress bar honest from its first frame.
        """
        opts = {
            "extract_flat": "in_playlist",
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }
        refs: list[TrackRef] = []
        for url in urls:
            with self._factory(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            entries = info.get("entries") if isinstance(info, Mapping) else None
            candidates = entries if isinstance(entries, list) else [info]
            refs.extend(
                ref
                for entry in candidates
                if isinstance(entry, Mapping)
                and (ref := _ref_from_entry(entry)) is not None
            )
        return refs

    def _download_opts(
        self, staging: Path, on_progress: Callable[[float], None] | None
    ) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "format": "bestaudio/best",
            "outtmpl": {"default": str(staging / "%(id)s.%(ext)s")},
            "noplaylist": True,
            "writethumbnail": self._settings.embed_cover,
            "concurrent_fragment_downloads": self._settings.fragments,
            "http_chunk_size": _HTTP_CHUNK_SIZE,
            "retries": 5,
            "fragment_retries": 5,
            "socket_timeout": 20,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            # ffmpeg is ours; yt-dlp must not post-process behind our back.
            "postprocessors": [],
        }
        if self._settings.cookies_from_browser:
            opts["cookiesfrombrowser"] = (self._settings.cookies_from_browser,)
        if on_progress is not None:
            opts["progress_hooks"] = [_make_hook(on_progress)]
        return opts

    def fetch(
        self,
        ref: TrackRef,
        staging: Path,
        on_progress: Callable[[float], None] | None = None,
    ) -> SourceMedia:
        """Download one track's audio (and thumbnail) into ``staging``."""
        with self._factory(self._download_opts(staging, on_progress)) as ydl:
            info = ydl.extract_info(ref.url, download=True)
        if not isinstance(info, Mapping):
            raise DownloadError(f"yt-dlp returned no metadata for {ref.url}")
        return SourceMedia(
            audio=_downloaded_path(info, ref),
            cover=(
                _find_cover(staging, ref.video_id)
                if self._settings.embed_cover
                else None
            ),
            info=info,
        )


def _make_hook(
    on_progress: Callable[[float], None],
) -> Callable[[dict[str, Any]], None]:
    def hook(status: dict[str, Any]) -> None:
        if status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        done = status.get("downloaded_bytes")
        if (
            isinstance(total, (int, float))
            and total > 0
            and isinstance(done, (int, float))
        ):
            on_progress(min(1.0, done / total))

    return hook


def _downloaded_path(info: Mapping[str, Any], ref: TrackRef) -> Path:
    requested = info.get("requested_downloads")
    if isinstance(requested, list) and requested:
        filepath = requested[0].get("filepath")
        if isinstance(filepath, str):
            return Path(filepath)
    raise DownloadError(f"yt-dlp did not report a downloaded file for {ref.url}")


def _find_cover(staging: Path, video_id: str) -> Path | None:
    for suffix in _COVER_SUFFIXES:
        candidate = staging / f"{video_id}{suffix}"
        if candidate.exists():
            return candidate
    return None
