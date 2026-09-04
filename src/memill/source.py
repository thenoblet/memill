"""yt-dlp adapter: playlist expansion and per-track retrieval.

The ``ydl_factory`` seam exists so the whole module can be exercised without a
network call. Nothing else in the package imports yt_dlp.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL
from yt_dlp.cookies import CookieLoadError
from yt_dlp.utils import DownloadCancelled, YoutubeDLError

from memill.config import Settings
from memill.errors import DownloadError

YdlFactory = Callable[[dict[str, Any]], AbstractContextManager[Any]]
# Told ``(url, reason)`` for each URL ``expand`` skips rather than fails on.
UrlErrorHandler = Callable[[str, str], None]

_COVER_SUFFIXES = (".jpg", ".jpeg", ".webp", ".png")
_HTTP_CHUNK_SIZE = 10 << 20  # 10 MiB; blunts YouTube's per-connection throttling.

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ERROR_PREFIX = "ERROR:"


class _Recorder:
    """Diverts yt-dlp's console output here, keeping its latest complaint.

    ``quiet`` is not enough. yt-dlp's ``report_error`` goes through
    ``to_stderr``, which never consults that flag, so the last retry's failure
    is written to the real stderr straight across the Rich live display -- and
    the downloader prefixes that text with a carriage return, so it lands at
    column 0 and overwrites the line we had already drawn there. Passing a
    logger in the options intercepts every one of those calls instead.

    One instance per ``YoutubeDL``, never one per ``Downloader``: a single
    Downloader is shared by the whole thread pool, so a recorder living on it
    would let one track's failure explain another's. Within a single download
    yt-dlp's fragment threads can report concurrently, which needs no lock
    here -- the attribute store is atomic and last-writer-wins is the whole
    contract.
    """

    __slots__ = ("message",)

    def __init__(self) -> None:
        self.message: str = ""

    def debug(self, message: str) -> None:
        """Dropped: ``to_screen`` lands here, progress chatter included."""

    def info(self, message: str) -> None:
        """Dropped: unused by yt-dlp today, but part of the interface."""

    def warning(self, message: str) -> None:
        self.message = message

    def error(self, message: str) -> None:
        self.message = message


def _plain(message: str) -> str:
    """yt-dlp's console text, made fit to sit inside a message of ours.

    Two prefixes have to go. ``report_error`` adds "ERROR:", coloured when
    stderr is a tty, and the retry path adds a carriage return; carried
    verbatim into our own sentence the latter sends the cursor back to column
    0 and overwrites the explanation with the reason. That is precisely what
    the user saw -- a line ending in a bare "ERROR:", the rest of it printed
    over its own beginning. Every other run of whitespace is collapsed too, so
    what we raise is always one line.
    """
    text = _ANSI.sub("", message).strip()
    if text.startswith(_ERROR_PREFIX):
        text = text[len(_ERROR_PREFIX) :]
    return " ".join(text.split())


def _reason(exc: YoutubeDLError, recorder: _Recorder) -> str:
    """Why the fetch failed, preferring the exception's own words.

    The recorder is the fallback rather than the first choice because it holds
    only the most recent line: accurate when the exception says nothing, but
    the exception is the one that belongs to this failure. Some yt-dlp paths
    raise with an empty message, or with nothing but the stripped "ERROR:",
    and it is those that would otherwise reach the user explaining nothing.
    """
    return _plain(str(exc)) or _plain(recorder.message) or type(exc).__name__


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


def as_float(value: Any) -> float | None:
    """A yt-dlp numeric field as a float, or ``None`` if it is not one.

    Public because the pipeline needs the identical coercion for the duration
    it reads out of the info dict, and two copies of one rule drift apart.
    """
    return float(value) if isinstance(value, (int, float)) else None


def _ref_from_entry(entry: Mapping[str, Any]) -> TrackRef | None:
    # Under extract_flat="in_playlist" a channel URL's top-level entries are
    # playlist tabs -- Videos, Shorts, Live -- not videos. Each carries an id
    # and a title, so without this check every tab becomes a TrackRef whose
    # fetch fails with "yt-dlp did not report a downloaded file", a message
    # that never mentions the channel URL that actually caused it.
    #
    # Defensive: this sits below the network seam, so the shape of a real
    # channel response cannot be proven by the offline suite. The check costs
    # nothing if yt-dlp never sets _type on the entries we do want, because a
    # video entry's _type is "url" or "video", never "playlist".
    if entry.get("_type") == "playlist":
        return None
    video_id = entry.get("id")
    if not isinstance(video_id, str) or not video_id:
        return None
    url = entry.get("webpage_url") or entry.get("url")
    return TrackRef(
        video_id=video_id,
        url=str(url) if url else f"https://www.youtube.com/watch?v={video_id}",
        title=str(entry.get("title") or video_id),
        duration=as_float(entry.get("duration")),
    )


class Downloader:
    """Retrieves the audio stream, falling back to a muxed file only if it must.

    The format selector is ``bestaudio/best``, and both halves matter. The
    first asks for an audio-only stream, which is what YouTube serves for
    essentially every video and is the saving this tool exists for: the video
    track is far larger than the audio it carries.

    The second is a fallback, and it is not audio-only. A source that offers
    no audio-only format at all -- some of yt-dlp's 1800+ extractors never do
    -- yields a muxed audio+video file, downloaded whole; ffmpeg then maps
    ``0:a:0`` out of it, so the MP3 that reaches the library is the same
    either way, but the bytes on the wire are not.

    Kept rather than tightened to ``bestaudio`` deliberately. Requiring an
    audio-only format would turn every such source from "a bigger download"
    into "no download", which is the worse outcome for the user actually
    holding the URL, and the end-to-end suite already exercises a muxed input
    for exactly this path.
    """

    __slots__ = ("_factory", "_settings")

    def __init__(
        self, settings: Settings, *, ydl_factory: YdlFactory | None = None
    ) -> None:
        self._settings = settings
        self._factory: YdlFactory = ydl_factory or _default_factory

    def expand(
        self, urls: Sequence[str], *, on_error: UrlErrorHandler | None = None
    ) -> list[TrackRef]:
        """Resolve URLs to individual tracks without downloading anything.

        Flat extraction is cheap and gives an accurate total up front, which is
        what makes the batch progress bar honest from its first frame.

        A URL that cannot be resolved is skipped, not fatal: ``--from-file``
        with fifty links whose seventh is private must still download the other
        forty-nine, the same rule ``process_track`` applies one track at a time.
        Each failure is passed to ``on_error`` as ``(url, reason)`` so the
        caller can say which link was dropped and why. If every URL fails,
        there is nothing left to do and that is an error rather than a silent
        empty run.
        """
        opts: dict[str, Any] = {
            **self._common_opts(),
            "extract_flat": "in_playlist",
            "skip_download": True,
        }
        refs: list[TrackRef] = []
        failures: list[str] = []
        first: YoutubeDLError | None = None
        for url in urls:
            # yt-dlp raises its own hierarchy, whose DownloadError shares our
            # name but is unrelated to it. Unwrapped, an ordinary unavailable
            # video escapes past the pipeline's Yt2Mp3Error handler and kills
            # the whole batch instead of failing one track. The dict is copied
            # per iteration because YoutubeDL writes its defaults into the
            # options it is handed -- and each copy gets its own recorder, so
            # one URL's complaint can never be reported against the next.
            recorder = _Recorder()
            try:
                with self._factory({**opts, "logger": recorder}) as ydl:
                    info = ydl.extract_info(url, download=False)
            except (CookieLoadError, DownloadCancelled):
                # Run-level aborts, not per-video failures: yt-dlp re-raises
                # these unconverted too. An unreadable cookie store must stop
                # resolution once, not repeat itself for every URL given.
                raise
            except YoutubeDLError as exc:
                reason = _reason(exc, recorder)
                failures.append(f"could not resolve {url}: {reason}")
                # The first one, not the last: it is the cause of the raise
                # below, and a chained traceback should name the failure the
                # message opens with.
                first = first or exc
                if on_error is not None:
                    on_error(url, reason)
                continue
            entries = info.get("entries") if isinstance(info, Mapping) else None
            candidates = entries if isinstance(entries, list) else [info]
            refs.extend(
                ref
                for entry in candidates
                if isinstance(entry, Mapping)
                and (ref := _ref_from_entry(entry)) is not None
            )
        if failures and not refs:
            raise DownloadError("; ".join(failures)) from first
        return refs

    def _common_opts(self) -> dict[str, Any]:
        """The options every yt-dlp session gets, whichever call site opens it.

        There are two of those -- ``expand`` resolves, ``fetch`` downloads --
        and they were written as separate literals, which is how
        ``--cookies-from-browser`` came to be honoured by only one of them.
        Resolution runs first, so age-restricted content failed before the
        download the flag was meant to unlock ever started; the fix had to be
        applied twice and the second site was nearly missed. Adding a key here
        now reaches both by construction.

        A fresh dict per call, never a shared or cached one: ``YoutubeDL``
        writes its own defaults into the options it is handed, so a dict that
        outlived one session would carry that pollution into the next.
        """
        opts: dict[str, Any] = {"quiet": True, "no_warnings": True}
        if self._settings.cookies_from_browser:
            opts["cookiesfrombrowser"] = (self._settings.cookies_from_browser,)
        return opts

    def _download_opts(
        self,
        staging: Path,
        on_progress: Callable[[float], None] | None,
        recorder: _Recorder,
    ) -> dict[str, Any]:
        opts: dict[str, Any] = {
            **self._common_opts(),
            "format": "bestaudio/best",
            "outtmpl": {"default": str(staging / "%(id)s.%(ext)s")},
            "noplaylist": True,
            "writethumbnail": self._settings.embed_cover,
            "concurrent_fragment_downloads": self._settings.fragments,
            "http_chunk_size": _HTTP_CHUNK_SIZE,
            "retries": 5,
            "fragment_retries": 5,
            "socket_timeout": 20,
            "noprogress": True,
            # Neither that nor the shared ``quiet``/``no_warnings`` is enough
            # on its own: report_error consults none of them. The logger is
            # what actually keeps yt-dlp's last retry failure off our stderr,
            # and it hands us the reason it would otherwise have printed
            # there. Fresh per call, because the options dict is too.
            "logger": recorder,
            # ffmpeg is ours; yt-dlp must not post-process behind our back.
            "postprocessors": [],
        }
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
        # Per call, not per Downloader: one Downloader serves the whole pool.
        recorder = _Recorder()
        try:
            with self._factory(
                self._download_opts(staging, on_progress, recorder)
            ) as ydl:
                info = ydl.extract_info(ref.url, download=True)
        except (CookieLoadError, DownloadCancelled):
            # The same run-level aborts, on the path that runs once per track:
            # a bad --cookies-from-browser must stop the run, not turn into N
            # identical per-track failures.
            raise
        except YoutubeDLError as exc:
            raise DownloadError(
                f"could not download {ref.url}: {_reason(exc, recorder)}"
            ) from exc
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
