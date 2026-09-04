"""ffmpeg command construction, progress parsing and execution."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from memill.config import Quality
from memill.errors import DependencyError, EncodeError, os_errors_as
from memill.naming import TrackTags

COVER_SIZE = 600
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"


def build_encode_command(
    *,
    audio: Path,
    output: Path,
    quality: Quality,
    tags: TrackTags,
    cover: Path | None = None,
    normalize: bool = False,
    cover_size: int = COVER_SIZE,
) -> list[str]:
    """Build the single ffmpeg invocation that encodes, tags and illustrates.

    One pass, two inputs. Doing this as three chained calls -- encode, then
    rewrite for tags, then rewrite again for artwork -- rewrites a 200MB file
    twice for no benefit.
    """
    argv: list[str] = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(audio)
    ]
    if cover is not None:
        argv += ["-i", str(cover)]

    argv += ["-map", "0:a:0"]
    if cover is not None:
        argv += ["-map", "1:v:0"]

    argv += ["-c:a", "libmp3lame", *quality.ffmpeg_args()]
    if normalize:
        argv += ["-af", LOUDNORM]

    if cover is not None:
        # Centre-crop the 16:9 thumbnail to the square that music players expect.
        argv += [
            "-c:v",
            "mjpeg",
            "-filter:v",
            f"crop='min(iw,ih)':'min(iw,ih)',scale={cover_size}:{cover_size}",
            "-disposition:v",
            "attached_pic",
        ]

    # Must precede the -metadata flags below: this clears mapped metadata, and
    # ordering it after them invites a clear that outranks the tags we set.
    argv += ["-map_metadata", "-1"]
    if cover is not None:
        argv += [
            "-metadata:s:v",
            "title=Album cover",
            "-metadata:s:v",
            "comment=Cover (front)",
        ]

    for key, value in (
        ("title", tags.title),
        ("artist", tags.artist),
        ("album", tags.album),
        ("date", tags.year),
        ("comment", tags.source_url),
    ):
        if value:
            argv += ["-metadata", f"{key}={value}"]

    argv += [
        "-id3v2_version", "3", "-write_id3v1", "1",
        "-progress", "pipe:1", "-nostats"
    ]
    argv.append(str(output))
    return argv


class ProgressParser:
    """Turn ffmpeg's ``-progress`` key/value stream into a 0..1 fraction.

    ffmpeg emits one ``key=value`` per line and terminates each block with
    ``progress=continue`` or ``progress=end``. Only ``out_time_us`` carries
    position, and it is meaningless without a duration -- which yt-dlp has
    already told us, so we never pay for a separate ffprobe call.
    """

    __slots__ = ("_duration_us",)

    def __init__(self, duration: float | None) -> None:
        self._duration_us = duration * 1_000_000 if duration and duration > 0 else None

    def feed(self, line: str) -> float | None:
        """Return the new fraction if this line moved it, else ``None``."""
        key, separator, value = line.strip().partition("=")
        if not separator:
            return None
        if key == "progress" and value == "end":
            return 1.0
        if key != "out_time_us" or self._duration_us is None:
            return None
        try:
            elapsed_us = float(value)
        except ValueError:
            return None
        return min(1.0, max(0.0, elapsed_us / self._duration_us))


_STDERR_TAIL_CHARS = 2000


def require_ffmpeg() -> None:
    """Fail early and clearly rather than at the end of a long download."""
    for program in ("ffmpeg", "ffprobe"):
        if shutil.which(program) is None:
            raise DependencyError(f"{program} is not on PATH")


def run_encode(
    argv: Sequence[str],
    *,
    duration: float | None,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    """Run ffmpeg, streaming progress, raising ``EncodeError`` on failure.

    stderr goes to a temporary file rather than a second pipe. Reading two pipes
    from one thread deadlocks the moment either fills its buffer, and the usual
    fixes (a reader thread, ``communicate``) would either add machinery or
    discard the incremental progress this function exists to deliver.
    """
    parser = ProgressParser(duration)
    # Starting the process is a failure mode of its own, distinct from the
    # non-zero exit below: ffmpeg uninstalled between the startup check and
    # this track, a fork refused with EAGAIN under a large -j, or a full /tmp
    # that the scratch file cannot be created in. ``require_ffmpeg`` closes
    # only the startup window. A bare OSError escaping here is outside
    # errors.py's contract, so ``process_track`` would let it through and one
    # track's mishap would cancel the whole batch. The EncodeError raised
    # inside for a non-zero exit is already in the hierarchy and passes
    # through this untouched, keeping ffmpeg's own words.
    with (
        os_errors_as(EncodeError, "could not run ffmpeg"),
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr,
    ):
        # argv is built by us from Path objects and never a shell string.
        process = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Popen is itself a context manager: it closes the pipes and waits,
        # so there is no hand-rolled cleanup to get wrong on the error path.
        with process:
            for line in process.stdout or ():
                fraction = parser.feed(line)
                if fraction is not None and on_progress is not None:
                    on_progress(fraction)

        if process.returncode != 0:
            stderr.seek(0)
            detail = stderr.read().strip()[-_STDERR_TAIL_CHARS:]
            raise EncodeError(f"ffmpeg exited {process.returncode}: {detail}")
