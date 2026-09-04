"""ffmpeg command construction, progress parsing and execution."""

from __future__ import annotations

from pathlib import Path

from yt2mp3.config import Quality
from yt2mp3.naming import TrackTags

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
