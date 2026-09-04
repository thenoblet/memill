"""Filename and tag derivation.

Pure functions only: no I/O, no subprocess, no network. The rules here are
fiddly and easy to get subtly wrong, which is exactly why they live in a module
that can be exhaustively tested in microseconds.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memill.errors import TransferError

DEFAULT_MAX_STEM = 150
MIN_STEM = 16
WINDOWS_PATH_LIMIT = 260

# NTFS rejects these outright. ext4 accepts every one of them, so a sanitiser
# written for Linux alone produces names that only fail once they reach the
# Windows mount -- long after the expensive download has happened.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE_RUN = re.compile(r"\s+")

# Windows refuses these as filenames whatever the extension.
_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

_PART_SUFFIX = ".part"  # transfer writes "<name>.mp3.part" before renaming

# ext4 and every other Linux filesystem limit ONE PATH COMPONENT to 255 bytes,
# not 255 characters. A 150-character CJK or Cyrillic title is 450 bytes of
# UTF-8 and cannot be written at all, so the track dies at the encode step --
# after the download has been paid for, which is exactly what this module
# exists to prevent. The suffixes the run appends are reserved here as well as
# in stem_budget, because that budget counts characters and this one counts
# bytes: the longest name a stem ever wears is "<stem>.webm.part", the kept
# source mid-copy.
NAME_BYTE_LIMIT = 255
_LONGEST_SUFFIX = ".webm"
DEFAULT_MAX_STEM_BYTES = NAME_BYTE_LIMIT - len(_LONGEST_SUFFIX) - len(_PART_SUFFIX)


def _truncate_bytes(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` UTF-8 bytes, never mid-character.

    ``errors="ignore"`` drops the incomplete sequence the cut may leave rather
    than turning it into a replacement character, so a truncated CJK title
    ends on the last character that fully fits.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


def sanitize_stem(
    stem: str,
    *,
    max_length: int = DEFAULT_MAX_STEM,
    max_bytes: int = DEFAULT_MAX_STEM_BYTES,
) -> str:
    """Return ``stem`` as a filename NTFS will accept, without its extension.

    Illegal characters become spaces rather than being deleted, so that
    ``Artist: Title`` keeps its word boundary instead of collapsing to
    ``ArtistTitle``. The runs of whitespace that produces are then collapsed.

    Both budgets are enforced: ``max_length`` in characters, for the Windows
    path limit, and ``max_bytes`` in UTF-8 bytes, for the per-component limit
    every Linux filesystem imposes.
    """
    cleaned = _ILLEGAL.sub(" ", stem)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned).strip()
    cleaned = cleaned.rstrip(". ")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    cleaned = _truncate_bytes(cleaned, max_bytes).rstrip(". ")
    if not cleaned:
        return "untitled"
    # Guard AFTER truncation, not before: slicing a long name can land exactly
    # on a reserved device name, and a small max_length is the regime where
    # that becomes reachable.
    #
    # The segment before the FIRST dot, not the whole stem: Windows reserves
    # the device name whatever follows it, so "CON.txt" is refused by NTFS
    # just as "CON" is -- and would be refused only on arrival at the mount,
    # once the download has already been paid for.
    if cleaned.partition(".")[0].upper() in _RESERVED:
        if max_length <= 1 or max_bytes <= 1:
            return "untitled"
        head = _truncate_bytes(cleaned[: max_length - 1], max_bytes - 1)
        cleaned = f"_{head}"
    return cleaned or "untitled"


def stem_budget(
    destination: Path,
    *,
    extension: str = ".mp3",
    limit: int = WINDOWS_PATH_LIMIT,
) -> int:
    """Longest stem keeping ``destination/<stem><extension>`` inside ``limit``.

    Budgets the ``.part`` suffix too: the transfer step writes that longer name
    first and renames, so a stem that only fits the final name would fail during
    the copy.
    """
    # Reserve 1 character for Windows' NUL terminator (MAX_PATH counts it)
    used = len(str(destination)) + 1 + len(extension) + len(_PART_SUFFIX) + 1
    available = limit - used
    if available < MIN_STEM:
        raise TransferError(
            f"destination path is too long for a filename: {destination} leaves "
            f"{available} characters, need at least {MIN_STEM}"
        )
    return min(DEFAULT_MAX_STEM, available)


# Trailing decoration that carries no information once the file is in a library.
_NOISE = re.compile(
    r"""\s*[\(\[]\s*
        (?:official\s+(?:music\s+)?(?:video|audio|visuali[sz]er)
          |official\s+lyrics?\s+video
          |lyrics?(?:\s+video)?
          |audio|visuali[sz]er|hd|hq|4k|remastered)
        \s*[\)\]]\s*""",
    re.IGNORECASE | re.VERBOSE,
)
_TOPIC_SUFFIX = re.compile(r"\s*-\s*topic\s*$", re.IGNORECASE)
# Bounded on the left so a title that merely contains a dash late on is not
# mistaken for an "Artist - Song" pair.
_ARTIST_TITLE = re.compile("^(?P<artist>.{1,80}?)\\s+[-\u2013\u2014]\\s+(?P<title>.+)$")


@dataclass(frozen=True, slots=True)
class TrackTags:
    """The ID3 fields we are prepared to assert. Everything else is dropped."""

    title: str
    artist: str | None = None
    album: str | None = None
    year: str | None = None
    source_url: str | None = None


def _first_str(info: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _year_of(info: Mapping[str, Any]) -> str | None:
    release = info.get("release_year")
    if isinstance(release, int):
        return str(release)
    upload = _first_str(info, "upload_date")
    return upload[:4] if upload and len(upload) >= 4 else None


def infer_tags(info: Mapping[str, Any], *, clean: bool = True) -> TrackTags:
    """Derive ID3 tags from a yt-dlp info dict.

    YouTube Music entries carry real ``track``/``artist`` fields and are trusted
    verbatim. Everything else is a guess built from the title and uploader, so
    ``clean=False`` turns the guessing off entirely.

    The two explicit fields are trusted independently. An entry carrying
    ``artist`` without ``track`` is ordinary on YouTube Music, and guarding the
    ``Artist - Song`` split on the track alone let the split overwrite a real
    artist with the channel name that happened to open the title.
    """
    explicit_track = _first_str(info, "track")
    title = explicit_track or _first_str(info, "title") or "untitled"
    explicit_artist = _first_str(info, "artist", "creator")
    artist = explicit_artist
    if artist is None:
        uploader = _first_str(info, "uploader", "channel", "uploader_id")
        artist = _TOPIC_SUFFIX.sub("", uploader).strip() or None if uploader else None

    if clean and explicit_track is None:
        stripped = _NOISE.sub(" ", title).strip()
        # If the title was nothing but decoration, the strip was wrong: keep
        # what YouTube actually said rather than asserting an empty title.
        title = stripped or title
        match = _ARTIST_TITLE.match(title)
        if match:
            # The title is still split -- "Some Channel - Song Name" names the
            # song either way -- but the guessed artist only stands in when
            # the entry named none.
            if explicit_artist is None:
                artist = match.group("artist").strip()
            title = match.group("title").strip()

    return TrackTags(
        title=title,
        artist=artist,
        album=_first_str(info, "album", "playlist_title"),
        year=_year_of(info),
        source_url=_first_str(info, "webpage_url", "original_url"),
    )


def output_stem(tags: TrackTags, *, max_length: int = DEFAULT_MAX_STEM) -> str:
    """Return the destination filename stem, already NTFS-safe."""
    raw = f"{tags.artist} - {tags.title}" if tags.artist else tags.title
    return sanitize_stem(raw, max_length=max_length)
