"""Filename and tag derivation.

Pure functions only: no I/O, no subprocess, no network. The rules here are
fiddly and easy to get subtly wrong, which is exactly why they live in a module
that can be exhaustively tested in microseconds.
"""

from __future__ import annotations

import re
from pathlib import Path

from yt2mp3.errors import TransferError

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


def sanitize_stem(stem: str, *, max_length: int = DEFAULT_MAX_STEM) -> str:
    """Return ``stem`` as a filename NTFS will accept, without its extension.

    Illegal characters become spaces rather than being deleted, so that
    ``Artist: Title`` keeps its word boundary instead of collapsing to
    ``ArtistTitle``. The runs of whitespace that produces are then collapsed.
    """
    cleaned = _ILLEGAL.sub(" ", stem)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned).strip()
    cleaned = cleaned.rstrip(". ")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(". ")
    if not cleaned:
        return "untitled"
    # Guard AFTER truncation, not before: slicing a long name can land exactly
    # on a reserved device name, and a small max_length is the regime where
    # that becomes reachable.
    if cleaned.upper() in _RESERVED:
        cleaned = f"_{cleaned[: max_length - 1]}" if max_length > 1 else "untitled"
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
