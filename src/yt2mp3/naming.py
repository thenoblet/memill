"""Filename and tag derivation.

Pure functions only: no I/O, no subprocess, no network. The rules here are
fiddly and easy to get subtly wrong, which is exactly why they live in a module
that can be exhaustively tested in microseconds.
"""

from __future__ import annotations

import re

DEFAULT_MAX_STEM = 150

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


def sanitize_stem(stem: str, *, max_length: int = DEFAULT_MAX_STEM) -> str:
    """Return ``stem`` as a filename NTFS will accept, without its extension.

    Illegal characters become spaces rather than being deleted, so that
    ``Artist: Title`` keeps its word boundary instead of collapsing to
    ``ArtistTitle``. The runs of whitespace that produces are then collapsed.
    """
    cleaned = _ILLEGAL.sub(" ", stem)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned).strip()
    cleaned = cleaned.rstrip(". ")
    if not cleaned:
        return "untitled"
    if cleaned.upper() in _RESERVED:
        cleaned = f"_{cleaned}"
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(". ")
    return cleaned or "untitled"
