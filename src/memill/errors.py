"""Exception hierarchy.

Every failure this tool raises deliberately is a ``Yt2Mp3Error``, so the CLI can
distinguish an expected failure (report it, keep going) from a genuine bug
(let it propagate with its traceback intact).
"""

from __future__ import annotations


class Yt2Mp3Error(Exception):
    """Base class for every error this package raises deliberately."""


class DependencyError(Yt2Mp3Error):
    """A required external program is missing or unusable."""


class DownloadError(Yt2Mp3Error):
    """yt-dlp could not retrieve the media."""


class EncodeError(Yt2Mp3Error):
    """ffmpeg exited non-zero."""


class TransferError(Yt2Mp3Error):
    """The finished file could not be published to its destination."""
