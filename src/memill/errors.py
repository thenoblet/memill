"""Exception hierarchy.

Every failure this tool raises deliberately is a ``Yt2Mp3Error``, so the CLI can
distinguish an expected failure (report it, keep going) from a genuine bug
(let it propagate with its traceback intact).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


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


@contextmanager
def os_errors_as(error: type[Yt2Mp3Error], message: str) -> Iterator[None]:
    """Run a block of I/O, restating any ``OSError`` as ``error``.

    Every filesystem and process call this tool makes can fail for a reason
    that is the user's situation rather than our bug -- a full disk, a
    read-only mount, an absent destination, a fork refused under a large
    ``-j`` -- and the module docstring's contract says those arrive as a
    ``Yt2Mp3Error`` so ``process_track`` fails one track instead of the batch.

    That contract used to be met by a hand-written ``try/except OSError`` at
    each site, which is to say it was met wherever someone remembered: a
    review found ``publish``'s ``mkdir`` and ``run_encode``'s ``Popen`` had
    both been missed, and each of them cancelled every remaining track in a
    playlist. Wrapping the block instead makes the next I/O call obey the
    contract by construction.

    The raised message is ``"{message}: {exc}"``, and the original is kept as
    ``__cause__`` so a traceback still names the errno that started it.

    Only ``OSError`` is caught, deliberately narrowly. A ``Yt2Mp3Error``
    raised inside the block -- ``run_encode`` reports ffmpeg's non-zero exit
    from within one of these -- is already in the hierarchy and passes
    through with its own wording, and anything else is the bug the contract
    exists to let through.
    """
    try:
        yield
    except OSError as exc:
        raise error(f"{message}: {exc}") from exc
