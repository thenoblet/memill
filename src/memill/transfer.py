"""Publishing finished audio to its destination, and remembering what is done."""

from __future__ import annotations

import shutil
import threading
from contextlib import suppress
from pathlib import Path

from memill.errors import TransferError, os_errors_as

ARCHIVE_FILENAME = ".memill-archive"


def publish(source: Path, destination_dir: Path, filename: str) -> Path:
    """Copy ``source`` into ``destination_dir`` and reveal it under ``filename``.

    ``os.rename`` cannot cross a filesystem boundary, and staging deliberately
    lives on a different one from the library, so this is a copy. A copy can be
    interrupted, which would leave a truncated file wearing the real name and
    looking finished -- so the bytes land under ``.part`` first and are revealed
    by a rename, which within one filesystem is atomic.

    The ``mkdir`` is inside the wrapped block with the copy, not before it. An
    absent destination mount or an unwritable ``-o`` directory raises there,
    and a bare ``OSError`` escaping this call flies past the pipeline's
    ``Yt2Mp3Error`` handler and cancels every remaining track in the batch --
    the same defect ``Archive.add`` already guards against below.
    """
    final = destination_dir / filename
    partial = destination_dir / f"{filename}.part"
    try:
        with os_errors_as(TransferError, f"could not publish {filename}"):
            destination_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, partial)
            partial.replace(final)
    except TransferError:
        # Caught after the conversion rather than instead of it, so this stays
        # a cleanup and the wrapping stays in one place. The cleanup can fail
        # too -- a leftover .part in a directory that has since gone read-only
        # -- and that second OSError must not replace the first on its way out.
        with suppress(OSError):
            partial.unlink(missing_ok=True)
        raise
    return final


class Archive:
    """The set of track ids already finished, persisted one id per line.

    Disabled instances answer "no" to everything and write nothing, so callers
    never need to branch on whether resume is switched on.
    """

    __slots__ = ("_enabled", "_lock", "_path", "_seen")

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self._path = path
        self._enabled = enabled
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        if enabled and path.exists():
            self._seen = {
                line.strip()
                for line in path.read_text("utf-8").splitlines()
                if line.strip()
            }

    def __contains__(self, key: str) -> bool:
        if not self._enabled:
            return False
        with self._lock:
            return key in self._seen

    def add(self, key: str) -> None:
        """Record ``key`` as finished. Called only after the file is in place.

        Two orderings matter here. The write is wrapped in ``TransferError``
        because an unwritable archive is an expected failure -- a read-only
        mount, a bad path -- and a bare ``PermissionError`` escaping this call
        would fly past the pipeline's ``Yt2Mp3Error`` handler and abort a batch
        whose files were all published successfully.

        And ``_seen`` is updated only once the line is on disk. Marking it
        first would leave this process believing a track is recorded when the
        write failed, so a later duplicate would be skipped on the strength of
        a record that does not exist.
        """
        if not self._enabled:
            return
        with self._lock:
            if key in self._seen:
                return
            with os_errors_as(
                TransferError, f"could not record {key} in the archive {self._path}"
            ):
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{key}\n")
            self._seen.add(key)
