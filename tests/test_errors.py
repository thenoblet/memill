from __future__ import annotations

import pytest

from memill.errors import EncodeError, TransferError, Yt2Mp3Error, os_errors_as


def test_a_successful_block_raises_nothing() -> None:
    done: list[str] = []
    with os_errors_as(TransferError, "could not publish a.mp3"):
        done.append("ran")
    assert done == ["ran"]


def test_an_os_error_becomes_the_named_error_with_the_prefix() -> None:
    """The message is the prefix, a colon, and what the OS actually said.

    Both halves are load-bearing: the prefix says which of our steps failed,
    and the errno text is the only thing that tells the user it was a full
    disk rather than a bad path. Dropping either -- or dropping the ``from``
    that keeps the original as ``__cause__``, so a traceback still names the
    errno that started it -- fails here.
    """
    original = OSError(28, "No space left on device")
    with (
        pytest.raises(TransferError) as excinfo,
        os_errors_as(TransferError, "could not publish a.mp3"),
    ):
        raise original

    assert str(excinfo.value) == f"could not publish a.mp3: {original}"
    assert excinfo.value.__cause__ is original
    # The type is what ``process_track`` actually branches on.
    assert isinstance(excinfo.value, Yt2Mp3Error)


def test_the_error_type_is_the_one_the_caller_named() -> None:
    """Four sites raise two different errors through the one helper."""
    with (
        pytest.raises(EncodeError, match="could not run ffmpeg: boom"),
        os_errors_as(EncodeError, "could not run ffmpeg"),
    ):
        raise OSError("boom")


def test_an_os_error_subclass_is_converted_too() -> None:
    """Every real trigger is a subclass: PermissionError, FileNotFoundError.

    ``except OSError`` narrowed to ``except BlockingIOError`` -- or to a
    tuple that forgets one of these -- passes every other test in this file.
    """
    for original in (PermissionError("read-only"), FileNotFoundError("gone")):
        with (
            pytest.raises(TransferError) as excinfo,
            os_errors_as(TransferError, "could not record aaa"),
        ):
            raise original
        assert excinfo.value.__cause__ is original


def test_an_error_of_ours_raised_inside_passes_through_unchanged() -> None:
    """``run_encode`` reports ffmpeg's non-zero exit from inside one of these.

    That message carries ffmpeg's own stderr and is already in the hierarchy,
    so it must reach the user as written. Broadening the ``except OSError``
    to ``except Exception`` is the mutation this kills: every non-zero encode
    would then be relabelled "could not run ffmpeg: ffmpeg exited 1: ...",
    blaming the launch for a failure that happened long after it.
    """
    inner = EncodeError("ffmpeg exited 1: Invalid data found")
    with (
        pytest.raises(EncodeError) as excinfo,
        os_errors_as(EncodeError, "could not run ffmpeg"),
    ):
        raise inner

    assert excinfo.value is inner
    assert excinfo.value.__cause__ is None


def test_a_genuine_bug_is_left_alone() -> None:
    """The other half of the contract: only deliberate failures are ours.

    A ``KeyError`` or ``TypeError`` from inside a wrapped block is a defect in
    this package, and ``process_track`` must let it propagate with its
    traceback rather than report it as one track's bad luck and carry on.
    """
    with (
        pytest.raises(KeyError),
        os_errors_as(TransferError, "could not publish a.mp3"),
    ):
        raise KeyError("requested_downloads")
