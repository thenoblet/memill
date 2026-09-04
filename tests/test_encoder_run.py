from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from memill.config import VbrQuality
from memill.encoder import build_encode_command, require_ffmpeg, run_encode
from memill.errors import DependencyError, EncodeError
from memill.naming import TrackTags

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


@pytest.fixture
def tone(tmp_path: Path) -> Path:
    """Two seconds of 440Hz. Synthesised locally: these tests never use network."""
    path = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2", str(path)],
        check=True,
    )
    return path


def test_encode_writes_an_mp3_and_reports_progress(tone: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.mp3"
    seen: list[float] = []
    run_encode(
        build_encode_command(
            audio=tone,
            output=output,
            quality=VbrQuality(5),
            tags=TrackTags(title="Tone"),
        ),
        duration=2.0,
        on_progress=seen.append,
    )
    assert output.stat().st_size > 0
    assert seen and seen[-1] == 1.0
    assert all(0.0 <= value <= 1.0 for value in seen)


def test_encode_without_progress_callback(tone: Path, tmp_path: Path) -> None:
    """The default ``on_progress=None`` must be exercised on a real encode.

    A hand-built argv with no ``-progress pipe:1`` (as in the failing-encode
    test below) emits no stdout lines, so it never reaches the
    ``on_progress is not None`` guard. Only a real encode does.
    """
    output = tmp_path / "out.mp3"
    run_encode(
        build_encode_command(
            audio=tone,
            output=output,
            quality=VbrQuality(5),
            tags=TrackTags(title="Tone"),
        ),
        duration=2.0,
    )
    assert output.stat().st_size > 0


def test_progress_is_reported_more_than_once(tone: Path, tmp_path: Path) -> None:
    """A single terminal ``1.0`` would satisfy the other progress assertions.

    This ran on a ten-second fixture said to be "long enough to emit multiple
    progress blocks". Measured, that was false: libmp3lame encodes at roughly
    380x realtime here, so ten seconds of audio finishes inside ffmpeg's first
    stats period and yields exactly the two blocks two seconds does. The
    assertion held anyway because ffmpeg bookends every run with a start
    report and a final ``progress=end``, which is what it is really proving --
    that run_encode hands each block on as it parses it rather than calling
    back once at the end. No fixture short enough for a test suite can span
    two stats periods, so the longer tone bought nothing and is gone.
    """
    output = tmp_path / "out.mp3"
    seen: list[float] = []
    run_encode(
        build_encode_command(
            audio=tone,
            output=output,
            quality=VbrQuality(5),
            tags=TrackTags(title="Tone"),
        ),
        duration=2.0,
        on_progress=seen.append,
    )
    assert len(seen) > 1, "progress must stream, not arrive only at completion"


def test_a_failing_encode_raises_with_ffmpeg_stderr(tmp_path: Path) -> None:
    argv = [
        "ffmpeg", "-nostdin", "-y", "-i",
        str(tmp_path / "nope.wav"), str(tmp_path / "o.mp3"),
    ]
    with pytest.raises(EncodeError) as excinfo:
        run_encode(argv, duration=None)
    assert "nope.wav" in str(excinfo.value)


def test_large_stderr_does_not_deadlock() -> None:
    """The stderr-to-tempfile design exists for this; a second pipe hangs.

    Verified by mutation: changing ``stderr`` to ``subprocess.PIPE`` makes this
    fail. A deadlocked thread can never be joined, so the worker must be a
    daemon -- otherwise the failure hangs the session instead of reporting it.
    """
    argv = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('x' * 5_000_000); sys.exit(1)",
    ]
    captured: list[Exception | None] = [None]

    def target() -> None:
        try:
            run_encode(argv, duration=None)
        except Exception as exc:
            captured[0] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "run_encode deadlocked on large stderr"
    assert isinstance(captured[0], EncodeError)


def test_require_ffmpeg_passes_when_both_present() -> None:
    require_ffmpeg()


def test_require_ffmpeg_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(DependencyError, match="ffmpeg"):
        require_ffmpeg()
