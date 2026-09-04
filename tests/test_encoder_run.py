from __future__ import annotations

import concurrent.futures
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from yt2mp3.config import VbrQuality
from yt2mp3.encoder import build_encode_command, require_ffmpeg, run_encode
from yt2mp3.errors import DependencyError, EncodeError
from yt2mp3.naming import TrackTags

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


@pytest.fixture
def long_tone(tmp_path: Path) -> Path:
    """Ten seconds of 440Hz -- long enough to emit multiple progress blocks."""
    path = tmp_path / "long_tone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=10", str(path)],
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


def test_progress_is_reported_incrementally(long_tone: Path, tmp_path: Path) -> None:
    """A single terminal ``1.0`` would satisfy the other progress assertions."""
    output = tmp_path / "out.mp3"
    seen: list[float] = []
    run_encode(
        build_encode_command(
            audio=long_tone,
            output=output,
            quality=VbrQuality(5),
            tags=TrackTags(title="Tone"),
        ),
        duration=10.0,
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


def test_large_stderr_does_not_deadlock(tmp_path: Path) -> None:
    """The stderr-to-tempfile design exists for this. A second pipe would hang."""
    argv = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('x' * 5_000_000); sys.exit(1)",
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_encode, argv, duration=None)
        with pytest.raises(EncodeError):
            # A deadlock shows up as TimeoutError rather than hanging the suite.
            future.result(timeout=30)


def test_require_ffmpeg_passes_when_both_present() -> None:
    require_ffmpeg()


def test_require_ffmpeg_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(DependencyError, match="ffmpeg"):
        require_ffmpeg()
