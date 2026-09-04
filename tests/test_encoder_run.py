from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from yt2mp3.config import VbrQuality
from yt2mp3.encoder import build_encode_command, run_encode
from yt2mp3.errors import EncodeError
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


def test_a_failing_encode_raises_with_ffmpeg_stderr(tmp_path: Path) -> None:
    argv = [
        "ffmpeg", "-nostdin", "-y", "-i",
        str(tmp_path / "nope.wav"), str(tmp_path / "o.mp3"),
    ]
    with pytest.raises(EncodeError) as excinfo:
        run_encode(argv, duration=None)
    assert "nope.wav" in str(excinfo.value)
