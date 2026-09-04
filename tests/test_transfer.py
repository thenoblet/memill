from __future__ import annotations

import concurrent.futures
import shutil
from pathlib import Path

import pytest

from yt2mp3.errors import TransferError
from yt2mp3.transfer import Archive, publish


def test_publish_moves_bytes_and_leaves_no_part_file(tmp_path: Path) -> None:
    source = tmp_path / "encoded.mp3"
    source.write_bytes(b"audio-bytes")
    destination = tmp_path / "library"

    result = publish(source, destination, "Artist - Song.mp3")

    assert result == destination / "Artist - Song.mp3"
    assert result.read_bytes() == b"audio-bytes"
    assert list(destination.glob("*.part")) == []


def test_publish_creates_a_missing_destination(tmp_path: Path) -> None:
    source = tmp_path / "a.mp3"
    source.write_bytes(b"x")
    assert publish(source, tmp_path / "deep" / "nested", "a.mp3").exists()


def test_a_failed_copy_leaves_no_partial_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "a.mp3"
    source.write_bytes(b"x")
    destination = tmp_path / "library"

    def explode(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copyfile", explode)
    with pytest.raises(TransferError):
        publish(source, destination, "a.mp3")
    assert list(destination.iterdir()) == []


def test_publish_stages_through_a_part_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy straight to the final name passes the other tests. Not this one."""
    source = tmp_path / "s.mp3"
    source.write_bytes(b"data")
    destination = tmp_path / "library"
    observed: list[str] = []
    real_copyfile = shutil.copyfile

    def recording_copyfile(src: object, dst: object, **kwargs: object) -> object:
        observed.append(str(dst))
        return real_copyfile(src, dst, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shutil, "copyfile", recording_copyfile)
    publish(source, destination, "Song.mp3")

    assert observed == [str(destination / "Song.mp3.part")]


def test_a_failed_copy_removes_the_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "s.mp3"
    source.write_bytes(b"data")
    destination = tmp_path / "library"
    real_copyfile = shutil.copyfile

    def failing_copyfile(src: object, dst: object, **kwargs: object) -> object:
        # Write real bytes BEFORE failing: an interrupted copy leaves a partial
        # file on disk, and removing it is the behaviour under test.
        real_copyfile(src, dst, **kwargs)  # type: ignore[arg-type]
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copyfile", failing_copyfile)
    with pytest.raises(TransferError):
        publish(source, destination, "Song.mp3")

    assert not (destination / "Song.mp3.part").exists()
    assert list(destination.iterdir()) == []


def test_archive_remembers_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "archive.txt"
    archive = Archive(path)
    assert "abc123" not in archive
    archive.add("abc123")
    assert "abc123" in archive
    assert "abc123" in Archive(path)


def test_a_disabled_archive_remembers_nothing(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive.txt", enabled=False)
    archive.add("abc123")
    assert "abc123" not in archive
    assert not (tmp_path / "archive.txt").exists()


def test_archive_is_safe_under_concurrent_writers(tmp_path: Path) -> None:
    path = tmp_path / "archive.txt"
    archive = Archive(path)

    def add_many(worker: int) -> None:
        for index in range(200):
            archive.add(f"id-{worker}-{index}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add_many, range(8)))

    lines = path.read_text("utf-8").splitlines()
    assert len(lines) == 1600
    assert len(set(lines)) == 1600
    # A torn write would splice two ids into one line.
    assert all(line.count("id-") == 1 for line in lines)

    reloaded = Archive(path)
    assert all(f"id-{w}-{i}" in reloaded for w in range(8) for i in range(200))
