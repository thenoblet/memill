from __future__ import annotations

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
