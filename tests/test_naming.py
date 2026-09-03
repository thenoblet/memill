from __future__ import annotations

import pytest

from yt2mp3.naming import sanitize_stem


def test_replaces_every_ntfs_illegal_character() -> None:
    assert sanitize_stem('AC/DC: Live | "Best" <2024>?*') == "AC DC Live Best 2024"


def test_strips_trailing_dots_and_spaces() -> None:
    assert sanitize_stem("Mix Vol. 4...  ") == "Mix Vol. 4"


@pytest.mark.parametrize("name", ["CON", "com1", "NUL", "lpt9"])
def test_prefixes_reserved_device_names(name: str) -> None:
    assert sanitize_stem(name) == f"_{name}"


def test_truncates_and_leaves_no_trailing_dot() -> None:
    result = sanitize_stem("A" * 120 + "." * 40, max_length=150)
    assert len(result) <= 150
    assert not result.endswith(".")


def test_falls_back_when_nothing_survives() -> None:
    assert sanitize_stem("///:::") == "untitled"
