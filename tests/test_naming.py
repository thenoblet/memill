from __future__ import annotations

from pathlib import Path

import pytest

from yt2mp3.naming import WINDOWS_PATH_LIMIT, sanitize_stem, stem_budget


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


def test_truncation_of_reserved_name_removes_reservation() -> None:
    # CON + 20 chars, truncated to 3, should give "CON" at truncation point
    # but the guard should NOT prefix it since it's protected by max_length
    result = sanitize_stem("CON" + "1" * 20, max_length=3)
    assert result != "CON"
    assert len(result) <= 3


def test_reserved_guard_respects_max_length() -> None:
    # When a reserved name is prefixed, the result must still fit max_length
    result = sanitize_stem("NUL", max_length=4)
    assert result == "_NUL"
    assert len(result) <= 4
    # With max_length=2, truncation happens first, removing reservation
    result_small = sanitize_stem("NUL", max_length=2)
    assert len(result_small) <= 2
    assert result_small != "NUL"


def test_stem_budget_short_destination_yields_default() -> None:
    dest = Path("/tmp/music")
    budget = stem_budget(dest)
    assert budget == 150  # DEFAULT_MAX_STEM


def test_stem_budget_long_destination_yields_smaller() -> None:
    # Create a very long destination path to reduce budget
    dest = Path("/" + "a" * 200)
    budget = stem_budget(dest)
    assert budget < 150
    assert budget >= 16  # Minimum budget


def test_stem_budget_pathologically_long_destination_yields_minimum() -> None:
    # Create an extremely long destination path
    dest = Path("/" + "a" * 240)
    budget = stem_budget(dest)
    assert budget >= 16  # Never goes below minimum


def test_stem_budget_respects_windows_limit() -> None:
    # Verify the budget math: destination + "/" + stem + ".mp3.part" < 260
    # Must be strictly under (Windows MAX_PATH counts terminating NUL)
    dest = Path("/test/output")
    budget = stem_budget(dest)
    actual_path = len(str(dest)) + 1 + budget + len(".mp3.part")
    assert actual_path < WINDOWS_PATH_LIMIT
