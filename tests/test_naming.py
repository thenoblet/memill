from __future__ import annotations

from pathlib import Path

import pytest

from memill.errors import TransferError
from memill.naming import (
    WINDOWS_PATH_LIMIT,
    TrackTags,
    infer_tags,
    output_stem,
    sanitize_stem,
    stem_budget,
)


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
    # Also verify the total stays within limit
    actual_path = len(str(dest)) + 1 + budget + len(".mp3.part")
    assert actual_path < WINDOWS_PATH_LIMIT


def test_stem_budget_boundary_at_233_chars_returns_minimum() -> None:
    # Boundary test: destination length 233 leaves exactly 16 available (259 total)
    # Should return 16 without raising
    dest = Path("/" + "a" * 232)
    assert len(str(dest)) == 233
    budget = stem_budget(dest)
    assert budget == 16
    # Verify total is exactly at the usable limit
    actual_path = len(str(dest)) + 1 + budget + len(".mp3.part")
    assert actual_path == 259


def test_stem_budget_boundary_at_234_chars_raises() -> None:
    # Boundary test: destination length 234 leaves only 15 available (< MIN_STEM)
    # Should raise TransferError
    dest = Path("/" + "a" * 233)
    assert len(str(dest)) == 234
    with pytest.raises(TransferError) as exc_info:
        stem_budget(dest)
    assert "too long for a filename" in str(exc_info.value)


def test_stem_budget_excessive_destination_raises() -> None:
    # Test case from finding: 258-character destination leaves only -2 available
    # Should raise TransferError
    dest = Path("/" + "a" * 257)
    assert len(str(dest)) == 258
    with pytest.raises(TransferError) as exc_info:
        stem_budget(dest)
    assert "too long for a filename" in str(exc_info.value)


def test_stem_budget_respects_windows_limit() -> None:
    # Boundary test: destination long enough that DEFAULT_MAX_STEM cap doesn't apply
    # Destination length 201 gives budget 48, total exactly 259 (one under limit)
    dest = Path("/" + "a" * 200)
    budget = stem_budget(dest)
    actual_path = len(str(dest)) + 1 + budget + len(".mp3.part")
    assert actual_path == WINDOWS_PATH_LIMIT - 1


def test_prefers_explicit_music_metadata() -> None:
    tags = infer_tags(
        {
            "track": "Sankofa",
            "artist": "Gyakie",
            "album": "Sankofa EP",
            "release_year": 2021,
            "webpage_url": "https://example.test/watch?v=1",
        }
    )
    assert tags == TrackTags(
        title="Sankofa",
        artist="Gyakie",
        album="Sankofa EP",
        year="2021",
        source_url="https://example.test/watch?v=1",
    )


def test_strips_topic_suffix_from_uploader() -> None:
    tags = infer_tags({"title": "Sankofa", "uploader": "Gyakie - Topic"})
    assert tags.artist == "Gyakie"


def test_splits_artist_dash_title_and_drops_noise() -> None:
    tags = infer_tags(
        {
            "title": "Kofi Kinaata - Things Fall Apart (Official Video)",
            "uploader": "Chan",
        }
    )
    assert tags.artist == "Kofi Kinaata"
    assert tags.title == "Things Fall Apart"


def test_mix_title_without_separator_keeps_uploader_as_artist() -> None:
    tags = infer_tags(
        {"title": "GHANA AFROBEAT 2026 HOT MIX VOL. 4", "uploader": "DJ Sedan"}
    )
    assert tags.artist == "DJ Sedan"
    assert tags.title == "GHANA AFROBEAT 2026 HOT MIX VOL. 4"


def test_raw_mode_leaves_the_title_verbatim() -> None:
    info = {
        "title": "Kofi Kinaata - Things Fall Apart (Official Video)",
        "uploader": "Chan",
    }
    tags = infer_tags(info, clean=False)
    assert tags.title == "Kofi Kinaata - Things Fall Apart (Official Video)"
    assert tags.artist == "Chan"


def test_year_falls_back_to_upload_date() -> None:
    assert infer_tags({"title": "X", "upload_date": "20240711"}).year == "2024"


def test_output_stem_joins_artist_and_title() -> None:
    assert output_stem(TrackTags(title="Things Fall Apart", artist="Kofi Kinaata")) == (
        "Kofi Kinaata - Things Fall Apart"
    )


def test_output_stem_without_artist_is_just_the_title() -> None:
    assert output_stem(TrackTags(title="Some Mix")) == "Some Mix"


def test_output_stem_is_ntfs_safe() -> None:
    assert output_stem(TrackTags(title="A/B", artist="C:D")) == "C D - A B"


def test_pure_decoration_title_preserved() -> None:
    # When a title is entirely decoration, preserve it rather than
    # asserting an empty title. Regression test for noise stripper edge case.
    tags = infer_tags({"title": "(Official Video)"})
    assert tags.title == "(Official Video)"
    assert tags.title != ""


def test_explicit_track_not_split_even_with_dash() -> None:
    # Explicit track/artist fields from YouTube Music are trusted verbatim
    # and must never be split on dashes, even if they contain one.
    tags = infer_tags(
        {
            "track": "Intro - Reprise",
            "artist": "Gyakie",
            "uploader": "Some Channel",
        }
    )
    assert tags.title == "Intro - Reprise"
    assert tags.artist == "Gyakie"


def test_raw_mode_still_normalizes_topic_suffix() -> None:
    # clean=False disables title inference (e.g., no dash splitting),
    # but still strips the "- Topic" channel suffix because that is
    # channel-name normalisation, not inference: YouTube auto-generates
    # "<Artist> - Topic" channels, and including it in ID3 is never useful.
    tags = infer_tags(
        {"title": "X (Official Video)", "uploader": "Gyakie - Topic"},
        clean=False,
    )
    assert tags.title == "X (Official Video)"  # Title is verbatim (no cleaning)
    assert tags.artist == "Gyakie"  # But Topic suffix is still stripped
