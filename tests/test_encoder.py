from __future__ import annotations

from pathlib import Path

import pytest

from yt2mp3.config import CbrQuality, VbrQuality
from yt2mp3.encoder import ProgressParser, build_encode_command
from yt2mp3.naming import TrackTags

AUDIO = Path("/stage/a.opus")
OUT = Path("/stage/out.mp3")
TAGS = TrackTags(title="Things Fall Apart", artist="Kofi Kinaata", source_url="https://u")


def _cmd(**kwargs: object) -> list[str]:
    base: dict[str, object] = {
        "audio": AUDIO,
        "output": OUT,
        "quality": VbrQuality(0),
        "tags": TAGS,
    }
    return build_encode_command(**{**base, **kwargs})  # type: ignore[arg-type]


def test_without_cover_there_is_one_input_and_no_video_stream() -> None:
    argv = _cmd()
    assert argv.count("-i") == 1
    assert "attached_pic" not in argv
    assert "-c:v" not in argv


def test_with_cover_the_image_is_mapped_and_marked_attached() -> None:
    argv = _cmd(cover=Path("/stage/c.webp"))
    assert argv.count("-i") == 2
    assert argv[argv.index("-disposition:v") + 1] == "attached_pic"
    assert argv[argv.index("-c:v") + 1] == "mjpeg"


def test_map_metadata_clear_precedes_the_tags_it_could_clobber() -> None:
    argv = _cmd()
    assert argv.index("-map_metadata") < argv.index("-metadata")


def test_quality_flags_come_from_the_quality_object() -> None:
    vbr = _cmd(quality=VbrQuality(0))
    assert vbr[vbr.index("-q:a") + 1] == "0"
    cbr = _cmd(quality=CbrQuality("320k"))
    assert cbr[cbr.index("-b:a") + 1] == "320k"
    assert "-q:a" not in cbr


def test_normalize_adds_a_single_pass_loudnorm_filter() -> None:
    argv = _cmd(normalize=True)
    assert argv[argv.index("-af") + 1].startswith("loudnorm=")


def test_tags_are_emitted_as_key_equals_value() -> None:
    argv = _cmd()
    assert "title=Things Fall Apart" in argv
    assert "artist=Kofi Kinaata" in argv
    assert "comment=https://u" in argv


def test_absent_tags_are_omitted_entirely() -> None:
    argv = _cmd(tags=TrackTags(title="Only"))
    assert not [item for item in argv if item.startswith("artist=")]


def test_output_is_the_final_argument() -> None:
    assert _cmd()[-1] == str(OUT)


def test_out_time_becomes_a_fraction_of_the_duration() -> None:
    parser = ProgressParser(duration=100.0)
    assert parser.feed("out_time_us=50000000") == pytest.approx(0.5)


def test_unrelated_keys_report_nothing() -> None:
    assert ProgressParser(duration=100.0).feed("bitrate=245.0kbits/s") is None


def test_malformed_lines_are_ignored_rather_than_raising() -> None:
    parser = ProgressParser(duration=100.0)
    assert parser.feed("out_time_us=") is None
    assert parser.feed("garbage") is None


def test_overrun_is_clamped_to_one() -> None:
    assert ProgressParser(duration=10.0).feed("out_time_us=20000000") == 1.0


def test_end_marker_completes_even_without_a_known_duration() -> None:
    assert ProgressParser(duration=None).feed("progress=end") == 1.0


def test_unknown_duration_cannot_produce_a_fraction() -> None:
    assert ProgressParser(duration=None).feed("out_time_us=5000000") is None
