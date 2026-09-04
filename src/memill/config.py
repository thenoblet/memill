"""Runtime configuration.

A leaf module: it imports nothing else from the package, so every other module
is free to depend on it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# A universal default: every desktop Linux and WSL install has ~/Music, and a
# published tool cannot assume anyone else's mount points. Set MEMILL_OUTPUT to
# point at your own library once instead of passing -o on every run.
DEFAULT_DESTINATION = Path.home() / "Music"
OUTPUT_ENV_VAR = "MEMILL_OUTPUT"


def default_destination(environ: Mapping[str, str] | None = None) -> Path:
    """Where tracks land unless ``-o`` overrides it.

    Resolved at call time rather than import time so the environment can be
    changed by a test, and so a shell that exports MEMILL_OUTPUT after the
    package is imported still wins.
    """
    env = os.environ if environ is None else environ
    configured = env.get(OUTPUT_ENV_VAR)
    return Path(configured).expanduser() if configured else DEFAULT_DESTINATION
DEFAULT_STAGING_ROOT = Path.home() / ".cache" / "memill"

# Total concurrent HTTP connections. Four tracks times eight fragments is
# thirty-two sockets, which is how you get throttled rather than how you go
# fast, so jobs and fragments are derived from one another to hold this line.
CONNECTION_BUDGET = 8

_BITRATE = re.compile(r"^\d{1,4}k$")


@dataclass(frozen=True, slots=True)
class VbrQuality:
    """LAME variable bitrate. ``level`` 0 is best, 9 is smallest."""

    level: int

    def __post_init__(self) -> None:
        if not 0 <= self.level <= 9:
            raise ValueError(f"VBR level must be 0..9, got {self.level}")

    def ffmpeg_args(self) -> tuple[str, ...]:
        return ("-q:a", str(self.level))


@dataclass(frozen=True, slots=True)
class CbrQuality:
    """Constant bitrate, expressed the way ffmpeg wants it, e.g. ``320k``."""

    bitrate: str

    def __post_init__(self) -> None:
        if not _BITRATE.match(self.bitrate):
            raise ValueError(f"bitrate must look like '320k', got {self.bitrate!r}")

    def ffmpeg_args(self) -> tuple[str, ...]:
        return ("-b:a", self.bitrate)


Quality = VbrQuality | CbrQuality


def plan_concurrency(
    jobs: int | None, *, cpu_count: int, budget: int = CONNECTION_BUDGET
) -> tuple[int, int]:
    """Return ``(jobs, fragments_per_job)`` holding total sockets to ``budget``.

    The floor is 1, not 2. A floor of 2 lets the product grow without bound once
    ``jobs`` exceeds the budget -- 16 jobs would open 32 sockets, precisely the
    throttling this budget exists to prevent. One fragment per job is yt-dlp's
    own default and costs nothing when track-level parallelism already supplies
    the concurrency.
    """
    if jobs is not None and jobs < 1:
        raise ValueError("jobs must be at least 1")
    resolved = jobs if jobs is not None else min(4, max(1, cpu_count))
    fragments = max(1, budget // resolved)
    return resolved, fragments


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the pipeline needs, fixed for the duration of one run."""

    destination: Path
    staging_root: Path
    quality: Quality
    jobs: int
    fragments: int
    normalize: bool = False
    embed_cover: bool = True
    clean_titles: bool = True
    keep_source: bool = False
    use_archive: bool = True
    cookies_from_browser: str | None = None
    dry_run: bool = False
