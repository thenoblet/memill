"""Command-line surface."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from yt2mp3 import __version__
from yt2mp3.config import (
    DEFAULT_DESTINATION,
    DEFAULT_STAGING_ROOT,
    CbrQuality,
    Settings,
    VbrQuality,
    plan_concurrency,
)
from yt2mp3.encoder import require_ffmpeg
from yt2mp3.errors import DependencyError, Yt2Mp3Error
from yt2mp3.pipeline import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    BatchResult,
    run_batch,
)
from yt2mp3.reporting import select_reporter
from yt2mp3.source import Downloader
from yt2mp3.transfer import ARCHIVE_FILENAME, Archive

# A second, friendlier gate in front of the validation config.py already does.
# Reached first, so a typo becomes a usage message and exit 2 rather than the
# traceback a raw ValueError out of Settings construction would produce. The
# checks below defer to config.py wherever it already owns the rule, so the two
# gates cannot drift apart and disagree about what is valid.


def _vbr_level(text: str) -> int:
    try:
        level = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an integer 0-9, got {text!r}"
        ) from None
    try:
        return VbrQuality(level).level
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _bitrate(text: str) -> str:
    # Also rejects the empty string, which would otherwise be falsy at the
    # `if args.bitrate` branch and silently fall through to VBR.
    try:
        return CbrQuality(text).bitrate
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _job_count(text: str) -> int:
    try:
        jobs = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {text!r}"
        ) from None
    try:
        # Defers to config.py's floor rather than restating it. cpu_count is
        # unused when jobs is given, so any value serves.
        plan_concurrency(jobs, cpu_count=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return jobs


def _existing_file(text: str) -> Path:
    """Refuse a path that is plainly unusable, without opening it.

    exists(), not is_file(): a FIFO from process substitution
    (``--from-file <(...)``) and ``/dev/stdin`` are both perfectly good
    sources of a URL list, and both are ``is_file() == False``.

    Opening the path would classify those correctly, but opening a FIFO for
    reading blocks until a writer attaches -- measured, not assumed -- so the
    parser would hang on exactly the input this exists to accept.
    """
    path = Path(text)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"no such file: {text}")
    if path.is_dir():
        raise argparse.ArgumentTypeError(f"is a directory, not a file: {text}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt2mp3",
        description="Download YouTube audio and convert it to a tagged MP3 "
        "with ffmpeg.",
        epilog="Use this for content you have the rights to.",
    )
    parser.add_argument(
        "urls", nargs="*", help="video or playlist URLs; '-' reads stdin"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="destination directory",
    )
    parser.add_argument(
        "-j", "--jobs", type=_job_count, default=None, help="tracks in parallel"
    )

    quality = parser.add_mutually_exclusive_group()
    quality.add_argument(
        "-q", "--quality", type=_vbr_level, default=None, help="LAME VBR level 0-9"
    )
    quality.add_argument(
        "-b",
        "--bitrate",
        type=_bitrate,
        default=None,
        help="constant bitrate, e.g. 320k",
    )

    parser.add_argument("--normalize", action="store_true", help="apply loudnorm")
    parser.add_argument("--no-cover", action="store_true", help="do not embed artwork")
    parser.add_argument(
        "--raw-title", action="store_true", help="do not infer artist/title"
    )
    parser.add_argument(
        "--keep-source", action="store_true", help="also keep the original audio"
    )
    parser.add_argument(
        "--no-archive", action="store_true", help="do not skip finished tracks"
    )
    parser.add_argument(
        "--cookies-from-browser", default=None, help="e.g. firefox, chrome"
    )
    parser.add_argument(
        "--from-file",
        type=_existing_file,
        default=None,
        help="file of URLs, one per line",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be fetched"
    )
    parser.add_argument(
        "--plain", action="store_true", help="plain output, no live bars"
    )
    return parser


def settings_from_args(args: argparse.Namespace, *, cpu_count: int) -> Settings:
    jobs, fragments = plan_concurrency(args.jobs, cpu_count=cpu_count)
    return Settings(
        destination=args.output,
        staging_root=DEFAULT_STAGING_ROOT,
        quality=(
            # `is not None`, not truthiness: an empty bitrate must reach
            # CbrQuality and be refused, not fall through to VBR. The parser
            # already rejects it, but this function is public and reachable
            # with a namespace the parser never built.
            CbrQuality(args.bitrate)
            if args.bitrate is not None
            else VbrQuality(args.quality or 0)
        ),
        jobs=jobs,
        fragments=fragments,
        normalize=args.normalize,
        embed_cover=not args.no_cover,
        clean_titles=not args.raw_title,
        keep_source=args.keep_source,
        use_archive=not args.no_archive,
        cookies_from_browser=args.cookies_from_browser,
        dry_run=args.dry_run,
    )


def _lines(text: str) -> list[str]:
    return [
        stripped
        for line in text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]


def collect_urls(args: argparse.Namespace, stdin: TextIO) -> list[str]:
    """Positional URLs, then any file, then stdin if '-' was given."""
    urls = [url for url in args.urls if url != "-"]
    if args.from_file:
        urls += _lines(args.from_file.read_text("utf-8"))
    if "-" in args.urls:
        urls += _lines(stdin.read())
    return urls


def render_summary(result: BatchResult, stream: TextIO) -> None:
    counts = dict.fromkeys((STATUS_DONE, STATUS_SKIPPED, STATUS_FAILED), 0)
    for outcome in result.outcomes:
        # Direct subscript, not .get(): a status this summary cannot print
        # must fail here rather than be counted into a key nobody reads.
        counts[outcome.status] += 1
    stream.write(
        f"\n{counts[STATUS_DONE]} done, "
        f"{counts[STATUS_SKIPPED]} skipped, "
        f"{counts[STATUS_FAILED]} failed\n"
    )
    for outcome in result.failed:
        stream.write(f"  failed: {outcome.ref.title} - {outcome.error}\n")
    stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        require_ffmpeg()
    except DependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    urls = collect_urls(args, sys.stdin)
    if not urls:
        parser.error("no URLs given")

    settings = settings_from_args(args, cpu_count=os.cpu_count() or 1)
    downloader = Downloader(settings)

    try:
        refs = downloader.expand(urls)
    except Yt2Mp3Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # yt-dlp raises its own exception types, outside our hierarchy.
        print(f"error: could not resolve URLs: {exc}", file=sys.stderr)
        return 1

    if not refs:
        print("nothing to download", file=sys.stderr)
        return 1

    if settings.dry_run:
        for ref in refs:
            print(f"{ref.video_id}  {ref.title}")
        return 0

    archive_path = settings.destination / ARCHIVE_FILENAME
    try:
        archive = Archive(archive_path, enabled=settings.use_archive)
    except (OSError, UnicodeDecodeError) as exc:
        # This file lives in the user's library on an NTFS mount, so an
        # unreadable or non-UTF-8 archive is a thing that happens, not a bug.
        print(
            f"error: could not read the archive {archive_path}: {exc}",
            file=sys.stderr,
        )
        return 1
    reporter = select_reporter(stream=sys.stderr, force_plain=args.plain)
    interrupted = False
    aborted: str | None = None
    result = BatchResult()
    try:
        result = run_batch(
            refs,
            settings=settings,
            downloader=downloader,
            reporter=reporter,
            archive=archive,
        )
    except KeyboardInterrupt:
        interrupted = True
    except Yt2Mp3Error as exc:
        # A run-level abort: one bad track returns an outcome, but an
        # unreadable cookie store or a cancelled run propagates out of
        # run_batch by design, and stops everything.
        aborted = f"error: {exc}"
    except Exception as exc:
        # Deliberately broad, and only here. yt-dlp's own aborts (a mistyped
        # --cookies-from-browser raises CookieLoadError) reach this line, and
        # a user who mistypes a browser name deserves one clear message rather
        # than a stack trace. Catching it here rather than naming the type
        # keeps yt-dlp's existence a fact only source.py knows.
        #
        # Named differently from the handler above on purpose: errors.py's
        # contract is that a Yt2Mp3Error was anticipated and anything else is
        # a bug, and flattening both into one message would report a genuine
        # defect as though it were an ordinary bad URL.
        aborted = f"error: unexpected {type(exc).__name__}: {exc}"
    finally:
        # Nothing may be printed before this. While the live display runs, rich
        # replaces sys.stdout and sys.stderr with its own proxy, so anything
        # written goes through the console and is re-rendered rather than
        # delivered as written. Closing first keeps user-facing output verbatim.
        reporter.close()

    if interrupted:
        print("\ninterrupted", file=sys.stderr)
        return 130
    if aborted is not None:
        print(aborted, file=sys.stderr)
        return 1

    render_summary(result, sys.stderr)
    return result.exit_code
