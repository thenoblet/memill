# yt2mp3

Pull the audio out of a YouTube video and land a tagged, cover-arted MP3 in your
music library. One command:

```bash
yt2mp3 "https://youtube.com/watch?v=..."
```

Use this for content you have the rights to.

## Install

Requires Python 3.11+ and ffmpeg (with `libmp3lame` and `mjpeg`).

```bash
./setup.sh
```

That builds a virtualenv in `.venv`, installs the package into it along with the
`[dev]` extra (pytest, mypy, ruff — what the Development commands below need),
and links `~/.local/bin/yt2mp3` at the launcher. It is safe to re-run; if
something unrelated already occupies that path it says so and stops rather than
replacing it.

## What it does

1. yt-dlp fetches the **audio-only** stream. A video track is far larger than
   the audio it carries, whatever resolution YouTube happens to serve, so
   skipping it is the single biggest saving in the run -- and this never
   downloads it.
2. One ffmpeg pass encodes to MP3, writes ID3v2.3 tags (plus a v1 trailer for
   older players) and embeds the thumbnail as square cover art, centre-cropped
   from 16:9. Not three chained passes rewriting the file each time.
3. The finished file is copied to the library under a `.part` name and revealed
   by an atomic rename, so an interrupted run never leaves a truncated file
   wearing a finished name.

Downloading and encoding happen on the Linux filesystem (`~/.cache/yt2mp3`), and
only the finished file is written to the mounted Windows drive. The mount is a
9p filesystem and is substantially slower than ext4 for both bulk writes and the
many small writes a fragmented download makes — repeated benchmarking puts the
direction beyond doubt but gives multipliers too unstable to quote, so none is
quoted here. Staging locally means a track crosses the mount once, at its final
size, instead of thousands of times while it downloads.

A track's staging directory is keyed on its video id, and is removed only once
that track finishes. A failed or interrupted one is left where it is, so the
next run resumes from the partial file yt-dlp wrote there instead of starting
the download again — a network blip at 81% of a three-hour mix should not cost
81% of a three-hour mix. The price is roughly one track's worth of disk per
unfinished track, reclaimed by the next successful run of the same track, or by
deleting `~/.cache/yt2mp3`.

Filenames are sanitised for NTFS rather than only for ext4, so a title that
Linux would happily accept cannot fail on arrival at the Windows mount, long
after the expensive download has already happened.

## Options

| Flag | Effect |
|---|---|
| `URL ...` | one or more video or playlist URLs; `-` reads them from stdin |
| `-o, --output DIR` | destination (default `/mnt/e/Media/Music/New`) |
| `-j, --jobs N` | tracks in parallel; fragments per track are derived to hold ~8 sockets |
| `-q, --quality 0..9` | LAME VBR level (default 0, roughly 245 kbps) |
| `-b, --bitrate 320k` | constant bitrate instead — mutually exclusive with `-q` |
| `--normalize` | single-pass loudnorm to -14 LUFS, for consistent volume across sources |
| `--cookies-from-browser firefox` | age-restricted or region-locked material |
| `--from-file urls.txt` | batch from a list; blank lines and `#` comments are skipped |
| `--keep-source` | keep the original Opus/M4A alongside the MP3 |
| `--no-cover` | do not embed artwork |
| `--raw-title` | keep the video title verbatim instead of inferring artist and title |
| `--no-archive` | do not skip tracks already recorded as finished |
| `--dry-run` | list what would be fetched and download nothing |
| `--plain` | plain line-by-line output instead of live progress bars |
| `--version` | print the version and exit |

Finished tracks are recorded in `.yt2mp3-archive` in the destination directory,
so re-running on a playlist skips what you already have. `--no-archive` turns
that off.

Exit status is 0 when everything landed, 1 when any track failed or the run was
aborted, 2 for a usage error or a missing ffmpeg, and 130 on Ctrl-C.

## Why VBR V0 by default

YouTube's audio is already lossy (Opus ~160k, AAC ~128k). 320k CBR spends about
30% more bytes padding detail the source never had. V0 is the honest sweet spot
for transcoded material; `-b 320k` is there if you want it anyway.

## Development

```bash
.venv/bin/pytest        # no network required
.venv/bin/ruff check .
.venv/bin/mypy src
```

`tests/test_end_to_end.py` is the only suite that shells out to ffmpeg. It
synthesises its own tone and artwork with `lavfi` and skips itself when ffmpeg
or ffprobe is absent. It needs three encoders the tool itself never uses:
`mpeg4`, to build a muxed audio+video fixture; `png`, for the cover fixtures;
and `rawvideo`, to read a pixel back out. Nothing in the suite touches the
network; a real download is the one thing left that does, and that is your call
to run.
