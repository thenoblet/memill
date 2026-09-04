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

That builds a virtualenv in `.venv`, installs the package into it, and links
`~/.local/bin/yt2mp3` at the launcher. It is safe to re-run; if something
unrelated already occupies that path it says so and stops rather than replacing
it.

## What it does

1. yt-dlp fetches the **audio-only** stream — a music video is roughly 50 MB of
   H.264 wrapped around 4 MB of audio, and this never downloads the video.
2. One ffmpeg pass encodes to MP3, writes ID3v2.3 tags (plus a v1 trailer for
   older players) and embeds the thumbnail as square cover art, centre-cropped
   from 16:9. Not three chained passes rewriting the file each time.
3. The finished file is copied to the library under a `.part` name and revealed
   by an atomic rename, so an interrupted run never leaves a truncated file
   wearing a finished name.

Downloading and encoding happen on the Linux filesystem (`~/.cache/yt2mp3`), not
on the mounted Windows drive. Measured on the development machine — ext4 against
the same machine's 9p `/mnt/e` mount, best of three runs each — that mount takes
0.55 s to write 64 MB against ext4's 0.03 s, and 2.8 s to write 400 fsynced
64 KB files against ext4's 0.6 s: roughly 18x slower for a bulk write and 5x
slower for the many small writes a fragmented download makes.

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
or ffprobe is absent. Nothing in the suite touches the network; a real download
is the one thing left that does, and that is your call to run.
