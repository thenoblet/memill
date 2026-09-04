#!/usr/bin/env bash
# Create the tool's own virtualenv and link the launcher onto PATH.
#
# Safe to re-run. An existing .venv is reused rather than recreated, and an
# existing file at the link target is reported rather than overwritten --
# ~/.local/bin is a shared directory and clobbering a stranger's tool is not
# this script's business.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi

./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -e ".[dev]"

if ! command -v ffmpeg > /dev/null || ! command -v ffprobe > /dev/null; then
    echo "warning: ffmpeg and ffprobe are not on PATH, and yt2mp3 needs both" >&2
fi

launcher="$PWD/.venv/bin/yt2mp3"
link="$HOME/.local/bin/yt2mp3"
mkdir -p "$HOME/.local/bin"

if [[ -e "$link" || -L "$link" ]]; then
    if [[ -L "$link" && "$(readlink -f -- "$link")" == "$launcher" ]]; then
        echo "already linked -> $link"
    else
        echo "refusing to overwrite $link" >&2
        if [[ -L "$link" ]]; then
            echo "  it is a symlink to: $(readlink -- "$link")" >&2
        else
            echo "  it is an existing regular file" >&2
        fi
        echo "  remove it yourself, or just run $launcher" >&2
        exit 1
    fi
else
    ln -s -- "$launcher" "$link"
    echo "installed -> $link"
fi

case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) echo "note: $HOME/.local/bin is not on your PATH" >&2 ;;
esac

echo "run: yt2mp3 'https://youtube.com/watch?v=...'"
