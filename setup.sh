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

# Both sides resolved the same way. bash's $PWD is the *logical* path, so if
# this script is reached through a symlinked directory it does not spell the
# launcher the way readlink does -- and comparing one against the other makes
# the script refuse to overwrite the very link it created on the last run.
launcher_real="$(readlink -f -- "$launcher")"

if [[ -e "$link" || -L "$link" ]]; then
    if [[ -L "$link" && "$(readlink -f -- "$link")" == "$launcher_real" ]]; then
        echo "already linked -> $link"
    else
        echo "refusing to overwrite $link" >&2
        if [[ -L "$link" ]]; then
            echo "  it is a symlink to: $(readlink -- "$link")" >&2
        elif [[ -d "$link" ]]; then
            echo "  it is an existing directory" >&2
        else
            echo "  it is an existing file" >&2
        fi
        echo "  remove it yourself, or just run $launcher" >&2
        exit 1
    fi
else
    # The RESOLVED path, not $PWD's logical spelling: if this script was
    # reached through a symlinked directory, a link recording the logical
    # form goes stale the moment that directory is removed -- and the next
    # direct run compares it against $launcher_real, finds a mismatch and
    # refuses to overwrite the link it created itself.
    ln -s -- "$launcher_real" "$link"
    echo "installed -> $link"
fi

case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) echo "note: $HOME/.local/bin is not on your PATH" >&2 ;;
esac

echo "run: yt2mp3 'https://youtube.com/watch?v=...'"
