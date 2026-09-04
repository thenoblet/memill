"""Allow ``python -m yt2mp3``."""

from __future__ import annotations

import sys

from yt2mp3.cli import main

if __name__ == "__main__":
    sys.exit(main())
