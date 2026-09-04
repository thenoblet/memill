"""Allow ``python -m memill``."""

from __future__ import annotations

import sys

from memill.cli import main

if __name__ == "__main__":
    sys.exit(main())
