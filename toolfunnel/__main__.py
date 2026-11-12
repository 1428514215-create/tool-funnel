"""Allow ``python -m toolfunnel ...`` to act as the CLI.

This gives a zero-install entry point: with nothing but a Python interpreter and
this directory on ``sys.path`` (or the built ``.pyz``), ``python -m toolfunnel``
routes queries exactly like the installed ``tool-funnel`` command.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
