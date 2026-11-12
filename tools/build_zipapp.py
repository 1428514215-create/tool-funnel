"""Build a single-file, dependency-free executable: ``tool-funnel.pyz``.

A ``.pyz`` is a zipapp: a zip archive with a ``__main__.py`` that CPython runs
directly. It needs **no pip, no virtualenv and no network** -- just a Python 3.8+
interpreter. That makes it the most "download and run" distribution this project
can offer, and it sidesteps the one thing that otherwise blocks frozen/air-gapped
environments: installing the package.

Usage::

    python tools/build_zipapp.py            # writes dist/tool-funnel.pyz
    python dist/tool-funnel.pyz --version
    python dist/tool-funnel.pyz route examples/compose.example.json "查订单"

The archive contains the ``toolfunnel`` package plus the example catalogs, so a
downloaded ``.pyz`` can be exercised immediately without any other files.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipapp
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "dist", "tool-funnel.pyz")

ZIPAPP_MAIN = '''\
"""Entry point for the tool-funnel zipapp."""
import sys
from toolfunnel.cli import main

if __name__ == "__main__":
    sys.exit(main())
'''


def _copy_tree(src: str, dst: str, *, ignore=None) -> None:
    shutil.copytree(src, dst, ignore=ignore)


def build(out_path: str = DEFAULT_OUT, *, include_examples: bool = True) -> str:
    """Assemble the zipapp from the package source. Returns the output path."""
    staging = tempfile.mkdtemp(prefix="toolfunnel-zipapp-")
    try:
        # 1) The package itself (skip caches).
        pkg_src = os.path.join(ROOT, "toolfunnel")
        _copy_tree(
            pkg_src,
            os.path.join(staging, "toolfunnel"),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

        # 2) Bundle the example catalogs/demos so the archive is self-contained
        #    and immediately runnable (``--example`` uses them).
        if include_examples:
            ex_src = os.path.join(ROOT, "examples")
            ex_dst = os.path.join(staging, "examples")
            _copy_tree(
                ex_src, ex_dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "catalog.json"),
            )

        # 3) The zipapp entry point.
        with open(os.path.join(staging, "__main__.py"), "w", encoding="utf-8") as fh:
            fh.write(ZIPAPP_MAIN)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # ``main`` = the module the archive runs when executed.
        zipapp.create_archive(staging, target=out_path, interpreter="/usr/bin/env python3")
        return out_path
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = argv[0] if argv else DEFAULT_OUT
    path = build(out)
    size_kb = os.path.getsize(path) / 1024.0
    print(f"built {path} ({size_kb:.1f} KiB)")
    print("run it anywhere with a Python 3.8+ interpreter:")
    print(f"  python {path} --version")
    print(f'  python {path} route {os.path.join("examples", "compose.example.json")} "查订单"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
