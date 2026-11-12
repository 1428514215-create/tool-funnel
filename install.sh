#!/usr/bin/env bash
# tool-funnel one-step installer / launcher.
#
# Usage:
#   ./install.sh              # install into the current Python environment (pip)
#   ./install.sh --user       # install for the current user only
#   ./install.sh --zipapp     # build a standalone dist/tool-funnel.pyz instead
#   ./install.sh --venv       # create .venv and install there
#
# It is deliberately dependency-free and works offline for the --zipapp path
# (a .pyz needs only a Python interpreter).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "Python 3 not found (set PYTHON=...)"; exit 1; }

MODE="pip"
USER_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --user) USER_FLAG="--user" ;;
    --zipapp) MODE="zipapp" ;;
    --venv) MODE="venv" ;;
    -h|--help)
      grep '^#' "$0" | sed -e 's/^# \{0,1\}//' -e '/^!\/usr\/bin\/env/d' | head -n 12
      exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "tool-funnel installer"
echo "  python: $("$PY" --version 2>&1)"
echo "  mode:   $MODE"
echo

case "$MODE" in
  zipapp)
    "$PY" tools/build_zipapp.py
    echo
    echo "Done. Run it anywhere (no install needed):"
    echo "  $PY dist/tool-funnel.pyz demo"
    ;;
  venv)
    "$PY" -m venv .venv
    # shellcheck disable=SC1091
    . .venv/bin/activate
    python -m pip install --upgrade pip >/dev/null 2>&1 || true
    python -m pip install -e .
    echo
    echo "Done. Activate and run:"
    echo "  source .venv/bin/activate && tool-funnel demo"
    ;;
  pip)
    if "$PY" -m pip install -e . $USER_FLAG 2>/dev/null; then
      echo
      echo "Done. Try:  tool-funnel demo"
    else
      echo "pip install failed (no network or an externally-managed env)." >&2
      echo "Falling back to the standalone zipapp..." >&2
      "$PY" tools/build_zipapp.py
      echo
      echo "Done. Run it anywhere:"
      echo "  $PY dist/tool-funnel.pyz demo"
    fi
    ;;
esac
