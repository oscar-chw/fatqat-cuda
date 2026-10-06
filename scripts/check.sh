#!/usr/bin/env bash
# The whole local gate: the test suite (CUDA tests skip without CuPy and a GPU), then the demo.
# Exits 0 only if both pass; a test run that collects nothing fails (pytest exit 5).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
echo "== tests"
"$PY" -m pytest -q -p no:cacheprovider -rfE --tb=short --disable-warnings
echo "== demo"
bash scripts/demo.sh
