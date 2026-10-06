#!/usr/bin/env bash
# Accuracy on every runtime present, against a 60-digit reference (perf/precision.py), for all four methods:
# once as written and once with simplify=True. CPU-only machines measure NumPy and Numba; with CuPy and an
# NVIDIA GPU the CUDA runtime is measured too. The script aborts if a CPU runtime is off by more than 1e-12.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
echo "== round-off in machine epsilons, worst case over 2 seeded circuits per case"
"$PY" perf/precision.py --seeds 2 --out "$out/precision.json"
echo "== the same with simplify=True (the reference still evolves the unsimplified circuit)"
"$PY" perf/precision.py --seeds 2 --simplify --out "$out/precision-simplify.json"
