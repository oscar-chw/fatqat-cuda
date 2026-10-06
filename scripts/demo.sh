#!/usr/bin/env bash
# Accuracy on every runtime present, against a 60-digit reference (perf/precision.py), for all four methods.
# CPU-only machines measure NumPy and Numba; with CuPy and an NVIDIA GPU the CUDA runtime is measured too.
# Fails if any runtime's worst error exceeds the README's "<= 3.1 eps on every runtime" (precision.py itself
# aborts only past its 1e-12 control, about 4,500 eps, so that bound alone would not test the claim).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
echo "== round-off in machine epsilons, worst case over 2 seeded circuits per case"
"$PY" perf/precision.py --seeds 2 --out "$out/precision.json"
echo "== every runtime within the README's 3.1 eps"
"$PY" - "$out/precision.json" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))["rows"]
worst = {}
for row in rows:
    for seed in row["seeds"]:
        for arm, err in seed["errors"].items():
            worst[arm] = max(worst.get(arm, 0.0), err["linf_eps"])
if not worst:
    sys.exit("no runtime was measured")
for arm, eps in sorted(worst.items()):
    print(f"  {arm:6s} worst {eps:.2f} eps  {'ok' if eps <= 3.1 else 'OVER 3.1'}")
sys.exit(0 if all(eps <= 3.1 for eps in worst.values()) else 1)
EOF
