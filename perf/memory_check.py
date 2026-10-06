"""Measure the peak host memory of each r10 change, one fresh process per arm.

Each arm evolves a tile-check workload's plan in its own child process and
reports how far its peak resident memory rose while doing so; separate
processes keep one arm's peak from hiding another's. Arms: per-gate passes,
tiles where every target takes a bit, tiles where controls and diagonals take
none, the same with ``simplify``, and (where the prototype is built) the
Apple-GPU engine.

Usage, from the repository root:
    python perf/memory_check.py --qubits 24 --out results/memory-check.json
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("per_gate", "every_target", "insular", "simplify", "metal")


def child(arm: str, workload: str, n: int) -> None:
    """Run one arm and print its peak memory growth in MiB."""
    import resource  # pylint: disable=import-outside-toplevel  # POSIX only

    sys.path.insert(0, str(ROOT / "perf"))
    sys.path.insert(0, str(ROOT / "prototypes" / "metal"))
    # pylint: disable=import-outside-toplevel,import-error
    from tile_check import WORKLOADS, engine_classes, evolve

    from fatqat._backends.simplify import simplify_plan
    from fatqat.simulator import Simulator

    plan, _ = Simulator(
        "statevector", runtime="numpy"
    )._lower_program(  # pylint: disable=protected-access
        WORKLOADS[workload](n)
    )
    if arm == "metal":
        import metal_engine

        cls = metal_engine.MetalSVEngine
    else:
        cls = engine_classes("numba")["insular" if arm == "simplify" else arm]
    if arm == "simplify":
        plan = simplify_plan(plan, (2,) * n, zero_start=True)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    evolve(cls, n, plan, export=False)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in KiB on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    print(f"{(peak - before) * scale / 2**20:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qubits", type=int, default=24)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child", nargs=3, metavar=("ARM", "WORKLOAD", "QUBITS"))
    args = parser.parse_args()
    if args.child:
        child(args.child[0], args.child[1], int(args.child[2]))
        return 0
    sys.path.insert(0, str(ROOT / "perf"))
    # pylint: disable=import-outside-toplevel,import-error
    from tile_check import WORKLOADS, revision

    rows = []
    for workload in WORKLOADS:
        row = {"workload": workload, "qubits": args.qubits, "peak_growth_mib": {}}
        for arm in ARMS:
            completed = subprocess.run(
                [sys.executable, __file__, "--child", arm, workload, str(args.qubits)],
                capture_output=True, text=True, check=False, cwd=ROOT,
            )  # fmt: skip
            if completed.returncode == 0:
                row["peak_growth_mib"][arm] = float(completed.stdout.split()[-1])
            elif arm != "metal":  # the prototype is optional; everything else must run
                print(completed.stderr, file=sys.stderr)
                return 1
        rows.append(row)
        print(workload, row["peak_growth_mib"], flush=True)
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "description": __doc__.splitlines()[0],
                    "measurement_date": date.today().isoformat(),
                    "code_revision": revision(),
                    "state_mib": 2**args.qubits * 16 / 2**20,
                    "unit": "MiB of peak resident memory added while evolving the plan",
                    "rows": rows,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
