"""Check that runtime="metal" is faster than Numba alone, bit for bit.

Arms, alternating in one process (a warm-up round first, medians of the
repeats): Numba alone with its cache tiles, and the Metal engine, whose tile
batches the Apple GPU shares with Numba, with its adaptive share and at fixed
shares. Every arm's final state must have Numba's bit patterns (which tell
-0.0 from +0.0). A memory check creates and drops the engine's state 100
times: the resident set must stay flat, or a Metal buffer leaks.

Exit status is nonzero when a state differs, when the adaptive share is
slower than Numba on any workload, or when the memory check grows.

Usage, on a Mac with an Apple GPU and fatqat[metal]:
    python perf/metal_check.py --qubits 24 26 --out results/metal-check.json
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# pylint: disable=wrong-import-position,import-error  # path set up above
from tile_check import WORKLOADS, revision  # noqa: E402
from fatqat.simulator import Simulator  # noqa: E402
from fatqat.simulator._engine.metal import MetalSVEngine  # noqa: E402
from fatqat.simulator._engine.nb import NumbaSVEngine  # noqa: E402

# Leak allowance over 100 cycles of a 20-qubit state (16 MiB each).
MAX_GROWTH_MIB = 8


def fixed(share):
    class Fixed(MetalSVEngine):
        def __init__(self):
            super().__init__()
            self.gpu_share = share

        def _learn(self, *args):
            del args

    return Fixed


def evolve(cls, n, plan):
    engine = cls()
    engine.initialize((2,) * n)
    for step in plan:
        engine.apply(step)
    return np.array(engine.state)


def rss_mib() -> float:
    # ru_maxrss is bytes on macOS.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def memory_check(n: int = 20, cycles: int = 100) -> dict:
    class Always(MetalSVEngine):
        _TILE_MIN_BYTES = 0

    engine = Always()
    for _ in range(5):  # warm: the context and the first buffers
        engine.initialize((2,) * n)
    before = rss_mib()
    for _ in range(cycles):
        engine.initialize((2,) * n)
    growth = rss_mib() - before
    return {"qubits": n, "cycles": cycles, "peak_rss_growth_mib": growth}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qubits", type=int, nargs="+", default=[24, 26])
    parser.add_argument("--shares", type=float, nargs="+", default=[0.25, 0.4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    failures = []
    rows = []
    for n in args.qubits:
        for name, build in WORKLOADS.items():
            plan, _ = Simulator("statevector", runtime="numpy")._lower_program(
                build(n)
            )  # pylint: disable=protected-access
            arms = {"numba": NumbaSVEngine, "metal_adaptive": MetalSVEngine}
            arms.update({f"metal_{s:.2f}": fixed(s) for s in args.shares})
            reference = evolve(NumbaSVEngine, n, plan).view(np.uint64)
            identical = {
                arm: bool(
                    np.array_equal(evolve(cls, n, plan).view(np.uint64), reference)
                )
                for arm, cls in arms.items()
            }
            times = {arm: [] for arm in arms}
            for repeat in range(args.repeats):
                order = list(arms) if repeat % 2 == 0 else list(arms)[::-1]
                for arm in order:
                    start = time.perf_counter()
                    evolve(arms[arm], n, plan)
                    times[arm].append(time.perf_counter() - start)
            medians = {arm: statistics.median(t) for arm, t in times.items()}
            speedup = {
                arm: medians["numba"] / m
                for arm, m in medians.items()
                if arm != "numba"
            }
            rows.append(
                {
                    "workload": name,
                    "qubits": n,
                    "steps": len(plan),
                    "bit_identical_to_numba": identical,
                    "median_s": medians,
                    "speedup_over_numba": speedup,
                }
            )
            print(
                f"{name:11s} {n} qubits identical={all(identical.values())} "
                + " ".join(f"{a} {r:.2f}x" for a, r in speedup.items()),
                flush=True,
            )
            if not all(identical.values()):
                failures.append(f"{name} at {n} qubits: a state differs from Numba's")
            if speedup["metal_adaptive"] < 1.0:
                failures.append(f"{name} at {n} qubits: the adaptive share is slower")
    memory = memory_check()
    print(f"memory: {memory}", flush=True)
    if memory["peak_rss_growth_mib"] > MAX_GROWTH_MIB:
        failures.append("the memory check grew: a Metal buffer leaks")
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "description": "runtime='metal' (software binary64 tiles on the Apple GPU, sharing each tile batch with Numba on the CPU) against Numba alone; same process, arms alternating.",
                    "measurement_date": date.today().isoformat(),
                    "code_revision": revision(),
                    "equality": "bit patterns (tells -0.0 from +0.0), against Numba alone",
                    "rows": rows,
                    "memory": memory,
                    "failures": failures,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
    for failure in failures:
        print("FAIL:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
