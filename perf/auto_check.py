"""Check that runtime="auto" picks the faster of Numba and CUDA at every size.

For statevectors and density matrices across sizes, the same circuit (two
layers of RY and RZ on every qubit plus a ring of CX, measured) runs on
Numba, on CUDA (one GPU) and on "auto" (which may use every visible GPU),
arms alternating in one process after a warm-up, medians of the repeats.
The crossover where CUDA overtakes Numba is reported. The check fails when
the runtime "auto" chose is slower than the other by more than
``TOLERANCE``, judged on the two fixed arms' own medians (auto's own time
runs the chosen runtime's code, so it differs from that arm by timing noise
alone, about 20% on runs of a few milliseconds), or when its counts differ
from the chosen runtime's.

Usage, on a machine with a CUDA GPU:
    python perf/auto_check.py --out results/auto-check.json
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# pylint: disable=wrong-import-position,import-error  # path set up above
from tile_check import revision  # noqa: E402
import fatqat as fq  # noqa: E402
import fatqat.operations as ops  # noqa: E402
from fatqat.simulator import Simulator  # noqa: E402

TOLERANCE = 1.15
SIZES = {
    "statevector": [10, 12, 14, 16, 18, 20, 22, 24],
    "density_matrix": [5, 6, 7, 8, 9, 10, 11, 12],
}


def circuit(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    program = fq.Program(n, n)
    for _ in range(2):
        for q in range(n):
            program.add(ops.RY(float(rng.uniform(0, np.pi))), q)
            program.add(ops.RZ(float(rng.uniform(0, np.pi))), q)
        for q in range(n):
            program.add(ops.CX, (q, (q + 1) % n))
    program.measure_all()
    return program


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows, failures = [], []
    for method, sizes in SIZES.items():
        backends = {
            name: Simulator(method, runtime=name) for name in ("numba", "cuda", "auto")
        }
        for n in sizes:
            program = circuit(n)
            options = {"shots": args.shots, "simulation_config": {"seed": 3}}
            results = {
                name: b.run(program, **options).result() for name, b in backends.items()
            }
            chosen = results["auto"].metadata["runtime"]
            if results["auto"].get_counts() != results[chosen].get_counts():
                failures.append(f"{method} {n}: auto's counts differ from {chosen}'s")
            times = {name: [] for name in backends}
            for repeat in range(args.repeats):
                order = list(backends) if repeat % 2 == 0 else list(backends)[::-1]
                for name in order:
                    start = time.perf_counter()
                    backends[name].run(program, **options).result()
                    times[name].append(time.perf_counter() - start)
            median = {name: statistics.median(t) for name, t in times.items()}
            best = min(median["numba"], median["cuda"])
            row = {
                "method": method,
                "qubits": n,
                "median_s": median,
                "auto_chose": chosen,
                "chosen_vs_best": median[chosen] / best,
                "auto_vs_best": median["auto"] / best,
            }
            rows.append(row)
            print(
                f"{method:15s} {n:2d}  numba {median['numba']:.4f}s  cuda {median['cuda']:.4f}s"
                f"  auto {median['auto']:.4f}s ({chosen})  chosen/best {row['chosen_vs_best']:.2f}",
                flush=True,
            )
            if row["chosen_vs_best"] > TOLERANCE:
                failures.append(
                    f"{method} {n}: auto chose {chosen}, "
                    f"{row['chosen_vs_best']:.2f}x the faster runtime"
                )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "description": "runtime='auto' against Numba and CUDA (one GPU) on the same circuit, arms alternating; auto may use every visible GPU.",
                    "measurement_date": date.today().isoformat(),
                    "code_revision": revision(),
                    "tolerance": TOLERANCE,
                    "rows": rows,
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
