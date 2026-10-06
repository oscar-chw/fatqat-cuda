"""Check that exact shot branching pays, with counts equal shot for shot.

Two arms of the same runtime run each workload's shots, alternating in one
process so drift and background load hit both alike:

- ``per_shot``: every shot evolved alone from the start (the previous loop);
- ``branching``: shots that share a state share the work on it.

Both must give equal counts on every repeat; branching may be slower on no
workload (beyond timing noise). Workloads:

- ``feedforward``: an ideal circuit with mid-circuit measurements and
  conditioned gates: few branches, many shots per branch;
- ``low_noise``: depolarizing noise (p = 0.002) after every gate;
- ``high_noise``: the same circuit at p = 0.05, where most shots soon run alone.

Usage:
    python perf/branching_check.py --runtimes numpy numba --out branching.json
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import fatqat as fq
import fatqat.operations as ops
from fatqat.noise import Depolarizing, NoiseModel
from fatqat.simulator import Simulator
from fatqat.simulator._engine import np as engine_np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# pylint: disable-next=import-error,wrong-import-position,wrong-import-order
from tile_check import revision

NOISE_FLOOR = 0.95


def _layers(program, n, layers):
    for layer in range(layers):
        for q in range(n):
            program.add(ops.RY(0.1 * (q + layer + 1)), q)
        for q in range(n - 1):
            program.add(ops.CX, (q, q + 1))


def feedforward(n):
    program = fq.Program(n, n)
    _layers(program, n, 2)
    for q in range(3):
        program.measure(q, q)
        program.add(ops.X, n - 1 - q, condition=(q, 1))
        _layers(program, n, 1)
    program.measure_all()
    return program, None


def _noisy(p):
    def build(n):
        noise = NoiseModel()
        noise.add(Depolarizing(p=p), operation=ops.CX)
        noise.add(Depolarizing(p=p), operation=ops.RY)
        program = fq.Program(n, n)
        _layers(program, n, 4)
        program.measure_all()
        return program, noise

    return build


WORKLOADS = {
    "feedforward": feedforward,
    "low_noise": _noisy(0.002),
    "high_noise": _noisy(0.05),
}


def _per_shot(engine, plan, seed_sequences, initial_occupied, initial_state):
    """The one-shot-at-a-time loop shot branching replaced."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    snapshots = []
    for seed_sequence in seed_sequences:
        engine.initialize(engine._dims, engine._n_clbits, initial_state=initial_state)
        snapshots.append(
            engine._run_one_shot(
                plan, np.random.default_rng(seed_sequence), initial_occupied
            )
        )
    return snapshots


def measure(runtime, workload, n, shots, repeats):
    program, noise = WORKLOADS[workload](n)
    backend = Simulator("statevector", runtime=runtime, noise=noise)
    config = {"seed": 3, "shot_parallelism": "serial"}
    if runtime == "numba":  # keep Numba off its own compiled multi-shot loop
        config["kernel_parallelism"] = "threads"
    shipped = engine_np._run_branched
    arms = {"per_shot": _per_shot, "branching": shipped}
    times = {arm: [] for arm in arms}
    counts = {}
    try:
        for repeat in range(repeats + 1):  # the first round warms up
            for arm, loop in arms.items():
                engine_np._run_branched = loop
                start = time.perf_counter()
                got = backend.run(program, shots=shots, simulation_config=config)
                got = got.result().get_counts()
                if repeat:
                    times[arm].append(time.perf_counter() - start)
                counts.setdefault(arm, got)
                assert got == counts[arm], f"{arm} counts changed between repeats"
    finally:
        engine_np._run_branched = shipped
    equal = counts["per_shot"] == counts["branching"]
    medians = {arm: statistics.median(t) for arm, t in times.items()}
    return {
        "runtime": runtime,
        "workload": workload,
        "qubits": n,
        "shots": shots,
        "repeats": repeats,
        "median_s": medians,
        "speedup": medians["per_shot"] / medians["branching"],
        "counts_equal": equal,
        "distinct_outcomes": len(counts["branching"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtimes", nargs="+", default=["numpy"])
    parser.add_argument("--qubits", type=int, nargs="+", default=[16])
    parser.add_argument("--shots", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    rows = []
    for runtime in args.runtimes:
        for n in args.qubits:
            for workload in WORKLOADS:
                row = measure(runtime, workload, n, args.shots, args.repeats)
                rows.append(row)
                print(json.dumps(row), flush=True)
    passed = all(r["counts_equal"] for r in rows) and all(
        r["speedup"] >= NOISE_FLOOR for r in rows
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "description": __doc__.splitlines()[0],
                    "measurement_date": date.today().isoformat(),
                    "code_revision": revision(),
                    "python": platform.python_version(),
                    "noise_floor": NOISE_FLOOR,
                    "passed": passed,
                    "rows": rows,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
