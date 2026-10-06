"""Check that controls and diagonal gates riding free in tiles pay, bit for bit.

Three arms of the same engine run each workload's lowered plan, alternating
in one process so drift and background load hit all of them alike:

- ``per_gate``: tiling off, one pass over the state per gate;
- ``every_target``: tiles in which every target takes a tile bit, the previous
  rule (gates on more than two qubits are not tiled);
- ``insular``: tiles in which only targets a gate actually moves take a bit;
  controls and diagonal gates are read from each amplitude's index.

All three must give equal states (exact array equality); ``insular`` may be
slower than ``every_target`` on no workload (beyond timing noise); and it
must beat it by ``MIN_SPEEDUP`` on the Fourier transform, whose controlled
phases the rule takes out of the tile (47 passes become 7 on CUDA's tiles).
The other workloads are reported, not gated: their gates sit on neighbouring
qubits, so the old rule already packed them (the adder needs 7 passes either
way on Numba's tiles, 13 against 11 on CUDA's), and what they gain comes from
visiting only the amplitudes a gate's controls select.

Usage:
    python perf/tile_check.py --runtimes numba --out results/tile-check.json
    python perf/tile_check.py --runtimes numba cuda --out results/tile-check.json
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

import fatqat as fq
import fatqat.operations as ops
from fatqat.simulator import Simulator
from fatqat.simulator._engine.base import _TileForm

MIN_SPEEDUP = 1.5
NOISE_FLOOR = 0.95
_GATED = {"numba": ("qft",), "cuda": ("qft",)}


# --- workloads -------------------------------------------------------------------


def qft(n: int) -> fq.Program:
    program = fq.Program(n)
    for q in range(n):
        program.add(ops.RY(0.3 + 0.01 * q), q)
    for q in range(n):
        program.add(ops.H, q)
        for k, r in enumerate(range(q + 1, n), start=2):
            program.add(ops.CPhase(2 * np.pi / 2**k), (r, q))
    return program


def _toffoli(program, a, b, c):
    for name, qubits in (
        ("H", (c,)), ("CX", (b, c)), ("Tdg", (c,)), ("CX", (a, c)), ("T", (c,)),
        ("CX", (b, c)), ("Tdg", (c,)), ("CX", (a, c)), ("T", (b,)), ("T", (c,)),
        ("H", (c,)), ("CX", (a, b)), ("T", (a,)), ("Tdg", (b,)), ("CX", (a, b)),
    ):  # fmt: skip
        program.add(getattr(ops, name), qubits)


def adder(n: int) -> fq.Program:
    """Cuccaro ripple-carry adder in Clifford+T on ``n`` qubits, inputs in superposition."""
    bits = (n - 2) // 2
    program = fq.Program(n)
    for q in range(1, 2 * bits + 1):
        program.add(ops.H, q)
    chain = [0] + list(range(1, 2 * bits + 1))
    for i in range(bits):
        x, y, w = chain[2 * i], chain[2 * i + 1], chain[2 * i + 2]
        program.add(ops.CX, (w, y))
        program.add(ops.CX, (w, x))
        _toffoli(program, x, y, w)
    program.add(ops.CX, (chain[-1], n - 1))
    for i in reversed(range(bits)):
        x, y, w = chain[2 * i], chain[2 * i + 1], chain[2 * i + 2]
        _toffoli(program, x, y, w)
        program.add(ops.CX, (w, x))
        program.add(ops.CX, (x, y))
    return program


def qaoa(n: int, layers: int = 2) -> fq.Program:
    rng = np.random.default_rng(11)
    edges = [(q, (q + 1) % n) for q in range(n)] + [
        tuple(int(v) for v in rng.choice(n, 2, replace=False)) for _ in range(n)
    ]
    program = fq.Program(n)
    for q in range(n):
        program.add(ops.H, q)
    for layer in range(layers):
        for a, b in edges:
            program.add(ops.CX, (a, b))
            program.add(ops.RZ(0.4 + 0.1 * layer), b)
            program.add(ops.CX, (a, b))
        for q in range(n):
            program.add(ops.RX(0.7 - 0.1 * layer), q)
    return program


def clifford_t(n: int, depth: int = 600) -> fq.Program:
    rng = np.random.default_rng(12)
    program = fq.Program(n)
    for _ in range(depth):
        q = int(rng.integers(n))
        if rng.random() < 0.35:
            r = int(rng.integers(n - 1))
            program.add(ops.CX, (q, r + (r >= q)))
        else:
            program.add(getattr(ops, str(rng.choice(["H", "S", "T", "Tdg", "X"]))), q)
    return program


WORKLOADS = {"qft": qft, "adder": adder, "qaoa": qaoa, "clifford_t": clifford_t}


# --- arms ----------------------------------------------------------------------------


def engine_classes(runtime: str) -> dict[str, type]:
    if runtime == "numba":
        from fatqat.simulator._engine.nb import (  # pylint: disable=import-outside-toplevel
            NumbaSVEngine as Engine,
        )

        threshold = {"_TILE_MIN_BYTES": 0}
    else:
        from fatqat.simulator._engine.cupy import (  # pylint: disable=import-outside-toplevel
            CupySVEngine as Engine,
        )

        threshold = {"_TILE_MIN_BYTES": 0}

    def every_target(self, step):
        # The previous rule: every target takes a tile bit, at most two.
        width = len(step.target_indices)
        if width > 2:
            return None
        matrix = np.asarray(step.matrix, dtype=np.complex128)
        return _TileForm(False, (), tuple(range(width)), matrix)

    return {
        "per_gate": type("PerGate", (Engine,), {"_TILE_BITS": 64}),
        "every_target": type(
            "EveryTarget", (Engine,), {**threshold, "_tile_form_of": every_target}
        ),
        "insular": type("Insular", (Engine,), threshold),
    }


def evolve(engine_cls, n: int, plan, *, export: bool = True):
    """Run ``plan``; return the state on the host, or ``None`` after a sync.

    Timed calls do not export: copying a large state to the host would add
    the same constant to every arm and hide the kernels being compared.
    """
    engine = engine_cls()
    engine.initialize((2,) * n)
    for step in plan:
        engine.apply(step)
    state = engine.state  # applies any queued tile batch
    if not export:
        if hasattr(state, "get"):  # a CuPy array: wait for its kernels
            state.device.synchronize()
        return None
    state = engine.export_state()
    return state.get() if hasattr(state, "get") else np.asarray(state)


def count_passes(engine_cls, n: int, plan) -> int:
    """Passes over the full state: tile batches plus gates applied alone."""
    count = [0]

    class Counted(engine_cls):
        def _apply_tile_batch(self, steps):
            count[0] += 1
            super()._apply_tile_batch(steps)

        def _apply_now(self, step):
            count[0] += 1
            super()._apply_now(step)

    evolve(Counted, n, plan)
    return count[0]


def measure(runtime: str, n: int, repeats: int) -> list[dict]:
    arms = engine_classes(runtime)
    rows = []
    for name, build in WORKLOADS.items():
        program = build(n)
        plan, _ = Simulator(
            "statevector", runtime="numpy"
        )._lower_program(  # pylint: disable=protected-access
            program
        )
        states = {arm: evolve(cls, n, plan) for arm, cls in arms.items()}  # warm
        identical = all(
            np.array_equal(states["per_gate"], state) for state in states.values()
        )
        del states
        passes = {
            arm: count_passes(arms[arm], n, plan) for arm in ("every_target", "insular")
        }
        times = {arm: [] for arm in arms}
        for _ in range(repeats):
            for arm, cls in arms.items():
                start = time.perf_counter()
                evolve(cls, n, plan, export=False)
                times[arm].append(time.perf_counter() - start)
        median = {arm: statistics.median(t) for arm, t in times.items()}
        row = {
            "workload": name,
            "runtime": runtime,
            "qubits": n,
            "steps": len(plan),
            "identical": identical,
            "tile_passes": passes,
            "median_s": median,
            "insular_over_every_target": median["every_target"] / median["insular"],
            "insular_over_per_gate": median["per_gate"] / median["insular"],
        }
        rows.append(row)
        print(
            f"{name:11s} {runtime:6s} {len(plan):5d} steps  identical={identical}  "
            f"passes {passes['every_target']}->{passes['insular']}  "
            f"vs every-target {row['insular_over_every_target']:.2f}x  "
            f"vs per-gate {row['insular_over_per_gate']:.2f}x",
            flush=True,
        )
    return rows


def verdict(rows: list[dict]) -> list[str]:
    failures = []
    for row in rows:
        if not row["identical"]:
            failures.append(f"{row['runtime']} {row['workload']}: arms differ")
        passes = row["tile_passes"]
        if passes["insular"] > passes["every_target"]:
            failures.append(
                f"{row['runtime']} {row['workload']}: more passes than before"
            )
        if row["workload"] == "qft" and passes["insular"] * 4 > passes["every_target"]:
            failures.append(f"{row['runtime']} qft: passes not cut at least fourfold")
        ratio = row["insular_over_every_target"]
        if row["workload"] in _GATED[row["runtime"]] and ratio < MIN_SPEEDUP:
            failures.append(f"{row['runtime']} {row['workload']}: below {MIN_SPEEDUP}x")
        if ratio < NOISE_FLOOR:
            failures.append(f"{row['runtime']} {row['workload']}: slower than before")
    return failures


def revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        # A source export (git archive) has no .git; the runner passes the
        # commit it exported so the result still names the code it measured.
        return os.environ.get("FATQAT_CODE_REVISION")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--runtimes", nargs="+", choices=("numba", "cuda"), default=["numba"]
    )
    parser.add_argument("--qubits", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    rows = []
    for runtime in args.runtimes:
        rows += measure(runtime, args.qubits, args.repeats)
    failures = verdict(rows)
    output = {
        "description": "Tiles where controls and diagonal gates take no tile bit, against tiles where every target does and against per-gate passes; same engine, arms alternating in one process.",
        "measurement_date": date.today().isoformat(),
        "code_revision": revision(),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "rows": rows,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
