"""Time one CUDA parameter sweep on 1, 2, ... N devices and check identical results.

The workload is a layered RY/CX/RZ ansatz with one parameter per qubit per
layer, measured on every qubit, run as a ``run_sweep`` with a fixed seed. Each
device count runs the whole sweep ``--repeats`` times after one warm-up; the
median wall time is reported, and every configuration's counts must equal the
single-device counts exactly.

Usage:
    python perf/sweep.py --devices 0 1 --qubits 24 --rows 32 --out sweep.json
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

import fatqat as fq
import fatqat.operations as ops
from fatqat.parameters import Parameter
from fatqat.simulator import Simulator


def ansatz(n: int, layers: int):
    program = fq.Program(n, n)
    parameters = []
    for layer in range(layers):
        for q in range(n):
            theta = Parameter(f"t{layer}_{q}")
            parameters.append(theta)
            program.add(ops.RY(theta), q)
        for q in range(layer % 2, n - 1, 2):
            program.add(ops.CX, (q, q + 1))
        for q in range(n):
            program.add(ops.RZ(0.1 * (q + 1)), q)
    program.measure_all()
    return program, parameters


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
        return os.environ.get("FATQAT_CODE_REVISION")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--qubits", type=int, default=24)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    program, parameters = ansatz(args.qubits, args.layers)
    rng = np.random.default_rng(20261006)
    bindings = {p: rng.uniform(0, np.pi, size=args.rows) for p in parameters}
    options = {
        "shots": args.shots,
        "simulation_config": {"seed": 5280},
        "result_config": {"counts": True, "final_state": False},
    }
    rows, reference = [], None
    for count in range(1, len(args.devices) + 1):
        devices = tuple(args.devices[:count])
        backend = Simulator("statevector", runtime="cuda", device_id=devices)
        backend.run_sweep(program, bindings, **options).result()  # warm-up
        times, counts = [], None
        for _ in range(args.repeats):
            start = time.perf_counter()
            results = backend.run_sweep(program, bindings, **options).result()
            times.append(time.perf_counter() - start)
            counts = [r.get_counts() for r in results]
        if reference is None:
            reference = counts
        identical = counts == reference
        if not identical:
            raise RuntimeError(f"{count} devices gave different counts")
        median = statistics.median(times) * 1e3
        rows.append(
            {
                "devices": count,
                "median_ms": median,
                "samples_ms": [t * 1e3 for t in times],
                "speedup_over_one_device": (
                    rows[0]["median_ms"] / median if rows else 1.0
                ),
            }
        )
        print(
            f"{count} device(s): {median:.1f} ms x{rows[-1]['speedup_over_one_device']:.2f}",
            flush=True,
        )
    output = {
        "description": "Wall time of one CUDA run_sweep (counts only) on 1..N devices.",
        "measurement_date": date.today().isoformat(),
        "code_revision": revision(),
        "workload": {
            "circuit": f"{args.layers} layers of parameterised RY, staggered CX, fixed RZ; measured",
            "qubits": args.qubits,
            "rows": args.rows,
            "shots_per_row": args.shots,
            "method": "statevector",
        },
        "repeats": args.repeats,
        "devices": len(args.devices),
        "identical_to_single_device": True,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
