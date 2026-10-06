"""Check that ``simplify=True`` is never less accurate and is worth running.

Accuracy: seeded Clifford+T circuits with and without the redundancy that
simplification removes run on every available runtime, with and without
``simplify``, from the all-zero state. Each result is compared with the
*ideal* circuit evolved in 60-digit arithmetic from the exact gate definitions
(``H`` is ``1/sqrt(2)``, not its rounded double), because that is what the
circuit means; the rounded stored coefficients are what both arms share.
Every circuit is one paired observation, so the mean of ``error(simplify) -
error(plain)`` with its standard error says whether simplification costs
accuracy. Control: every plain CPU arm must sit within 1e-12 of the ideal
circuit, or the reference's conventions are wrong and the run aborts.

Speed: four larger circuits, timed warm with both arms alternating in one
process, so drift and contention hit both alike, as exact expectation values
(the state stays where it was computed, as in an Estimator call):

- a ripple-carry adder whose Toffolis are written in Clifford+T, loaded with
  ``X`` gates from the all-zero state: what compiled arithmetic looks like;
- a random Clifford+T circuit with lecture-style redundancy (``H X H``, ``T T``,
  inverse pairs, commuting ``CX``);
- MaxCut QAOA, whose ``ZZ`` terms are written ``CX RZ CX``;
- a quantum Fourier transform after a rotation layer, where nothing simplifies:
  the control. Its wall-time ratio is reported, but the gate is the pass's own
  cost (the fastest of the repeats) as a fraction of the plain run, which
  background load cannot make look better or worse than it is.

Exit status is nonzero when the control fails, when simplification is
measurably less accurate on any runtime, when the three circuits with redundancy are
not faster on every CPU runtime, or when the pass costs more than a few per
cent of a plain run of the control circuit.

Usage:
    python perf/simplify_check.py --out results/simplify-check.json
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

import mpmath
import numpy as np

import fatqat as fq
import fatqat.operations as ops
from fatqat.simulator import Simulator

EPS = float(np.finfo(np.float64).eps)
DIGITS = 60
CONTROL_ATOL = 1e-12
# A speed-up must clear this on the redundant circuits; on the control
# circuit the pass may cost at most this fraction of a plain run.
MIN_SPEEDUP = 1.2
MAX_PASS_FRACTION = 0.05
_STATE = {"counts": False, "final_state": True}
_FIXED = ("H", "X", "Y", "Z", "S", "Sdg", "T", "Tdg", "SX")


# --- circuits: (name, angle or None, public qubits) --------------------------


def toffoli(a: int, b: int, c: int) -> list:
    """The standard 7-T Clifford+T Toffoli, controls ``a``, ``b``, target ``c``."""
    g = []
    g += [("H", None, (c,)), ("CX", None, (b, c)), ("Tdg", None, (c,))]
    g += [("CX", None, (a, c)), ("T", None, (c,)), ("CX", None, (b, c))]
    g += [("Tdg", None, (c,)), ("CX", None, (a, c)), ("T", None, (b,))]
    g += [("T", None, (c,)), ("H", None, (c,)), ("CX", None, (a, b))]
    g += [("T", None, (a,)), ("Tdg", None, (b,)), ("CX", None, (a, b))]
    return g


def adder(bits: int, a_value: int, b_value: int) -> tuple[int, list]:
    """Cuccaro ripple-carry adder on ``2 * bits + 2`` qubits, inputs loaded by X."""
    n = 2 * bits + 2
    carry, a, b, z = (
        0,
        [2 + 2 * i for i in range(bits)],
        [1 + 2 * i for i in range(bits)],
        n - 1,
    )
    gates = [("X", None, (a[i],)) for i in range(bits) if a_value >> i & 1]
    gates += [("X", None, (b[i],)) for i in range(bits) if b_value >> i & 1]

    def maj(x, y, w):
        return [("CX", None, (w, y)), ("CX", None, (w, x))] + toffoli(x, y, w)

    def uma(x, y, w):
        return toffoli(x, y, w) + [("CX", None, (w, x)), ("CX", None, (x, y))]

    chain = [carry] + [q for i in range(bits) for q in (b[i], a[i])]
    for i in range(bits):
        gates += maj(chain[2 * i], chain[2 * i + 1], chain[2 * i + 2])
    gates.append(("CX", None, (a[-1], z)))
    for i in reversed(range(bits)):
        gates += uma(chain[2 * i], chain[2 * i + 1], chain[2 * i + 2])
    return n, gates


def redundant_clifford_t(rng: np.random.Generator, n: int, depth: int) -> list:
    """Random Clifford+T with the identities a lecture on circuit synthesis lists."""
    gates: list = []
    while len(gates) < depth:
        q = int(rng.integers(n))
        r = int(rng.integers(n - 1))
        r += r >= q
        pick = rng.random()
        if pick < 0.12:
            gates += [("H", None, (q,)), ("X", None, (q,)), ("H", None, (q,))]
        elif pick < 0.22:
            gates += [("T", None, (q,)), ("T", None, (q,))]
        elif pick < 0.30:
            gates += [("CX", None, (q, r)), ("CX", None, (q, r))]
        elif pick < 0.38:
            gates += [("H", None, (q,)), ("H", None, (r,)), ("CX", None, (q, r))]
            gates += [("H", None, (q,)), ("H", None, (r,))]
        elif pick < 0.70:
            gates.append((str(rng.choice(_FIXED)), None, (q,)))
        elif pick < 0.85:
            gates.append((str(rng.choice(["CX", "CZ"])), None, (q, r)))
        else:
            gates.append(("RZ", float(rng.normal()), (q,)))
    return gates


def plain_clifford_t(rng: np.random.Generator, n: int, depth: int) -> list:
    """Random Clifford+T with no planted redundancy."""
    gates = []
    for _ in range(depth):
        q = int(rng.integers(n))
        if rng.random() < 0.3:
            r = int(rng.integers(n - 1))
            gates.append(("CX", None, (q, r + (r >= q))))
        else:
            gates.append((str(rng.choice(_FIXED)), None, (q,)))
    return gates


def qaoa(rng: np.random.Generator, n: int, layers: int = 2) -> list:
    """MaxCut QAOA on a ring plus random chords: ZZ terms as CX RZ CX."""
    edges = [(q, (q + 1) % n) for q in range(n)]
    edges += [
        tuple(int(v) for v in rng.choice(n, 2, replace=False)) for _ in range(n // 2)
    ]
    gates = [("H", None, (q,)) for q in range(n)]
    for _ in range(layers):
        gamma, beta = float(rng.uniform(0, np.pi)), float(rng.uniform(0, np.pi))
        for a, b in edges:
            gates += [("CX", None, (a, b)), ("RZ", gamma, (b,)), ("CX", None, (a, b))]
        gates += [("RX", beta, (q,)) for q in range(n)]
    return gates


def rotated_qft(rng: np.random.Generator, n: int) -> list:
    gates = [("RY", float(rng.uniform(0, np.pi)), (q,)) for q in range(n)]
    for q in range(n):
        gates.append(("H", None, (q,)))
        for k, r in enumerate(range(q + 1, n), start=2):
            gates.append(("CPhase", 2 * np.pi / 2**k, (r, q)))
    return gates


def build_program(n: int, gates: list):
    program = fq.Program(n)
    for name, angle, qubits in gates:
        operation = getattr(ops, name)
        program.add(operation if angle is None else operation(angle), qubits)
    return program


# --- the ideal circuit, 60 digits ---------------------------------------------


def ideal_matrix(name: str, angle):
    mp = mpmath.mp
    r = 1 / mp.sqrt(2)
    w = mp.expjpi(mp.mpf(1) / 4)
    fixed = {
        "H": [[r, r], [r, -r]],
        "X": [[0, 1], [1, 0]],
        "Y": [[0, -1j], [1j, 0]],
        "Z": [[1, 0], [0, -1]],
        "S": [[1, 0], [0, 1j]],
        "Sdg": [[1, 0], [0, -1j]],
        "T": [[1, 0], [0, w]],
        "Tdg": [[1, 0], [0, mp.conj(w)]],
        "SX": [[(1 + 1j) / 2, (1 - 1j) / 2], [(1 - 1j) / 2, (1 + 1j) / 2]],
        "CX": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        "CZ": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
    }
    if name in fixed:
        return [[mp.mpc(v) for v in row] for row in fixed[name]]
    # The angle is the double the user wrote; the ideal gate rotates by it exactly.
    theta = mp.mpf(angle)
    if name == "RZ":
        return [[mp.expj(-theta / 2), 0], [0, mp.expj(theta / 2)]]
    if name == "RY":
        c, s = mp.cos(theta / 2), mp.sin(theta / 2)
        return [[c, -s], [s, c]]
    if name == "RX":
        c, s = mp.cos(theta / 2), mp.sin(theta / 2)
        return [[c, -1j * s], [-1j * s, c]]
    if name == "CPhase":
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, mp.expj(theta)]]
    raise ValueError(f"no ideal definition for {name}")


def ideal_state(n: int, gates: list) -> list:
    """Evolve |0...0> exactly; public qubit 0 is the most significant bit."""
    state = [mpmath.mp.mpc(0)] * 2**n
    state[0] = mpmath.mp.mpc(1)
    for name, angle, qubits in gates:
        matrix = ideal_matrix(name, angle)
        shifts = [n - 1 - q for q in qubits]  # first target = most significant
        mask = sum(1 << s for s in shifts)
        out = list(state)
        for base in range(2**n):
            if base & mask:
                continue
            indices = [
                base
                | sum(
                    ((local >> (len(shifts) - 1 - i)) & 1) << s
                    for i, s in enumerate(shifts)
                )
                for local in range(2 ** len(shifts))
            ]
            amplitudes = [state[i] for i in indices]
            for row, index in enumerate(indices):
                out[index] = mpmath.mp.fdot(matrix[row], amplitudes)
        state = out
    return state


def error_eps(state: np.ndarray, reference: list) -> float:
    mp = mpmath.mp
    worst = mp.mpf(0)
    for value, exact in zip(np.asarray(state).reshape(-1), reference, strict=True):
        worst = max(worst, abs(mp.mpc(value.real, value.imag) - exact))
    return float(worst) / EPS


# --- arms -------------------------------------------------------------------------


def cuda_available() -> bool:
    try:
        import cupy  # pylint: disable=import-outside-toplevel

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pylint: disable=broad-except
        # Any import or driver failure means no usable device for this run.
        return False


def run(program, method: str, runtime: str, simplify: bool) -> np.ndarray:
    result = (
        Simulator(method, runtime=runtime)
        .run(
            program,
            shots=0,
            result_config=_STATE,
            simulation_config={"simplify": simplify},
        )
        .result()
    )
    return np.asarray(getattr(result, f"get_{method}")())


def expectation(program, n: int, runtime: str, simplify: bool) -> float:
    """An exact expectation value: the state stays where it was computed.

    Timed calls use this rather than exporting the state, whose copy to the
    host (a whole GiB at 26 qubits) would add the same constant to both arms.
    """
    observable = fq.Observable([("Z" + "I" * (n - 1), 1.0)])
    estimator = fq.Estimator(Simulator("statevector", runtime=runtime))
    job = estimator.run(
        program, observable, shots=0, simulation_config={"simplify": simplify}
    )
    return float(job.result().get_expectation())


def accuracy(runtimes: list[str], seeds: int) -> dict:
    families = {
        "redundant_clifford_t": lambda rng: (6, redundant_clifford_t(rng, 6, 120)),
        "plain_clifford_t": lambda rng: (6, plain_clifford_t(rng, 6, 120)),
        "adder": lambda rng: adder(2, int(rng.integers(4)), int(rng.integers(4))),
        "qaoa": lambda rng: (6, qaoa(rng, 6)),
    }
    rows = []
    for family, make in families.items():
        for seed in range(seeds):
            n, gates = make(np.random.default_rng(7000 + seed))
            reference = ideal_state(n, gates)
            program = build_program(n, gates)
            errors = {}
            for runtime in runtimes:
                for simplify in (False, True):
                    state = run(program, "statevector", runtime, simplify)
                    errors[f"{runtime}_{'simplify' if simplify else 'plain'}"] = (
                        error_eps(state, reference)
                    )
                if (
                    runtime != "cuda"
                    and errors[f"{runtime}_plain"] * EPS > CONTROL_ATOL
                ):
                    raise RuntimeError(
                        f"control failed: {runtime} on {family} seed {seed}; "
                        "the ideal reference's conventions are wrong"
                    )
            rows.append({"family": family, "seed": seed, "qubits": n, "errors": errors})
    paired = {}
    for runtime in runtimes:
        paired[runtime] = paired_stats(rows, runtime)
        paired[runtime]["by_family"] = {
            family: paired_stats([r for r in rows if r["family"] == family], runtime)
            for family in families
        }
    return {"rows": rows, "paired": paired}


def paired_stats(rows: list[dict], runtime: str) -> dict:
    """Mean of error(simplify) - error(plain) over circuits, with its standard error."""
    diffs = [
        row["errors"][f"{runtime}_simplify"] - row["errors"][f"{runtime}_plain"]
        for row in rows
    ]
    return {
        "mean_simplify_minus_plain_eps": statistics.fmean(diffs),
        "standard_error_eps": statistics.stdev(diffs) / len(diffs) ** 0.5,
        "circuits_simplify_not_worse": sum(d <= 0 for d in diffs),
        "circuits": len(diffs),
        "mean_plain_eps": statistics.fmean(
            r["errors"][f"{runtime}_plain"] for r in rows
        ),
        "mean_simplify_eps": statistics.fmean(
            r["errors"][f"{runtime}_simplify"] for r in rows
        ),
    }


def timing(runtimes: list[str], qubits: int, repeats: int) -> list[dict]:
    rng = np.random.default_rng(5280)
    bits = (qubits - 2) // 2
    n_adder, adder_gates = adder(bits, (1 << bits) - 3, 5)
    workloads = {
        "adder": (n_adder, adder_gates),
        "redundant_clifford_t": (qubits, redundant_clifford_t(rng, qubits, 1200)),
        "qaoa": (qubits, qaoa(rng, qubits)),
        "rotated_qft_control": (qubits, rotated_qft(rng, qubits)),
    }
    out = []
    for name, (n, gates) in workloads.items():
        program = build_program(n, gates)
        backend = Simulator("statevector", runtime="numpy")
        plan, _ = backend._lower_program(program)  # pylint: disable=protected-access
        from fatqat._backends.simplify import (
            simplify_plan,
        )  # pylint: disable=import-outside-toplevel

        steps_after = len(simplify_plan(plan, (2,) * n, zero_start=True))
        pass_seconds = []
        for _ in range(max(3, repeats)):
            start = time.perf_counter()
            simplify_plan(plan, (2,) * n, zero_start=True)
            pass_seconds.append(time.perf_counter() - start)
        for runtime in runtimes:
            times = {False: [], True: []}
            expectation(program, n, runtime, False)  # compile and warm
            expectation(program, n, runtime, True)
            for _ in range(repeats):
                for simplify in (False, True):
                    start = time.perf_counter()
                    expectation(program, n, runtime, simplify)
                    times[simplify].append(time.perf_counter() - start)
            plain, simple = statistics.median(times[False]), statistics.median(
                times[True]
            )
            out.append(
                {
                    "workload": name,
                    "qubits": n,
                    "runtime": runtime,
                    "steps_plain": len(plan),
                    "steps_simplify": steps_after,
                    "median_plain_s": plain,
                    "median_simplify_s": simple,
                    "speedup": plain / simple,
                    "pass_s": min(pass_seconds),
                    "pass_fraction_of_plain": min(pass_seconds) / plain,
                }
            )
            print(
                f"{name:22s} {runtime:6s} {len(plan):5d}->{steps_after:5d} steps  "
                f"{plain / simple:.2f}x  pass {min(pass_seconds) / plain:.1%} of a plain run",
                flush=True,
            )
    return out


def verdict(paired: dict, rows: list[dict]) -> list[str]:
    """Every reason the check fails; empty when it passes."""
    failures = []
    for runtime, stats in paired.items():
        # Pooled and per family, so a gain in one family cannot hide a loss
        # in another.
        groups = {"all circuits": stats, **stats.get("by_family", {})}
        for name, group in groups.items():
            if group["mean_simplify_minus_plain_eps"] > 2 * group["standard_error_eps"]:
                failures.append(
                    f"{runtime}: simplify is measurably less accurate on {name}"
                )
    for row in rows:
        # Wall time is gated on the CPU only. A GPU at this size applies a
        # gate in microseconds, so there the planning step is reported, not
        # gated: it pays only on larger states.
        if row["runtime"] == "cuda":
            continue
        if row["workload"] == "rotated_qft_control":
            if row["pass_fraction_of_plain"] > MAX_PASS_FRACTION:
                failures.append(
                    f"{row['runtime']}: the pass costs too much on the control"
                )
        elif row["speedup"] < MIN_SPEEDUP:
            failures.append(f"{row['runtime']}: {row['workload']} is not faster")
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
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--runtimes",
        nargs="+",
        choices=("numpy", "numba", "cuda"),
        help="runtimes to measure (default: numpy, numba, and cuda when a device "
        "is usable); a GPU pays for planning only on large states, so it can be "
        "timed alone at a larger --qubits",
    )
    args = parser.parse_args(argv)

    runtimes = args.runtimes or (
        ["numpy", "numba"] + (["cuda"] if cuda_available() else [])
    )
    with mpmath.mp.workdps(DIGITS):
        measured = accuracy(runtimes, args.seeds)
    for runtime, stats in measured["paired"].items():
        print(
            f"accuracy {runtime:6s} plain {stats['mean_plain_eps']:.3f} eps, simplify "
            f"{stats['mean_simplify_eps']:.3f} eps, diff "
            f"{stats['mean_simplify_minus_plain_eps']:+.3f} +- {stats['standard_error_eps']:.3f}",
            flush=True,
        )
    speed = timing(runtimes, args.qubits, args.repeats)
    failures = verdict(measured["paired"], speed)

    import numba  # pylint: disable=import-outside-toplevel

    output = {
        "description": "simplify=True against the ideal circuit (60-digit exact gate definitions) and its wall-time effect, both arms in one process.",
        "measurement_date": date.today().isoformat(),
        "code_revision": revision(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "mpmath": mpmath.__version__,
        },
        "error_unit": "float64 machine epsilon, max-abs over amplitudes",
        "accuracy_paired": measured["paired"],
        "accuracy_rows": measured["rows"],
        "timing": speed,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
