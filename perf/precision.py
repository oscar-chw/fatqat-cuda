"""Measure each runtime's round-off against a 60-digit oracle, per method.

For every method (statevector, density matrix, unitary, superoperator) and a
fixed set of seeded random circuits, the same public ``Simulator`` program runs
on every available runtime. Each result is compared with an independent
reference evolved in 60-digit arithmetic from the *stored* complex128
coefficients, so the only error measured is the runtime's own floating-point
round-off. Errors are reported in float64 machine epsilons.

The oracle follows the method from ``tests/simulator/test_cupy_accuracy.py``
(exact embedding of each binary64 coefficient, explicit basis-digit indexing,
no FatQat kernels) and extends it to all four methods and to Kraus channels.

Control: every CPU runtime must sit within 1e-12 of the oracle. If one does
not, the oracle or its index convention is wrong and the run aborts rather
than report a number.

The output carries software versions and the code revision, never machine
details. Run ``perf/scrub_check.py`` on it before committing.

Usage:
    python perf/precision.py --out results/precision.json
    python perf/precision.py --require-gpu --out results/precision.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from itertools import product
import json
import os
from math import ceil, prod
from pathlib import Path
import platform
import subprocess
import sys

import mpmath
import numpy as np

import fatqat as fq
import fatqat.operations as ops
from fatqat.implementation import MatrixImplementationMap
from fatqat.noise import Channel, ChannelImplementationMap
from fatqat.simulator import Simulator

EPS = float(np.finfo(np.float64).eps)
DIGITS = 60
# The CPU control tolerance: the repository's existing absolute contract.
CONTROL_ATOL = 1e-12
_STATE_ONLY = {"counts": False, "final_state": True}
_ACCESSOR = {
    "statevector": "get_statevector",
    "density_matrix": "get_density_matrix",
    "unitary": "get_unitary",
    "superop": "get_superop",
}


@dataclass(frozen=True)
class Case:
    """One family of seeded circuits for one method."""

    method: str
    name: str
    dims: tuple[int, ...]
    depth: int
    channel_every: int = 0  # 0: no channels; k: every k-th step is a channel
    reverse: bool = False  # append the adjoints: cancellation-rich circuits


CASES = (
    Case("statevector", "dense_qubits", (2, 2, 2, 2, 2), 48),
    Case("statevector", "mixed_radix", (3, 2, 3), 48),
    Case("statevector", "cancellation", (2, 2, 2, 2), 24, reverse=True),
    Case("density_matrix", "dense_qubits", (2, 2, 2), 24),
    Case("density_matrix", "kraus_qubits", (2, 2, 2), 24, channel_every=3),
    Case("density_matrix", "kraus_mixed_radix", (2, 3, 2), 18, channel_every=3),
    Case("unitary", "dense_qubits", (2, 2, 2), 32),
    Case("unitary", "mixed_radix", (3, 2), 32),
    Case("unitary", "cancellation", (2, 2, 2), 16, reverse=True),
    Case("superop", "kraus_qubits", (2, 2), 16, channel_every=2),
    Case("superop", "kraus_mixed_radix", (2, 3), 12, channel_every=2),
)


# --- circuit generation -----------------------------------------------------


def _frozen(matrix: np.ndarray) -> np.ndarray:
    # The stored array is the numerical problem for every runtime and for the
    # oracle; it is never recomputed or renormalised afterwards.
    matrix = np.array(matrix, dtype=np.complex128)
    matrix.flags.writeable = False
    return matrix


def _haar_like(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    """An isometry from QR of a complex Gaussian matrix (columns orthonormal)."""
    gaussian = rng.normal(size=(rows, cols)) + 1j * rng.normal(size=(rows, cols))
    q, _ = np.linalg.qr(gaussian)
    return q


def build_steps(case: Case, seed: int) -> list[tuple[str, tuple, tuple[int, ...]]]:
    """Return ``(kind, matrices, targets)`` steps in public subsystem order."""
    rng = np.random.default_rng(seed)
    n = len(case.dims)
    steps = []
    for index in range(case.depth):
        width = 1 + index % min(3, n)
        is_channel = case.channel_every and index % case.channel_every == 0
        if is_channel:
            width = min(width, 2)
        targets = tuple(int(q) for q in rng.choice(n, width, replace=False))
        size = prod(case.dims[q] for q in targets)
        if is_channel:
            count = 2 + index % 2
            stacked = _haar_like(rng, count * size, size)
            kraus = tuple(
                _frozen(stacked[k * size : (k + 1) * size]) for k in range(count)
            )
            steps.append(("channel", kraus, targets))
        else:
            steps.append(("gate", (_frozen(_haar_like(rng, size, size)),), targets))
    if case.reverse:
        for kind, matrices, targets in reversed(steps.copy()):
            # Adjoints of stored rounded matrices: cancellation-rich, but not
            # an exact identity, so the oracle is still required.
            steps.append((kind, (_frozen(matrices[0].conj().T),), targets))
    return steps


def initial_state(case: Case, seed: int) -> np.ndarray | None:
    size = prod(case.dims)
    rng = np.random.default_rng(seed + 1_000_003)
    kets = []
    for _ in range(2):
        ket = rng.normal(size=size) + 1j * rng.normal(size=size)
        kets.append(ket / np.linalg.norm(ket))
    if case.method == "statevector":
        return np.asarray(kets[0], dtype=np.complex128)
    if case.method == "density_matrix":
        rho = 0.6 * np.outer(kets[0], kets[0].conj()) + 0.4 * np.outer(
            kets[1], kets[1].conj()
        )
        return np.asarray(rho, dtype=np.complex128)
    return None


# --- public program ----------------------------------------------------------


def build_program(case: Case, steps):
    """One public Program whose every step is its own operation class."""
    registers = [fq.QuantumRegister(1, dim=dim) for dim in case.dims]
    program = fq.Program(registers)
    implementations = MatrixImplementationMap()
    channels = ChannelImplementationMap()
    noise = fq.NoiseModel()
    for index, (kind, matrices, targets) in enumerate(steps):
        width = len(targets)
        gate = type(
            f"PrecisionStep{index}", (ops.Operation,), {"num_subsystems": width}
        )
        gate.name = f"PrecisionStep{index}"
        if kind == "gate":
            implementations.add(gate, matrices[0])
        else:
            # A channel rides on an identity carrier; identity entries are exact.
            size = len(matrices[0])
            implementations.add(gate, np.eye(size, dtype=np.complex128))
            descriptor = type(
                f"PrecisionChannel{index}", (Channel,), {"num_subsystems": width}
            )
            channels.add(descriptor, lambda _channel, *, targets, _k=matrices: _k)
            noise.add(descriptor(), operation=gate)
        program.add(gate(), tuple(registers[q][0] for q in targets))
    options = {"implementation_map": implementations}
    if any(kind == "channel" for kind, _, _ in steps):
        options.update(noise=noise, channel_implementation_map=channels)
    return program, options


def run_arm(
    case: Case,
    steps,
    initial,
    runtime: str,
    simplify: bool,
    gpu_products: str = "compensated",
) -> np.ndarray:
    program, options = build_program(case, steps)
    backend = Simulator(case.method, runtime=runtime, **options)
    # Explicit: since r12 the default ("auto") may rewrite unit gates.
    config = {"simplify": simplify}
    if runtime == "cuda" and case.method in ("statevector", "unitary"):
        config["gpu_products"] = gpu_products  # the methods it applies to
    result = backend.run(
        program,
        shots=0,
        initial_state=initial,
        result_config=_STATE_ONLY,
        simulation_config=config,
    ).result()
    state = np.asarray(getattr(result, _ACCESSOR[case.method])())
    if state.dtype != np.complex128 or not np.all(np.isfinite(state)):
        raise RuntimeError(f"{runtime} returned a non-finite or non-complex128 state")
    return state


# --- 60-digit oracle ------------------------------------------------------------


def _exact(value, mp):
    """Embed a binary64 complex value by its exact rational components."""
    rn, rd = float(value.real).as_integer_ratio()
    im, idn = float(value.imag).as_integer_ratio()
    return mp.mpc(mp.mpf(rn) / rd, mp.mpf(im) / idn)


class _LocalMap:
    """Basis bookkeeping for a local operator, from public digits alone.

    Public order: subsystem 0 is the most significant digit. The local index
    of the targets uses the targets' given order, first target most
    significant, which is how a matrix rule's rows are labelled.
    """

    def __init__(self, dims, targets):
        basis = np.array(list(product(*(range(d) for d in dims))), dtype=np.int64)
        strides = np.array(
            [prod(dims[q + 1 :]) for q in range(len(dims))], dtype=np.int64
        )
        target_dims = [dims[t] for t in targets]
        local_strides = np.array(
            [prod(target_dims[i + 1 :]) for i in range(len(targets))], dtype=np.int64
        )
        target_digits = basis[:, list(targets)]
        self.row = (target_digits * local_strides).sum(axis=1)
        base = np.arange(len(basis)) - (target_digits * strides[list(targets)]).sum(
            axis=1
        )
        local_basis = np.array(
            list(product(*(range(d) for d in target_dims))), dtype=np.int64
        )
        offsets = (local_basis * strides[list(targets)]).sum(axis=1)
        self.source = base[:, None] + offsets[None, :]


def _apply_vector(vector, matrix, local):
    """``out[o] = sum_c M[row(o), c] * v[source(o, c)]`` in mp arithmetic."""
    return [
        mpmath.mp.fdot(matrix[local.row[o]], [vector[s] for s in local.source[o]])
        for o in range(len(vector))
    ]


def _apply_sandwich(rho, matrix, conj_matrix, local):
    """``M rho M^dagger`` on a list-of-rows matrix."""
    size = len(rho)
    columns = [
        _apply_vector([rho[r][c] for r in range(size)], matrix, local)
        for c in range(size)
    ]
    left = [[columns[c][r] for c in range(size)] for r in range(size)]
    return [_apply_vector(row, conj_matrix, local) for row in left]


def _apply_channel(rho, exact_ops, local):
    size = len(rho)
    total = [[mpmath.mp.mpc(0)] * size for _ in range(size)]
    for matrix, conj_matrix in exact_ops:
        term = _apply_sandwich(rho, matrix, conj_matrix, local)
        total = [[total[r][c] + term[r][c] for c in range(size)] for r in range(size)]
    return total


def oracle(case: Case, steps, initial) -> np.ndarray:
    """The exact evolution of the stored coefficients, rounded only at the end."""
    mp = mpmath.mp
    size = prod(case.dims)
    prepared = []
    for _, matrices, targets in steps:
        exact_ops = []
        for matrix in matrices:
            exact = [[_exact(v, mp) for v in row] for row in matrix]
            exact_ops.append((exact, [[mp.conj(v) for v in row] for row in exact]))
        prepared.append((exact_ops, _LocalMap(case.dims, targets)))

    def evolve_density(rho):
        for exact_ops, local in prepared:
            rho = _apply_channel(rho, exact_ops, local)
        return rho

    if case.method == "statevector":
        state = [_exact(v, mp) for v in initial]
        for exact_ops, local in prepared:
            state = _apply_vector(state, exact_ops[0][0], local)
        return state
    if case.method == "density_matrix":
        return evolve_density([[_exact(v, mp) for v in row] for row in initial])
    if case.method == "unitary":
        columns = []
        for column in range(size):
            state = [mp.mpc(1 if r == column else 0) for r in range(size)]
            for exact_ops, local in prepared:
                state = _apply_vector(state, exact_ops[0][0], local)
            columns.append(state)
        return [[columns[c][r] for c in range(size)] for r in range(size)]
    # Superoperator, public column-stacking: S vec_F(rho) = vec_F(channel(rho)),
    # vec_F index of entry (i, j) is i + j * size.
    superop = [[mp.mpc(0)] * (size * size) for _ in range(size * size)]
    for i, j in product(range(size), repeat=2):
        basis = [
            [mp.mpc(1 if (r, c) == (i, j) else 0) for c in range(size)]
            for r in range(size)
        ]
        image = evolve_density(basis)
        for r, c in product(range(size), repeat=2):
            superop[r + c * size][i + j * size] = image[r][c]
    return superop


def error_eps(state: np.ndarray, reference) -> dict:
    mp = mpmath.mp
    flat_state = np.asarray(state).reshape(-1)
    flat_ref = (
        [v for row in reference for v in row]
        if isinstance(reference[0], list)
        else reference
    )
    diffs = [abs(_exact(v, mp) - r) for v, r in zip(flat_state, flat_ref, strict=True)]
    return {
        "linf_eps": float(max(diffs)) / EPS,
        "l2_eps": float(mp.sqrt(mp.fsum(d * d for d in diffs))) / EPS,
    }


# --- driver ---------------------------------------------------------------------


def cuda_available() -> bool:
    try:
        import cupy  # pylint: disable=import-outside-toplevel

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pylint: disable=broad-except
        # Any import or driver failure means no usable device for this run.
        return False


def arms_for(with_gpu: bool) -> list[str]:
    # No Numba fusion arm: operator fusion engages only from 2**18 basis
    # entries, beyond the oracle's reach, and on these cases a fused run was
    # measured bit-identical to the unfused one, so the arm could never differ.
    return ["numpy", "numba"] + (["cuda"] if with_gpu else [])


def summarise(rows: list[dict], with_gpu: bool) -> None:
    """Add a per-case verdict comparing the GPU with the best CPU runtime."""
    for row in rows:
        if not with_gpu:
            row["verdict"] = None
            continue
        excess = []
        for seed in row["seeds"]:
            errors = seed["errors"]
            best_cpu = min(v["linf_eps"] for k, v in errors.items() if k != "cuda")
            excess.append(errors["cuda"]["linf_eps"] - best_cpu)
        worst = max(excess)
        row["gpu_minus_best_cpu_linf_eps"] = {
            "max": worst,
            "median": float(np.median(excess)),
            "seeds_gpu_not_worse": sum(e <= 0 for e in excess),
            "seeds": len(excess),
        }
        row["verdict"] = (
            "gpu_not_worse" if worst <= 0 else f"gpu_within_{ceil(worst)}_eps"
        )


def paired_comparisons(rows: list[dict]) -> dict:
    """Mean per-circuit error difference between every pair of runtimes.

    Each seeded circuit is one paired observation, so the standard error of
    the mean difference says whether a gap is real or rounding noise. The
    row verdicts above are deliberately stricter (worst seed, every CPU).
    """
    labels = list(rows[0]["seeds"][0]["errors"])
    output = {}
    for i, first in enumerate(labels):
        for second in labels[i + 1 :]:
            diffs = [
                seed["errors"][first]["linf_eps"] - seed["errors"][second]["linf_eps"]
                for row in rows
                for seed in row["seeds"]
            ]
            spread = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            output[f"{first}_minus_{second}"] = {
                "mean_eps": float(np.mean(diffs)),
                "standard_error_eps": spread / np.sqrt(len(diffs)),
                "circuits_first_not_worse": sum(d <= 0 for d in diffs),
                "circuits": len(diffs),
            }
    return output


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
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--methods", nargs="+", choices=sorted(_ACCESSOR), default=sorted(_ACCESSOR)
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="exit 2 instead of measuring CPU only when no CUDA device is usable",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="run every arm with simulation_config simplify=True; the oracle "
        "still evolves the circuit as written, which these circuits of "
        "Haar-random gates leave unchanged (perf/simplify_check.py measures "
        "simplification itself)",
    )
    parser.add_argument(
        "--gpu-products",
        choices=("compensated", "plain"),
        default="compensated",
        help="the CUDA arm's simulation_config gpu_products",
    )
    args = parser.parse_args(argv)

    with_gpu = cuda_available()
    if args.require_gpu and not with_gpu:
        print(
            "precision: --require-gpu given but no usable CUDA device", file=sys.stderr
        )
        return 2

    rows = []
    with mpmath.mp.workdps(DIGITS):
        for case in CASES:
            if case.method not in args.methods:
                continue
            row = {
                "method": case.method,
                "case": case.name,
                "dims": list(case.dims),
                "steps": None,
                "seeds": [],
            }
            for offset in range(args.seeds):
                seed = 5280 + 1000 * CASES.index(case) + offset
                steps = build_steps(case, seed)
                row["steps"] = len(steps)
                initial = initial_state(case, seed)
                reference = oracle(case, steps, initial)
                errors = {}
                for label in arms_for(with_gpu):
                    runtime = label
                    state = run_arm(
                        case, steps, initial, runtime, args.simplify, args.gpu_products
                    )
                    errors[label] = error_eps(state, reference)
                    if (
                        runtime != "cuda"
                        and errors[label]["linf_eps"] * EPS > CONTROL_ATOL
                    ):
                        raise RuntimeError(
                            f"control failed: {label} on {case.method}/{case.name} seed {seed} "
                            f"is {errors[label]['linf_eps'] * EPS:.3e} from the oracle; "
                            "the oracle or its index convention is wrong"
                        )
                row["seeds"].append({"seed": seed, "errors": errors})
            row["cpu_eps"] = {
                label: max(s["errors"][label]["linf_eps"] for s in row["seeds"])
                for label in row["seeds"][0]["errors"]
                if label != "cuda"
            }
            row["gpu_eps"] = (
                max(s["errors"]["cuda"]["linf_eps"] for s in row["seeds"])
                if with_gpu
                else None
            )
            rows.append(row)
            print(
                f"{case.method:15s} {case.name:18s} cpu_max_eps="
                + " ".join(f"{k}={v:.2f}" for k, v in row["cpu_eps"].items())
                + (f" cuda={row['gpu_eps']:.2f}" if with_gpu else ""),
                flush=True,
            )
    summarise(rows, with_gpu)

    import numba  # pylint: disable=import-outside-toplevel

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "mpmath": mpmath.__version__,
    }
    if with_gpu:
        import cupy  # pylint: disable=import-outside-toplevel

        versions["cupy"] = cupy.__version__
    output = {
        "description": "Round-off of each runtime against a 60-digit oracle of the stored complex128 coefficients, in float64 epsilons (2**-52).",
        "measurement_date": date.today().isoformat(),
        "code_revision": revision(),
        "software": versions,
        "gpu_measured": with_gpu,
        "simplify": args.simplify,
        "gpu_products": args.gpu_products,
        "oracle_decimal_digits": DIGITS,
        "error_unit": "float64 machine epsilon",
        "cpu_control_tolerance": CONTROL_ATOL,
        "interpretation": "Errors of a few epsilons are at the rounding floor; differences of one or two epsilons between runtimes are not an accuracy ranking.",
        "verdict_definition": "gpu_not_worse: on every seed the GPU error is at most every CPU runtime's; gpu_within_k_eps: the worst seed exceeds some CPU runtime by at most k epsilons",
        "paired_comparisons": paired_comparisons(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
