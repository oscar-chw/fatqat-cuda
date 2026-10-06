"""Time FatQat's public API across qubit counts, CPU (compiled Numba) vs one GPU.

Workloads are the ones behind ``results/benchmarks.json``: two layers of RY and
RZ on every qubit plus staggered nearest-neighbour CX, optionally with
amplitude and phase damping after every RY and RZ, and a three-term Pauli
observable. Each (workload, size, runtime) runs in a fresh subprocess, so the
thread limit is set before Numba loads and GPU memory is released between
sizes. Every timed call is one warm public FatQat call including GPU
synchronisation and the requested host output; the first call (JIT
compilation, plan caching) is reported separately and never in the median.

Correctness: every runtime's output is reduced to a fingerprint (its inner
product with a fixed seeded random vector, or the expectation value) and the
GPU fingerprint must match the CPU one, else the run aborts.

The CPU baseline is compiled Numba at the declared ``--threads`` count. That
number is a run configuration and is recorded; nothing about the machine is.
Run ``perf/scrub_check.py`` on the output before committing it.

Usage:
    python perf/scaling.py --threads 32 --out results/scaling.json
    python perf/scaling.py --threads 32 --require-gpu --out results/scaling.json
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

# Workload name -> (method, observable?, noisy?). Sizes are qubit counts; the
# matrix methods hold 4**n entries, so their sizes are about half the SV ones.
WORKLOADS = {
    "sv_state": ("statevector", False, False),
    "sv_observable": ("statevector", True, False),
    "dm_noise": ("density_matrix", False, True),
    "dm_observable": ("density_matrix", True, True),
    "unitary": ("unitary", False, False),
}
SV_SIZES = (12, 14, 16, 18, 20, 22, 24, 26, 28)
MATRIX_SIZES = (6, 7, 8, 9, 10, 11, 12, 13, 14)
FINGERPRINT_ATOL = 1e-10


def sizes_for(workload: str, max_qubits: int) -> list[int]:
    method = WORKLOADS[workload][0]
    pool = SV_SIZES if method == "statevector" else MATRIX_SIZES
    limit = max_qubits if method == "statevector" else max_qubits // 2 + 1
    return [n for n in pool if n <= limit]


# --- child: one measurement -----------------------------------------------------


def _program(fq, ops, n):
    program = fq.Program(n)
    for layer in range(2):
        for q in range(n):
            program.add(ops.RY(0.31 + q * 0.013 + layer * 0.071), q)
            program.add(ops.RZ(-0.27 + q * 0.011), q)
        for q in range(layer % 2, n - 1, 2):
            program.add(ops.CX, (q, q + 1))
    return program


def _quartiles(samples: list[float]) -> list[float]:
    """Q1, median, Q3 within the measured range.

    The default "exclusive" method extrapolates beyond the data for small
    samples (two warm calls can give a negative Q1), so "inclusive" is used.
    """
    if len(samples) < 2:
        return [samples[0]] * 3
    return statistics.quantiles(samples, n=4, method="inclusive")


def _peak_rss_mib() -> float | None:
    """Peak resident memory of this process; None where the OS has no rusage."""
    try:
        import resource  # pylint: disable=import-outside-toplevel
    except ImportError:  # Windows has no resource module
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _fingerprint(np, value) -> list[float]:
    """Inner product with a fixed random vector: any substantive error moves it."""
    flat = np.asarray(value).reshape(-1)
    rng = np.random.default_rng(20260915)
    probe = rng.standard_normal(flat.size) + 1j * rng.standard_normal(flat.size)
    result = np.vdot(probe, flat) / np.sqrt(flat.size)
    return [float(result.real), float(result.imag)]


def child(args) -> None:
    # Imports happen here, after the parent set the thread environment.
    import numpy as np  # pylint: disable=import-outside-toplevel

    import fatqat as fq  # pylint: disable=import-outside-toplevel
    import fatqat.operations as ops  # pylint: disable=import-outside-toplevel
    from fatqat.simulator import Simulator  # pylint: disable=import-outside-toplevel

    method, observable_case, noisy = WORKLOADS[args.workload]
    n = args.qubits
    circuit = _program(fq, ops, n)
    noise = None
    if noisy:
        noise = fq.NoiseModel()
        noise.add(fq.noise.AmplitudeDamping(p=0.07), operation=ops.RY)
        noise.add(fq.noise.PhaseDamping(p=0.02), operation=ops.RZ)
    observable = fq.Observable(
        [
            ("Z" + "I" * (n - 2) + "Z", 0.8),
            ("X" * n, 0.3),
            ("Y" + "I" * (n - 2) + "Y", -0.2),
        ]
    )
    backend = Simulator(method, runtime=args.runtime, noise=noise)
    config = {}
    if args.runtime == "numba":
        config = (
            {"kernel_parallelism": "serial"}
            if args.threads == 1
            else {"kernel_parallelism": "threads", "max_workers": args.threads}
        )
    if args.simplify:
        config["simplify"] = True

    def execute():
        if observable_case:
            job = fq.Estimator(backend).run(
                circuit, observable, shots=0, simulation_config=config
            )
            return job.result().get_expectation()
        job = backend.run(
            circuit,
            shots=0,
            simulation_config=config,
            result_config={"counts": False, "final_state": True},
        )
        return getattr(job.result(), "get_" + method)()

    times = []
    value = None
    for _ in range(args.repeats + 1):
        del value  # free the previous output before the next timed call
        start = time.perf_counter()
        value = execute()
        times.append(time.perf_counter() - start)
    # Resources, read before the fingerprint allocates its own probe arrays.
    peak_rss_mib = _peak_rss_mib()
    gpu_pool_mib = None
    if args.runtime == "cuda":
        import cupy  # pylint: disable=import-outside-toplevel

        # The pool keeps freed blocks, so its reserved size is the high-water mark.
        gpu_pool_mib = cupy.get_default_memory_pool().total_bytes() / 2**20
    if observable_case:
        fingerprint = [float(np.real(value)), float(np.imag(value))]
    else:
        fingerprint = _fingerprint(np, value)
    warm = times[1:]
    quartiles = _quartiles(warm)
    print(
        json.dumps(
            {
                "first_call_ms": times[0] * 1e3,
                "warm_ms": [t * 1e3 for t in warm],
                "median_ms": statistics.median(warm) * 1e3,
                "q1_ms": quartiles[0] * 1e3,
                "q3_ms": quartiles[2] * 1e3,
                "fingerprint": fingerprint,
                "peak_rss_mib": peak_rss_mib,
                "gpu_pool_reserved_mib": gpu_pool_mib,
            }
        ),
        flush=True,
    )


# --- parent: orchestration --------------------------------------------------------


def _thread_env(threads: int) -> dict:
    env = dict(os.environ)
    for key in (
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = str(threads)
    return env


def measure(workload, qubits, runtime, threads, repeats, timeout, simplify) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--workload",
        workload,
        "--qubits",
        str(qubits),
        "--runtime",
        runtime,
        "--threads",
        str(threads),
        "--repeats",
        str(repeats),
    ] + (["--simplify"] if simplify else [])
    completed = subprocess.run(
        command,
        env=_thread_env(threads),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{workload} n={qubits} {runtime} failed:\n{completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def cuda_available() -> bool:
    probe = (
        "import cupy, sys; sys.exit(0 if cupy.cuda.runtime.getDeviceCount() > 0 else 1)"
    )
    try:
        return (
            subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                check=False,
                timeout=120,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


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
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), help=argparse.SUPPRESS)
    parser.add_argument("--qubits", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--runtime", choices=["numba", "cuda"], help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--threads",
        type=int,
        required=True,
        help="Numba CPU threads (a run configuration)",
    )
    parser.add_argument("--repeats", type=int, default=5, help="warm calls per median")
    parser.add_argument(
        "--max-qubits", type=int, default=26, help="largest statevector size"
    )
    parser.add_argument(
        "--workloads", nargs="+", choices=sorted(WORKLOADS), default=sorted(WORKLOADS)
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="stop growing a runtime's workload once one warm call exceeds this",
    )
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--runtimes",
        nargs="+",
        choices=["numba", "cuda"],
        default=["numba", "cuda"],
        help="runtimes to time (cuda is skipped when no device is usable)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="earlier scaling JSON whose CPU rows verify (and serve as the CPU "
        "baseline for) GPU-only rows",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="time every call with simulation_config simplify=True",
    )
    parser.add_argument(
        "--revision-label", default=None, help="engine revision label, e.g. r8"
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.child:
        child(args)
        return 0
    if args.out is None:
        parser.error("--out is required")

    with_gpu = cuda_available()
    if args.require_gpu and not with_gpu:
        print("scaling: --require-gpu given but no usable CUDA device", file=sys.stderr)
        return 2
    runtimes = [r for r in args.runtimes if r != "cuda" or with_gpu]
    reference = {}
    if args.reference is not None:
        for ref_row in json.loads(args.reference.read_text(encoding="utf-8"))["rows"]:
            if "numba" in ref_row:
                reference[(ref_row["workload"], ref_row["qubits"])] = ref_row["numba"]

    rows = []
    for workload in args.workloads:
        stopped = set()
        for qubits in sizes_for(workload, args.max_qubits):
            row = {
                "workload": workload,
                "method": WORKLOADS[workload][0],
                "qubits": qubits,
            }
            for runtime in runtimes:
                if runtime in stopped:
                    continue
                # A child that cannot finish its repeats in this budget is a
                # hang, not a data point; the timeout stops it.
                timeout = 60 + args.max_seconds * (args.repeats + 2) * 4
                row[runtime] = measure(
                    workload,
                    qubits,
                    runtime,
                    args.threads,
                    args.repeats,
                    timeout,
                    args.simplify,
                )
                if row[runtime]["median_ms"] > args.max_seconds * 1e3:
                    stopped.add(runtime)
            cpu = row.get("numba") or reference.get((workload, qubits))
            if "cuda" in row:
                # Every GPU row is checked against a CPU result: from this run,
                # or from --reference (same workload, size and CPU code).
                row["verified_against"] = (
                    "numba, this run"
                    if "numba" in row
                    else "numba, reference file" if cpu is not None else None
                )
                if cpu is not None:
                    a, b = cpu["fingerprint"], row["cuda"]["fingerprint"]
                    if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > FINGERPRINT_ATOL:
                        raise RuntimeError(
                            f"GPU and CPU disagree on {workload} n={qubits}: {a} vs {b}"
                        )
                    key = (
                        "speedup_cpu_over_gpu"
                        if "numba" in row
                        else "speedup_reference_cpu_over_gpu"
                    )
                    row[key] = round(cpu["median_ms"] / row["cuda"]["median_ms"], 2)
            if any(r in row for r in runtimes):
                rows.append(row)
                print(
                    f"{workload:14s} n={qubits:2d} "
                    + " ".join(
                        f"{r}={row[r]['median_ms']:.2f}ms" for r in runtimes if r in row
                    )
                    + "".join(
                        f" x{row[k]}"
                        for k in (
                            "speedup_cpu_over_gpu",
                            "speedup_reference_cpu_over_gpu",
                        )
                        if k in row
                    ),
                    flush=True,
                )
            if all(r in stopped for r in runtimes):
                break

    import numpy  # pylint: disable=import-outside-toplevel
    import numba  # pylint: disable=import-outside-toplevel

    software = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "numba": numba.__version__,
    }
    if with_gpu:
        software["cupy"] = subprocess.run(
            [sys.executable, "-c", "import cupy; print(cupy.__version__)"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    output = {
        "description": "Median warm wall time of one public FatQat call per workload and size.",
        "measurement_date": date.today().isoformat(),
        "code_revision": revision(),
        "revision": args.revision_label,
        "simplify": args.simplify,
        "software": software,
        "gpu_measured": with_gpu,
        "configurations": {
            "numba": (
                f"CPU (compiled Numba), {args.threads} threads"
                if "numba" in runtimes
                else None
            ),
            "cuda": "GPU, 1 device" if "cuda" in runtimes else None,
        },
        "reference_file": None if args.reference is None else args.reference.name,
        "workload_common": {
            "circuit": "two layers of parameterised RY and RZ on every qubit plus staggered nearest-neighbour CX",
            "noise": "amplitude damping p=0.07 after RY, phase damping p=0.02 after RZ (dm_* workloads)",
            "observable": "0.8 Z..Z + 0.3 X..X - 0.2 Y..Y (*_observable workloads)",
            "precision": "complex128 on CPU and GPU",
            "shots": 0,
            "timed": "one warm public FatQat call including GPU synchronisation and the requested host output",
            "untimed": "imports, backend and program construction, the first call (reported as first_call_ms)",
        },
        "warm_calls_per_median": args.repeats,
        "max_seconds_per_call": args.max_seconds,
        "speedup_definition": "CPU median / GPU median at the same size; above 1 means the GPU is faster",
        "correctness": f"GPU output fingerprint matches CPU within {FINGERPRINT_ATOL} at every size",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
