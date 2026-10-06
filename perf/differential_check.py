"""Randomized differential check of every 'same result' claim, on fresh seeds.

Thousands of random circuits, with seeds far from the test suite's, compare:
tiles against per-gate kernels (Numba), exact simplifications against the
plain plan (values unchanged, Numba), simplification from the all-zero start,
Clifford+T simplification (equivalent to 1e-12), and, where the prototype is
built, the Apple-GPU engine against Numba's tiles (bit patterns). It reuses the
generators of the tile and simplify tests.

Usage, from the repository root:
    python perf/differential_check.py --out results/differential-check.json
"""

import argparse
from datetime import date
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "simulator"))
sys.path.insert(0, str(ROOT / "prototypes" / "metal"))
sys.path.insert(0, str(ROOT / "perf"))

# pylint: disable=wrong-import-position,import-error  # paths set up above
import test_numba_tiles as T  # noqa: E402
import test_simplify as S  # noqa: E402
from fatqat._backends.simplify import simplify_plan  # noqa: E402
from fatqat.simulator import Simulator  # noqa: E402
from fatqat.simulator._engine.nb import NumbaSVEngine  # noqa: E402
from tile_check import revision  # noqa: E402

BASE = 1_000_000
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", type=Path)
parser.add_argument(
    "--scale", type=float, default=1.0, help="multiply every case count"
)
args = parser.parse_args()


def count(n):
    return max(1, int(n * args.scale))


results = {}


def mixed_steps(rng, n, depth):
    half = depth // 2
    steps = T._random_steps(rng, n, half) + T._insular_steps(rng, n, depth - half)
    order = rng.permutation(len(steps))
    return [steps[i] for i in order]


# 1. Numba tiles == per-gate kernels (values), many shapes and sizes.
start = time.time()
fails = 0
cases = 0
for seed in range(count(3000)):
    rng = np.random.default_rng(BASE + seed)
    n = int(rng.integers(7, 13))
    initial = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    initial /= np.linalg.norm(initial)
    steps = mixed_steps(rng, n, int(rng.integers(20, 120)))
    if not np.array_equal(
        T._run(T._Tiled, n, steps, initial), T._run(T._PerGate, n, steps, initial)
    ):
        fails += 1
    cases += 1
results["numba tiles vs per-gate (random + controlled/diagonal gates)"] = (
    cases,
    fails,
    time.time() - start,
)
print(results, flush=True)

# 2. simplify on exact-only plans: values unchanged (Numba), mixed radix too; never longer.
start = time.time()
fails = 0
cases = 0
for seed in range(count(3000)):
    rng = np.random.default_rng(BASE + 10_000 + seed)
    dims = [(2, 2, 2, 2), (3, 2, 3), (2, 2, 2, 2, 2), (2, 3, 2, 2)][seed % 4]
    size = int(np.prod(dims))
    ket = rng.normal(size=size) + 1j * rng.normal(size=size)
    ket /= np.linalg.norm(ket)
    plan = S._random_plan(rng, dims, 40, noisy=False)
    short = simplify_plan(plan, dims)
    if len(short) > len(plan):
        fails += 1
    a = S._evolve(S.numba_engines[0](), dims, short, ket)
    b = S._evolve(S.numba_engines[0](), dims, plan, ket)
    if not np.array_equal(a, b):
        fails += 1
    cases += 1
results["simplify exact-only plans, Numba values unchanged"] = (
    cases,
    fails,
    time.time() - start,
)
print(results, flush=True)

# 3. simplify with known inputs (zero start), exact-only and rotations: values unchanged.
start = time.time()
fails = 0
cases = 0
for seed in range(count(2000)):
    rng = np.random.default_rng(BASE + 20_000 + seed)
    dims = (2, 2, 2, 2, 2)
    plan = S._random_plan(rng, dims, 40, noisy=False)
    short = simplify_plan(plan, dims, zero_start=True)
    a = S._evolve(S.numba_engines[0](), dims, short, None)
    b = S._evolve(S.numba_engines[0](), dims, plan, None)
    if not np.array_equal(a, b):
        fails += 1
    cases += 1
results["simplify from the zero start (known inputs), values unchanged"] = (
    cases,
    fails,
    time.time() - start,
)
print(results, flush=True)

# 4. simplify on Clifford+T circuits (rounding removed): equivalent within 1e-12, statevector and unitary.
start = time.time()
fails = 0
cases = 0
for seed in range(count(600)):
    rng = np.random.default_rng(BASE + 30_000 + seed)
    program = S._clifford_t_program(rng, 4, 80)
    for method in ("statevector", "unitary"):
        backend = Simulator(method, runtime="numba")
        request = {"counts": False, "final_state": True}
        plain, simple = (
            getattr(
                backend.run(
                    program,
                    shots=0,
                    result_config=request,
                    simulation_config={"simplify": s},
                ).result(),
                f"get_{method}",
            )()
            for s in (False, True)
        )
        if not np.allclose(plain, simple, atol=1e-12, rtol=0):
            fails += 1
        cases += 1
results["simplify on Clifford+T (rounding removed), equivalent to 1e-12"] = (
    cases,
    fails,
    time.time() - start,
)
print(results, flush=True)

# 5. Metal prototype == Numba tiles, bit patterns, random circuits at 11-13 qubits.
try:
    import metal_engine as M  # needs prototypes/metal/libfqmetal.dylib (macOS)

    start = time.time()
    fails = 0
    cases = 0
    numba_tiles = type("NumbaTiles", (NumbaSVEngine,), {"_TILE_MIN_BYTES": 0})
    for seed in range(count(300)):
        rng = np.random.default_rng(BASE + 40_000 + seed)
        n = int(rng.integers(11, 14))
        initial = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        initial /= np.linalg.norm(initial)
        steps = mixed_steps(rng, n, 80)
        states = []
        for cls in (numba_tiles, M.MetalSVEngine):
            engine = cls()
            engine.initialize((2,) * n, initial_state=initial)
            for step in steps:
                engine.apply(step)
            states.append(np.array(engine.state))
        if not M.bit_equal(*states):
            fails += 1
        cases += 1
    results["Metal prototype vs Numba tiles, bit patterns"] = (
        cases,
        fails,
        time.time() - start,
    )
except OSError as error:  # no Apple GPU or prototype not built: not a failure
    results["Metal prototype vs Numba tiles, bit patterns"] = ("skipped", 0, 0.0)
    print(f"Metal check skipped: {error}", flush=True)

print("\nSUMMARY")
for name, (cases, fails, seconds) in results.items():
    print(f"{name}: {cases} cases, {fails} failures ({seconds:.0f}s)")
if args.out:
    args.out.write_text(
        json.dumps(
            {
                "description": __doc__.splitlines()[0],
                "measurement_date": date.today().isoformat(),
                "code_revision": revision(),
                "seed_base": BASE,
                "checks": [
                    {"claim": name, "cases": cases, "failures": fails}
                    for name, (cases, fails, _seconds) in results.items()
                ],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
failed = any(f for _c, f, _s in results.values())
sys.exit(1 if failed else 0)
