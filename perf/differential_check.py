"""Randomized differential check of every 'same result' claim, on fresh seeds.

Thousands of random circuits, with seeds far from the test suite's, compare:
tiles against per-gate kernels (Numba), exact simplifications against the
plain plan (values unchanged, Numba), simplification from the all-zero start,
Clifford+T simplification (equivalent to 1e-12), simplify="auto" against
simplify=False through the public API (counts and states, bit for bit), the
value checks again on CUDA where a GPU is present, where the prototype is
built, the Apple-GPU engine against Numba's tiles (bit patterns), and shot
branching against the one-shot-at-a-time loop (every shot's classical bits, on
every engine available). It reuses the generators of the tile and simplify
tests.

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


def _value_engines():
    # Simplification claims unchanged values on Numba and CUDA; CUDA kernels
    # differ from Numba's (a specialised gate can switch kernel), so it is
    # checked on its own engine, not inferred from Numba.
    engines = {"Numba": NumbaSVEngine}
    try:
        import cupy  # pylint: disable=import-outside-toplevel

        if cupy.cuda.runtime.getDeviceCount() > 0:
            # pylint: disable-next=import-outside-toplevel
            from fatqat.simulator._engine.cupy import CupySVEngine

            engines["CUDA"] = CupySVEngine
    except Exception:  # pylint: disable=broad-except  # no CuPy or no device
        pass
    return engines


VALUE_ENGINES = _value_engines()


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

# 2. simplify on exact-only plans: values unchanged, mixed radix too; never longer.
for label, engine in VALUE_ENGINES.items():
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
        a = S._evolve(engine(), dims, short, ket)
        b = S._evolve(engine(), dims, plan, ket)
        if not np.array_equal(a, b):
            fails += 1
        cases += 1
    results[f"simplify exact-only plans, {label} values unchanged"] = (
        cases,
        fails,
        time.time() - start,
    )
    print(results, flush=True)

# 3. simplify with known inputs (zero start), exact-only and rotations: values unchanged.
for label, engine in VALUE_ENGINES.items():
    start = time.time()
    fails = 0
    cases = 0
    for seed in range(count(2000)):
        rng = np.random.default_rng(BASE + 20_000 + seed)
        dims = (2, 2, 2, 2, 2)
        plan = S._random_plan(rng, dims, 40, noisy=False)
        short = simplify_plan(plan, dims, zero_start=True)
        a = S._evolve(engine(), dims, short, None)
        b = S._evolve(engine(), dims, plan, None)
        if not np.array_equal(a, b):
            fails += 1
        cases += 1
    suffix = "" if label == "Numba" else f" ({label})"
    results[
        f"simplify from the zero start (known inputs), values unchanged{suffix}"
    ] = (cases, fails, time.time() - start)
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


# 4b. simplify="auto" (the default) == simplify=False, every count and state
# bit for bit, through the public API: noisy programs with mid-circuit
# measurement, reset and feedforward, statevector and density matrix, with
# the size threshold set to 0 so the pass always runs.
def _auto_program(rng):
    # pylint: disable=import-outside-toplevel
    import fatqat as fq
    import fatqat.operations as ops
    from fatqat.noise import AmplitudeDamping, Depolarizing, NoiseModel

    n = int(rng.integers(2, 6))
    # A third are ideal and static but for the measurements at the end, so
    # whether a measurement is deferrable decides the run's path.
    static = rng.random() < 1 / 3
    noise = NoiseModel()
    if not static:
        noise.add(Depolarizing(p=float(rng.uniform(0, 0.2))), operation=ops.H)
        noise.add(AmplitudeDamping(p=float(rng.uniform(0, 0.3))), operation=ops.RY)
    program = fq.Program(n, n)
    gates = [ops.H, ops.X, ops.S, ops.Sdg, ops.T, ops.Tdg, ops.Z, ops.SX]
    for _ in range(int(rng.integers(10, 40))):
        kind = int(rng.integers(0, 5 if static else 8))
        q = int(rng.integers(n))
        if kind <= 2:
            program.add(gates[int(rng.integers(len(gates)))], q)
        elif kind == 3:
            program.add(ops.RY(float(rng.uniform(0, 6.3))), q)
        elif kind == 4 and n > 1:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            program.add([ops.CX, ops.CZ, ops.Swap][int(rng.integers(3))], (a, b))
        elif kind == 5:
            program.measure(q, q)
        elif kind == 6:
            program.add(ops.Reset, q)
        else:
            program.add(ops.X, q, condition=(int(rng.integers(n)), 1))
    if not static and rng.random() < 0.5:
        program.measure_all()
    else:
        # Not every qubit re-measured: removing gates after a measurement
        # can then make it deferrable, which auto must not act on.
        for q in range(n):
            if q == 0 or rng.random() < 0.5:
                program.measure(q, q)
                program.add(ops.X, q)
                program.add(ops.X, q)
    return program, noise


from fatqat.simulator import simulator as simulator_module  # noqa: E402

shipped_min_work = dict(simulator_module._AUTO_SIMPLIFY_MIN_WORK)
for runtime in [label.lower() for label in VALUE_ENGINES]:
    start = time.time()
    fails = 0
    cases = 0
    simulator_module._AUTO_SIMPLIFY_MIN_WORK[runtime] = 0
    try:
        for seed in range(count(300)):
            rng = np.random.default_rng(BASE + 35_000 + seed)
            program, noise = _auto_program(rng)
            for method in ("statevector", "density_matrix"):
                backend = Simulator(method, runtime=runtime, noise=noise)
                runs = []
                for simplify in ("auto", False):
                    result = backend.run(
                        program,
                        shots=64,
                        simulation_config={"seed": seed, "simplify": simplify},
                    ).result()
                    single = backend.run(
                        program,
                        shots=1,
                        result_config={"counts": True, "final_state": True},
                        simulation_config={"seed": seed, "simplify": simplify},
                    ).result()
                    state = np.asarray(getattr(single, f"get_{method}")())
                    runs.append((result.get_counts(), state))
                if runs[0][0] != runs[1][0] or not np.array_equal(
                    runs[0][1], runs[1][1]
                ):
                    fails += 1
                cases += 1
    finally:
        simulator_module._AUTO_SIMPLIFY_MIN_WORK.update(shipped_min_work)
    results[f"simplify auto vs off, counts and states bit for bit ({runtime})"] = (
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


# 6. Shot branching == the one-shot-at-a-time loop: every shot's clbits, on
# random engine-level plans with every step kind (channels of both routes,
# conditions with several terms, readout confusion, remapped digits, reset,
# loss and reload), qubits and qutrits, given initial states, small budgets.
def _unitary(rng, size):
    q, r = np.linalg.qr(
        rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    )
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _random_shot_plan(rng, dims, n_clbits, depth):
    # pylint: disable=import-outside-toplevel
    from fatqat._backends.steps import (
        ApplyChannelStep,
        ApplyMatrixStep,
        LossStep,
        MeasurementStep,
        PutStep,
        ResetStep,
    )

    n = len(dims)
    steps = []

    def condition():
        if rng.random() < 0.7:
            return None
        terms = rng.choice(n_clbits, size=int(rng.integers(1, 3)), replace=False)
        return tuple((int(c), int(rng.integers(0, 2))) for c in terms)

    for _ in range(depth):
        kind = rng.choice(
            ["gate", "gate", "gate", "channel", "measure", "reset", "loss", "put"],
            p=[0.25, 0.15, 0.1, 0.2, 0.12, 0.08, 0.05, 0.05],
        )
        width = 1 if n == 1 or rng.random() < 0.6 else 2
        targets = tuple(int(t) for t in rng.choice(n, size=width, replace=False))
        local = int(np.prod([dims[t] for t in targets]))
        if kind == "gate":
            steps.append(ApplyMatrixStep(_unitary(rng, local), targets, condition()))
        elif kind == "channel":
            target = targets[:1]
            d = dims[target[0]]
            if rng.random() < 0.5:  # scaled unitaries: the sampled route
                weights = rng.dirichlet(np.ones(3))
                ops_ = [
                    np.sqrt(w) * (np.eye(d) if i == 0 else _unitary(rng, d))
                    for i, w in enumerate(weights)
                ]
            else:  # a general Kraus set from an isometry: the jump route
                k = int(rng.integers(2, 4))
                v = _unitary(rng, k * d)[:, :d]
                ops_ = [v[i * d : (i + 1) * d] for i in range(k)]
            steps.append(ApplyChannelStep(tuple(ops_), target, condition()))
        elif kind == "measure":
            measured = tuple(
                int(t)
                for t in rng.choice(n, size=int(rng.integers(1, n + 1)), replace=False)
            )
            clbits = tuple(
                int(c) for c in rng.choice(n_clbits, size=len(measured), replace=False)
            )
            confusions = None
            if rng.random() < 0.4:
                confusions = tuple(
                    (
                        rng.dirichlet(np.ones(dims[m]) * 4, size=dims[m]).T
                        if rng.random() < 0.6
                        else None
                    )
                    for m in measured
                )
            maps = None
            if rng.random() < 0.3:
                maps = tuple(
                    tuple(int(x) for x in rng.permutation(dims[m])) for m in measured
                )
            steps.append(MeasurementStep(measured, clbits, confusions, maps))
        elif kind == "reset":
            steps.append(ResetStep(targets, condition()))
        elif kind == "loss":
            steps.append(LossStep(targets, float(rng.uniform(0, 0.5)), condition()))
        else:
            steps.append(PutStep(targets, condition()))
    measured = tuple(range(min(n, n_clbits)))
    steps.append(MeasurementStep(measured, measured))
    return tuple(steps)


def _branching_check():
    # pylint: disable=import-outside-toplevel,protected-access
    from fatqat._backends.engine_contract import (
        _DensityMatrixResultRequest,
        _StateVectorResultRequest,
    )
    from fatqat.simulator._engine import branching, nb as engine_nb, np as engine_np
    from fatqat.simulator._execution_contract import _ExecutionContext, _ExecutionPolicy

    engines = [
        engine_np.NumpySVEngine,
        engine_np.NumpyDMEngine,
        engine_nb.NumbaSVEngine,
        engine_nb.NumbaDMEngine,
    ]
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            from fatqat.simulator._engine import cupy as engine_cupy

            engines += [engine_cupy.CupySVEngine, engine_cupy.CupyDMEngine]
    except Exception:  # pylint: disable=broad-except  # no CuPy or no device
        pass
    serial = _ExecutionPolicy("serial", "serial", 1, False, False)
    shipped_budget = branching._BRANCH_MEMORY_BYTES
    shipped_states = branching._BRANCH_STATES
    branching._BRANCH_STATES = 0  # the drawn byte budgets alone decide
    found = {}
    try:
        for seed in range(count(2500)):
            rng = np.random.default_rng(BASE + 50_000 + seed)
            n = int(rng.integers(1, 6))
            dims = tuple(int(d) for d in rng.choice([2, 2, 2, 3], size=n))
            n_clbits = n + 1
            plan = _random_shot_plan(rng, dims, n_clbits, int(rng.integers(3, 25)))
            size = int(np.prod(dims))
            occupied = (
                None
                if rng.random() < 0.7
                else frozenset(
                    int(t)
                    for t in rng.choice(
                        n, size=int(rng.integers(0, n + 1)), replace=False
                    )
                )
            )
            branching._BRANCH_MEMORY_BYTES = int(
                rng.choice([0, 64 * size, shipped_budget])
            )
            seeds = np.random.SeedSequence(seed).spawn(int(rng.integers(2, 120)))
            for cls in engines:
                engine = cls()
                sv = engine.state_semantics == "sv"
                initial = None
                if rng.random() < 0.3:
                    ket = rng.normal(size=size) + 1j * rng.normal(size=size)
                    ket /= np.linalg.norm(ket)
                    initial = ket if sv else np.outer(ket, ket.conj())
                request = (
                    _StateVectorResultRequest(True, False)
                    if sv
                    else _DensityMatrixResultRequest(True, False)
                )
                context = _ExecutionContext(
                    "per_shot",
                    request,
                    dims,
                    n_clbits,
                    len(seeds),
                    seed,
                    initial,
                    occupied,
                )
                payload = engine.materialize_execution(
                    plan,
                    system_dims=dims,
                    n_clbits=n_clbits,
                    deferred_measurements=(),
                    policy=serial,
                )
                branched = engine.execute_shot_batch(context, payload, seeds, serial)
                alone = [
                    engine.execute_shot_batch(context, payload, [s], serial)[0]
                    for s in seeds
                ]
                cases, fails = found.get(cls.__name__, (0, 0))
                found[cls.__name__] = (cases + 1, fails + (branched != alone))
    finally:
        branching._BRANCH_MEMORY_BYTES = shipped_budget
        branching._BRANCH_STATES = shipped_states
    return found


start = time.time()
for name, (cases, fails) in _branching_check().items():
    results[f"shot branching vs one shot at a time, every clbit ({name})"] = (
        cases,
        fails,
        time.time() - start,
    )
print(results, flush=True)

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
