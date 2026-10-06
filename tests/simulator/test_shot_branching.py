"""Shot branching gives exactly the one-shot-at-a-time results.

Each case runs a seeded program twice through the public API: once as shipped
(shots travel in groups, `branching._run_branched`) and once with that
replaced by the reference loop that evolves every shot alone. Counts must be
equal, which (shots being independent) means every shot's classical bits are.
A spy checks that the per-shot path really ran, so no case can pass by taking
another route.
"""

from pathlib import Path

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.noise import (
    AmplitudeDamping,
    Depolarizing,
    Loss,
    NoiseModel,
    PhaseDamping,
    ReadoutConfusion,
    TransitionRelaxation,
)
from fatqat.simulator import Simulator
from fatqat.simulator._engine import branching
from fatqat.simulator._engine import np as engine_np
from fatqat.simulator.fake_atom_array import AtomArraySimulator

# In this process (the spy and the reference loop live here, not in a
# process worker), and for Numba not its own compiled multi-shot loop.
_SERIAL = {"shot_parallelism": "serial"}
_PER_SHOT = {"shot_parallelism": "serial", "kernel_parallelism": "threads"}


def _require(runtime):
    if runtime == "numba":
        pytest.importorskip("numba")
    elif runtime == "cuda":
        cupy = pytest.importorskip("cupy")
        try:
            if cupy.cuda.runtime.getDeviceCount() == 0:
                pytest.skip("No CUDA device available")
        except cupy.cuda.runtime.CUDARuntimeError as error:
            pytest.skip(f"CUDA device discovery unavailable: {error}")


def _reference(engine, plan, seed_sequences, initial_occupied, initial_state):
    """The one-shot-at-a-time loop shot branching replaces."""
    snapshots = []
    for seed_sequence in seed_sequences:
        engine.initialize(engine._dims, engine._n_clbits, initial_state=initial_state)
        snapshots.append(
            engine._run_one_shot(
                plan, np.random.default_rng(seed_sequence), initial_occupied
            )
        )
    return snapshots


def _both(monkeypatch, run):
    """``run()``'s counts with branching and with the reference loop."""
    calls = []
    shipped = engine_np._run_branched

    def branched(*args):
        calls.append(len(args[2]))
        return shipped(*args)

    monkeypatch.setattr(engine_np, "_run_branched", branched)
    got = run()
    assert calls and all(n > 1 for n in calls), "the per-shot path did not run"
    monkeypatch.setattr(engine_np, "_run_branched", _reference)
    return got, run()


def _qubit_program(rng):
    n = int(rng.integers(2, 6))
    noise = NoiseModel()
    noise.add(Depolarizing(p=float(rng.uniform(0, 0.3))), operation=ops.H)
    noise.add(AmplitudeDamping(p=float(rng.uniform(0, 0.4))), operation=ops.RY)
    noise.add(PhaseDamping(p=float(rng.uniform(0, 0.4))), operation=ops.T)
    if rng.random() < 0.5:
        flip = float(rng.uniform(0, 0.2))
        noise.add(
            ReadoutConfusion(np.array([[1 - flip, flip], [flip, 1 - flip]])),
            targets=int(rng.integers(n)),
        )
    program = fq.Program(n, n)
    # A measurement mid-way makes every case a per-shot run.
    program.add(ops.H, 0)
    program.measure(0, 0)
    program.add(ops.H, 0)
    for _ in range(int(rng.integers(5, 30))):
        kind = int(rng.integers(0, 7))
        q = int(rng.integers(n))
        if kind == 0:
            program.add([ops.H, ops.X, ops.S, ops.T][int(rng.integers(4))], q)
        elif kind == 1:
            program.add(ops.RY(float(rng.uniform(0, 6.3))), q)
        elif kind == 2 and n > 1:
            a, b = rng.choice(n, 2, replace=False)
            program.add(ops.CX, (int(a), int(b)))
        elif kind == 3:
            program.measure(q, int(rng.integers(n)))
        elif kind == 4:
            program.add(ops.Reset, q)
        elif kind == 5:
            program.add(ops.H, q, condition=(int(rng.integers(n)), 1))
        else:
            program.add(ops.Reset, q, condition=(int(rng.integers(n)), 0))
    program.measure_all()
    return program, noise


@pytest.mark.parametrize("runtime", ["numpy", "numba", "cuda"])
@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
@pytest.mark.parametrize("seed", range(40))
def test_branching_counts_equal_one_shot_at_a_time(monkeypatch, runtime, method, seed):
    _require(runtime)
    program, noise = _qubit_program(np.random.default_rng(9_000 + seed))
    backend = Simulator(method, runtime=runtime, noise=noise)
    config = {"seed": seed, **(_PER_SHOT if runtime == "numba" else _SERIAL)}
    got, expected = _both(
        monkeypatch,
        lambda: backend.run(program, shots=300, simulation_config=config)
        .result()
        .get_counts(),
    )
    assert got == expected


@pytest.mark.parametrize("runtime", ["numpy", "numba", "cuda"])
@pytest.mark.parametrize("seed", range(8))
def test_branching_on_qutrits_equals_one_shot_at_a_time(monkeypatch, runtime, seed):
    _require(runtime)
    noise = NoiseModel()
    noise.add(
        TransitionRelaxation(p=0.2, coefficients={(1, 0): 1, (2, 1): 1}),
        operation=ops.Shift,
    )
    noise.add(PhaseDamping(p=0.15), operation=ops.Shift)
    qreg = fq.QuantumRegister(3, dim=3)
    creg = fq.ClassicalRegister(3, dim=3)
    program = fq.Program([qreg], [creg])
    program.add(ops.Shift(1), qreg[0])
    program.add(ops.Shift(2), qreg[2])
    program.measure(qreg[0], creg[0])
    program.add(ops.Shift(1), qreg[1], condition=(creg[0], 1))
    program.add(ops.Reset, qreg[0])
    program.add(ops.Shift(2), qreg[0], condition=(creg[0], 2))
    program.measure(qreg[1], creg[1])
    program.measure(qreg[2], creg[2])
    backend = Simulator("SV", runtime=runtime, noise=noise)
    config = {"seed": seed, **(_PER_SHOT if runtime == "numba" else _SERIAL)}
    got, expected = _both(
        monkeypatch,
        lambda: backend.run(program, shots=400, simulation_config=config)
        .result()
        .get_counts(),
    )
    assert len(expected) > 3
    assert got == expected


@pytest.mark.parametrize("seed", range(6))
def test_branching_through_atom_loss_and_reload_equals_one_shot(monkeypatch, seed):
    # Loss and reload continue shot by shot from the group's state.
    noise = NoiseModel()
    noise.add(Loss(p=0.3), operation=ops.RX)
    program = fq.Program(2, 2)
    program.add(ops.Put, (0, 1))
    program.add(ops.RX(0.7), 0)
    program.add(ops.RX(1.1), 1)
    program.measure(0, 0)
    program.add(ops.Put, (0, 1))
    program.add(ops.RX(0.4), 0, condition=(0, 1))
    program.measure_all()
    backend = AtomArraySimulator(noise=noise)
    got, expected = _both(
        monkeypatch,
        lambda: backend.run(
            program, shots=300, simulation_config={"seed": seed, **_SERIAL}
        )
        .result()
        .get_counts(),
    )
    assert len(expected) > 2
    assert got == expected


@pytest.mark.parametrize("budget", [0, 1, 3])
def test_a_tiny_memory_budget_still_gives_exact_counts(monkeypatch, budget):
    # Past the budget, split-off branches run shot by shot at once.
    states = []
    shipped_replay = branching._Branching._replay

    def replay(self, plan, shots):
        states.append(len(shots))
        return shipped_replay(self, plan, shots)

    monkeypatch.setattr(branching, "_BRANCH_MEMORY_BYTES", budget * 64)
    monkeypatch.setattr(branching, "_BRANCH_STATES", 0)  # the bytes alone
    monkeypatch.setattr(branching._Branching, "_replay", replay)
    program, noise = _qubit_program(np.random.default_rng(77))
    backend = Simulator("statevector", runtime="numpy", noise=noise)
    got, expected = _both(
        monkeypatch,
        lambda: backend.run(
            program, shots=200, simulation_config={"seed": 5, **_SERIAL}
        )
        .result()
        .get_counts(),
    )
    assert got == expected
    assert states, "the budget never forced a shot-by-shot replay"


def test_branching_shares_the_work_of_shots_in_one_state(monkeypatch):
    # An ideal circuit measured mid-way, with feedforward, has two branches,
    # so the gates after the measurement run twice, not once per shot.
    applied = []
    shipped = engine_np.NumpySVEngine.apply

    def apply(self, step):
        applied.append(step)
        return shipped(self, step)

    monkeypatch.setattr(engine_np.NumpySVEngine, "apply", apply)
    program = fq.Program(2, 2)
    program.add(ops.H, 0)
    program.measure(0, 0)
    program.add(ops.X, 1, condition=(0, 1))
    for _ in range(10):
        program.add(ops.RY(0.3), 1)
    program.measure(1, 1)
    Simulator("statevector", runtime="numpy").run(
        program, shots=500, simulation_config={"seed": 1, **_SERIAL}
    ).result()
    # H once, X once (only the branch that measured 1), each RY per branch.
    assert len(applied) == 1 + 1 + 2 * 10


def _wide_measurement_program(n):
    # Every shot gets its own outcome mid-way, and the plan goes on after it.
    program = fq.Program(n, n)
    for q in range(n):
        program.add(ops.H, q)
    program.measure(tuple(range(n)), tuple(range(n)))
    program.add(ops.X, 0, condition=(0, 1))
    program.add(ops.RY(0.4), 1)
    program.measure_all()
    return program


def test_many_outcomes_stay_within_the_memory_budget(monkeypatch):
    # 400 shots, nearly 400 outcomes: the outcome groups share one base and
    # are projected one at a time, so peak memory is a few states, not 400.
    import tracemalloc

    n = 14
    state_bytes = 16 << n
    monkeypatch.setattr(branching, "_BRANCH_MEMORY_BYTES", 4 * state_bytes)
    monkeypatch.setattr(branching, "_BRANCH_STATES", 0)  # the bytes alone
    program = _wide_measurement_program(n)
    backend = Simulator("statevector", runtime="numpy")

    def run():
        return (
            backend.run(program, shots=400, simulation_config={"seed": 2, **_SERIAL})
            .result()
            .get_counts()
        )

    run()  # warm caches outside the measurement
    tracemalloc.start()
    try:
        counts = run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(counts) > 300
    assert peak < 16 * state_bytes, f"peak {peak / state_bytes:.0f} states"
    got, expected = _both(monkeypatch, run)
    assert got == expected


def test_the_last_step_builds_no_state(monkeypatch):
    # The final measurement's outcomes only write classical bits.
    projections = []
    shipped = engine_np.NumpySVEngine._project

    def project(self, indices, idx):
        projections.append(tuple(indices))
        return shipped(self, indices, idx)

    monkeypatch.setattr(engine_np.NumpySVEngine, "_project", project)
    program = _wide_measurement_program(6)
    Simulator("statevector", runtime="numpy").run(
        program, shots=300, simulation_config={"seed": 4, **_SERIAL}
    ).result()
    # One projection per outcome of the mid-way measurement (at most 2**6 of
    # them), none at the end, which would add one per final outcome of each
    # of those groups (hundreds more for 300 shots).
    assert 0 < len(projections) <= 2**6


def test_channel_branches_stay_within_the_memory_budget(monkeypatch):
    # A two-qubit depolarizing channel has 16 branches; they share the one
    # state they were drawn from, so peak memory is a few states, not 16.
    import tracemalloc

    n = 14
    state_bytes = 16 << n
    monkeypatch.setattr(branching, "_BRANCH_MEMORY_BYTES", state_bytes)
    monkeypatch.setattr(branching, "_BRANCH_STATES", 0)  # the bytes alone
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.9), operation=ops.CX)
    program = fq.Program(n, n)
    for q in range(n):
        program.add(ops.H, q)
    program.add(ops.CX, (0, 1))
    program.add(ops.RY(0.3), 2)
    program.measure_all()
    backend = Simulator("statevector", runtime="numpy", noise=noise)

    def run():
        return (
            backend.run(program, shots=300, simulation_config={"seed": 6, **_SERIAL})
            .result()
            .get_counts()
        )

    run()
    tracemalloc.start()
    try:
        run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * state_bytes, f"peak {peak / state_bytes:.0f} states"
    got, expected = _both(monkeypatch, run)
    assert got == expected


@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_shots_in_chunks_equal_one_shot_at_a_time(monkeypatch, chunk):
    monkeypatch.setattr(branching, "_SHOTS_PER_CHUNK", chunk)
    program, noise = _qubit_program(np.random.default_rng(31))
    backend = Simulator("statevector", runtime="numpy", noise=noise)
    got, expected = _both(
        monkeypatch,
        lambda: backend.run(
            program, shots=150, simulation_config={"seed": 8, **_SERIAL}
        )
        .result()
        .get_counts(),
    )
    assert got == expected


@pytest.mark.parametrize("seed", [1, 6])
def test_the_largest_group_goes_on_so_a_tight_budget_replays_few_shots(
    monkeypatch, seed
):
    # With these seeds the first shot is in a rare noise branch. If its part
    # went on in place, the large no-error group would wait, overflow the
    # budget and replay shot by shot: 396 of 400 shots, measured. The largest
    # part goes on instead, and only small groups ever replay.
    import sys  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perf"))
    import branching_check  # pylint: disable=import-outside-toplevel,import-error

    n = 10
    program, noise = branching_check.WORKLOADS["low_noise"](n)
    monkeypatch.setattr(branching, "_BRANCH_MEMORY_BYTES", 2 * (16 << n))
    monkeypatch.setattr(branching, "_BRANCH_STATES", 0)
    replayed = []
    shipped_replay = branching._Branching._replay

    def replay(self, plan, shots):
        replayed.append(len(shots))
        return shipped_replay(self, plan, shots)

    monkeypatch.setattr(branching._Branching, "_replay", replay)
    backend = Simulator("statevector", runtime="numpy", noise=noise)

    def run():
        return (
            backend.run(program, shots=400, simulation_config={"seed": seed, **_SERIAL})
            .result()
            .get_counts()
        )

    run()
    assert sum(replayed) < 100
    got, expected = _both(monkeypatch, run)
    assert got == expected
