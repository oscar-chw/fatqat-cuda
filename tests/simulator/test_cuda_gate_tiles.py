"""Shared-memory gate tiles must reproduce the per-gate CUDA path bit for bit.

The tile kernel applies queued qubit gates with the same per-gate arithmetic
as the one-pass-per-gate kernel, so results are compared with exact array
equality, not a tolerance. Circuits mix diagonal, permutation and dense gates,
reversed two-qubit targets, and targets inside and outside the coalesced low
qubits, so tiles of every shape are exercised.
"""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.steps import ApplyMatrixStep
from fatqat.simulator import Simulator

_CX = np.eye(4, dtype=np.complex128)[[0, 1, 3, 2]]
_SWAP = np.eye(4, dtype=np.complex128)[[0, 2, 1, 3]]


@pytest.fixture(scope="module", name="engines")
def _engines():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device available")
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device discovery unavailable: {error}")
    from fatqat.simulator._engine.cupy import CupySVEngine

    class Tiled(CupySVEngine):
        # Test systems are smaller than any L2 cache; force tiling on.
        _TILE_MIN_BYTES = 0

    class Tiled12(CupySVEngine):
        _TILE_MIN_BYTES = 0
        _TILE_BITS = 12  # 64 KiB of shared memory: exercises the opt-in path

    class PerGate(CupySVEngine):
        _TILE_BITS = 64  # larger than any test system: tiling never engages

    return Tiled, PerGate, Tiled12


def _unitary(rng, size):
    q, _ = np.linalg.qr(
        rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    )
    return q


def _random_steps(rng, n, depth):
    steps = []
    for _ in range(depth):
        kind = rng.integers(6)
        if kind == 0:
            steps.append(ApplyMatrixStep(_unitary(rng, 2), (int(rng.integers(n)),)))
        elif kind == 1:
            phases = np.exp(1j * rng.normal(size=2))
            steps.append(ApplyMatrixStep(np.diag(phases), (int(rng.integers(n)),)))
        else:
            pair = tuple(int(q) for q in rng.choice(n, 2, replace=False))
            matrix = {
                2: _CX,
                3: _SWAP,
                4: np.diag(np.exp(1j * rng.normal(size=4))),
                5: _unitary(rng, 4),
            }[int(kind)]
            steps.append(ApplyMatrixStep(matrix, pair))
    return steps


def _run(engine_cls, n, steps, initial):
    engine = engine_cls(device_id=0)
    engine.initialize((2,) * n, initial_state=initial)
    for step in steps:
        engine.apply(step)
    return engine.export_state()


@pytest.mark.parametrize("n", [11, 12, 14, 16])
@pytest.mark.parametrize("seed", range(6))
def test_tiles_are_bit_identical_to_per_gate_kernels(engines, n, seed):
    tiled, per_gate, tiled12 = engines
    rng = np.random.default_rng(1000 * n + seed)
    initial = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    initial /= np.linalg.norm(initial)
    steps = _random_steps(rng, n, 80)
    expected = _run(per_gate, n, steps, initial)
    np.testing.assert_array_equal(_run(tiled, n, steps, initial), expected)
    if n >= 12:
        np.testing.assert_array_equal(_run(tiled12, n, steps, initial), expected)


def test_low_and_high_qubit_runs_split_into_several_tiles(engines):
    tiled, per_gate, _ = engines
    n = 16
    rng = np.random.default_rng(7)
    # One-qubit gates on every qubit in order force a flush when the tile fills.
    steps = [ApplyMatrixStep(_unitary(rng, 2), (q,)) for q in range(n)] * 2
    initial = np.zeros(1 << n, dtype=np.complex128)
    initial[0] = 1
    engine = tiled(device_id=0)
    engine.initialize((2,) * n, initial_state=initial)
    batches = []
    original = engine._apply_tile_batch
    engine._apply_tile_batch = lambda queued: (
        batches.append(len(queued)),
        original(queued),
    )
    for step in steps:
        engine.apply(step)
    result = engine.export_state()
    assert len(batches) >= 3 and sum(batches) <= len(steps)
    np.testing.assert_array_equal(result, _run(per_gate, n, steps, initial))


def test_reading_the_state_applies_queued_gates_first(engines):
    tiled = engines[0]
    n = 12
    engine = tiled(device_id=0)
    engine.initialize((2,) * n)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    engine.apply(ApplyMatrixStep(x, (0,)))
    engine.apply(ApplyMatrixStep(x, (1,)))
    assert engine._pending  # queued, not yet applied
    probabilities = engine.probabilities()
    assert probabilities[0b11] == 1.0
    assert not engine._pending


def _layered(n):
    program = fq.Program(n)
    for layer in range(2):
        for q in range(n):
            program.add(ops.RY(0.31 + q * 0.013 + layer * 0.071), q)
            program.add(ops.RZ(-0.27 + q * 0.011), q)
        for q in range(layer % 2, n - 1, 2):
            program.add(ops.CX, (q, q + 1))
    return program


@pytest.mark.parametrize("simplify", [False, True])
def test_public_runs_and_expectations_match_cpu(engines, simplify):
    del engines
    n = 14
    request = {"counts": False, "final_state": True}
    config = {"simplify": simplify}
    cpu = Simulator("statevector", runtime="numpy").run(
        _layered(n), shots=0, result_config=request, simulation_config=config
    )
    gpu = Simulator("statevector", runtime="cuda").run(
        _layered(n), shots=0, result_config=request, simulation_config=config
    )
    np.testing.assert_allclose(
        gpu.result().get_statevector(),
        cpu.result().get_statevector(),
        atol=1e-12,
        rtol=0,
    )
    observable = fq.Observable([("Z" + "I" * (n - 2) + "Z", 0.8), ("X" * n, 0.3)])
    values = [
        fq.Estimator(Simulator("statevector", runtime=runtime))
        .run(_layered(n), observable, shots=0, simulation_config=config)
        .result()
        .get_expectation()
        for runtime in ("numpy", "cuda")
    ]
    assert values[1] == pytest.approx(values[0], abs=1e-12)


def test_tiling_waits_for_states_larger_than_l2(engines):
    from fatqat.simulator._engine.cupy import CupySVEngine

    engine = CupySVEngine(device_id=0)
    engine.initialize((2,) * 12)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    engine.apply(ApplyMatrixStep(x, (0,)))
    # 2**12 amplitudes are far below any L2 cache: applied at once, not queued.
    assert not engine._pending
    assert engine._tile_min_bytes > 0
    del engines


def test_public_observable_above_l2_matches_cpu(engines):
    # 24 qubits (256 MiB) exceeds the L2 cache of current GPUs, so the public
    # path runs on tiles with the default engine settings.
    del engines
    n = 24
    observable = fq.Observable([("Z" + "I" * (n - 2) + "Z", 0.8), ("X" * n, 0.3)])
    values = [
        fq.Estimator(Simulator("statevector", runtime=runtime))
        .run(_layered(n), observable, shots=0)
        .result()
        .get_expectation()
        for runtime in ("numba", "cuda")
    ]
    assert values[1] == pytest.approx(values[0], abs=1e-12)
