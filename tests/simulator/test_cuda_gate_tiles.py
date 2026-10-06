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

    class Tiled(CupySVEngine):  # pylint: disable=too-many-ancestors
        # Test systems are smaller than any L2 cache; force tiling on.
        _TILE_MIN_BYTES = 0

    class Tiled12(CupySVEngine):  # pylint: disable=too-many-ancestors
        _TILE_MIN_BYTES = 0
        _TILE_BITS = 12  # 64 KiB of shared memory: exercises the opt-in path

    class PerGate(CupySVEngine):  # pylint: disable=too-many-ancestors
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


def test_reading_the_state_applies_queued_gates_first(engines):
    tiled = engines[0]
    n = 12
    engine = tiled(device_id=0)
    engine.initialize((2,) * n)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    engine.apply(ApplyMatrixStep(x, (0,)))
    engine.apply(ApplyMatrixStep(x, (1,)))
    probabilities = engine.probabilities()
    assert probabilities[0b11] == 1.0


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


@pytest.fixture(scope="module", name="unitary_engines")
def _unitary_engines(engines):
    del engines  # skips the module without a CUDA device
    from fatqat.simulator._engine.cupy import CupyUnitaryEngine

    class TiledUnitary(CupyUnitaryEngine):  # pylint: disable=too-many-ancestors
        _TILE_MIN_BYTES = 0

    class PerGateUnitary(CupyUnitaryEngine):  # pylint: disable=too-many-ancestors
        _TILE_BITS = 64

    return TiledUnitary, PerGateUnitary


def _run_unitary(engine_cls, n, steps):
    engine = engine_cls(device_id=0)
    engine.initialize((2,) * n)
    for step in steps:
        engine.apply(step)
    return engine.export_state()


@pytest.mark.parametrize("n", [6, 7, 8])
@pytest.mark.parametrize("seed", range(4))
def test_unitary_tiles_are_bit_identical_to_per_gate_kernels(unitary_engines, n, seed):
    tiled, per_gate = unitary_engines
    steps = _random_steps(np.random.default_rng(500 + 10 * n + seed), n, 60)
    np.testing.assert_array_equal(
        _run_unitary(tiled, n, steps), _run_unitary(per_gate, n, steps)
    )


def test_public_unitary_above_l2_matches_cpu(engines):
    # 12 qubits: a 256 MiB unitary, above current L2 caches, so tiles engage.
    del engines
    request = {"counts": False, "final_state": True}
    values = [
        Simulator("unitary", runtime=runtime)
        .run(_layered(12), shots=0, result_config=request)
        .result()
        .get_unitary()
        for runtime in ("numba", "cuda")
    ]
    np.testing.assert_allclose(values[1], values[0], atol=1e-12, rtol=0)


def _controlled(matrix, value=1):
    """``matrix`` on the later targets where the first target equals ``value``."""
    size = len(matrix)
    out = np.eye(2 * size, dtype=np.complex128)
    out[value * size : (value + 1) * size, value * size : (value + 1) * size] = matrix
    return out


def _insular_steps(rng, n, depth):
    """Gates whose controls and diagonal targets need no tile bit.

    The twin of the Numba tile tests' generator: Toffoli, Fredkin, three-qubit
    diagonals, open and closed controls on dense gates, and plain gates.
    """
    steps = []
    for _ in range(depth):
        kind = int(rng.integers(8))
        width = 3 if kind in (0, 1, 2, 5) else 2
        targets = tuple(int(q) for q in rng.choice(n, width, replace=False))
        if kind == 0:
            matrix = _controlled(_CX)
        elif kind == 1:
            matrix = _controlled(_SWAP)
        elif kind == 2:
            matrix = np.diag(np.exp(1j * rng.normal(size=8)))
        elif kind == 3:
            matrix = _controlled(_unitary(rng, 2), value=int(rng.integers(2)))
        elif kind == 4:
            matrix = np.diag(np.exp(1j * rng.normal(size=4)))
        elif kind == 5:
            matrix = _controlled(_unitary(rng, 4))
        elif kind == 6:
            matrix = _CX
        else:
            targets = targets[:1]
            matrix = _unitary(rng, 2)
        steps.append(ApplyMatrixStep(matrix, targets))
    return steps


@pytest.mark.parametrize("n", [11, 13, 16])
@pytest.mark.parametrize("seed", range(5))
def test_controls_and_diagonals_anywhere_stay_bit_identical(engines, n, seed):
    tiled, per_gate, _tiled12 = engines
    rng = np.random.default_rng(2000 * n + seed)
    initial = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    initial /= np.linalg.norm(initial)
    steps = _insular_steps(rng, n, 80)
    np.testing.assert_array_equal(
        _run(tiled, n, steps, initial), _run(per_gate, n, steps, initial)
    )


@pytest.mark.parametrize("seed", range(3))
def test_unitary_tiles_with_controls_anywhere_stay_bit_identical(unitary_engines, seed):
    tiled, per_gate = unitary_engines
    steps = _insular_steps(np.random.default_rng(700 + seed), 7, 60)
    np.testing.assert_array_equal(
        _run_unitary(tiled, 7, steps), _run_unitary(per_gate, 7, steps)
    )
