"""Numba cache tiles must reproduce the per-gate coset kernels bit for bit.

Tiles reuse the coset kernels' per-amplitude arithmetic, so the comparison is
exact array equality. Small tiles on small systems exercise many tiles, both
target orders, targets inside and outside the coalesced low qubits, and every
structure (diagonal, permutation, dense).
"""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.steps import ApplyMatrixStep
from fatqat.simulator import Simulator

pytest.importorskip("numba")

# pylint: disable=wrong-import-position  # imports require the guard above

from fatqat.simulator._engine.nb import NumbaSVEngine, NumbaUnitaryEngine

_CX = np.eye(4, dtype=np.complex128)[[0, 1, 3, 2]]
_SWAP = np.eye(4, dtype=np.complex128)[[0, 2, 1, 3]]


class _Tiled(NumbaSVEngine):
    _TILE_BITS = 6
    _TILE_MIN_BYTES = 0


class _PerGate(NumbaSVEngine):
    _TILE_BITS = 64  # larger than any test system: tiling never engages


def _unitary(rng, size):
    q, _ = np.linalg.qr(
        rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    )
    return q


def _random_steps(rng, n, depth):
    steps = []
    for _ in range(depth):
        kind = int(rng.integers(6))
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
            }[kind]
            steps.append(ApplyMatrixStep(matrix, pair))
    return steps


def _run(engine_cls, n, steps, initial):
    engine = engine_cls()
    engine.initialize((2,) * n, initial_state=initial)
    for step in steps:
        engine.apply(step)
    return engine.export_state()


@pytest.mark.parametrize("n", [6, 7, 9, 11])
@pytest.mark.parametrize("seed", range(5))
def test_tiles_are_bit_identical_to_per_gate_kernels(n, seed):
    rng = np.random.default_rng(100 * n + seed)
    initial = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    initial /= np.linalg.norm(initial)
    steps = _random_steps(rng, n, 60)
    np.testing.assert_array_equal(
        _run(_Tiled, n, steps, initial), _run(_PerGate, n, steps, initial)
    )


def test_runs_split_into_several_tiles_and_reads_flush_first():
    n = 10
    rng = np.random.default_rng(1)
    steps = [ApplyMatrixStep(_unitary(rng, 2), (q,)) for q in range(n)] * 2
    engine = _Tiled()
    engine.initialize((2,) * n)
    flushed = []
    original = engine._flush_pending
    engine._flush_pending = lambda: (flushed.append(len(engine._pending)), original())
    for step in steps:
        engine.apply(step)
    assert engine._pending  # still queued until something reads the state
    result = engine.export_state()
    assert not engine._pending
    assert sum(1 for size in flushed if size > 1) >= 3
    initial = np.zeros(1 << n, dtype=np.complex128)
    initial[0] = 1
    np.testing.assert_array_equal(result, _run(_PerGate, n, steps, initial))


def test_small_states_and_mixed_radix_are_not_tiled():
    engine = NumbaSVEngine()
    engine.initialize((2,) * 12)  # 64 KiB: below the default threshold
    engine.apply(ApplyMatrixStep(np.eye(2, dtype=np.complex128)[::-1], (0,)))
    assert not engine._pending
    tiled = _Tiled()
    tiled.initialize((2,) * 6 + (3,))
    tiled.apply(ApplyMatrixStep(np.eye(2, dtype=np.complex128)[::-1], (0,)))
    assert not tiled._pending


def test_unitary_engine_never_queues():
    class TiledUnitary(NumbaUnitaryEngine):  # pylint: disable=too-many-ancestors
        _TILE_BITS = 2
        _TILE_MIN_BYTES = 0

    program = fq.Program(3)
    for q in range(3):
        program.add(ops.H, q)
    program.add(ops.CX, (0, 1))
    request = {"counts": False, "final_state": True}
    reference = Simulator("unitary", runtime="numpy").run(
        program, shots=0, result_config=request
    )
    backend = Simulator("unitary", runtime="numba")
    backend._engine = TiledUnitary()
    actual = backend.run(program, shots=0, result_config=request)
    np.testing.assert_allclose(
        actual.result().get_unitary(),
        reference.result().get_unitary(),
        atol=1e-12,
        rtol=0,
    )


def _layered(n):
    program = fq.Program(n)
    for layer in range(2):
        for q in range(n):
            program.add(ops.RY(0.31 + q * 0.013 + layer * 0.071), q)
            program.add(ops.RZ(-0.27 + q * 0.011), q)
        for q in range(layer % 2, n - 1, 2):
            program.add(ops.CX, (q, q + 1))
    return program


def test_public_run_above_the_threshold_matches_numpy():
    # 2**22 amplitudes (64 MiB) exceed the default threshold: tiles engage.
    n = 22
    observable = fq.Observable([("Z" + "I" * (n - 2) + "Z", 0.8), ("X" * n, 0.3)])
    values = [
        fq.Estimator(Simulator("statevector", runtime=runtime))
        .run(_layered(n), observable, shots=0)
        .result()
        .get_expectation()
        for runtime in ("numpy", "numba")
    ]
    assert values[1] == pytest.approx(values[0], abs=1e-12)
