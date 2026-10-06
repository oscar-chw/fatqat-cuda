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


def test_reading_the_state_mid_run_sees_every_queued_gate():
    n = 10
    rng = np.random.default_rng(1)
    first = _random_steps(rng, n, 30)
    second = _random_steps(rng, n, 30)
    results = []
    for engine_cls in (_Tiled, _PerGate):
        engine = engine_cls()
        engine.initialize((2,) * n)
        for step in first:
            engine.apply(step)
        middle = engine.probabilities()
        for step in second:
            engine.apply(step)
        results.append((middle, engine.export_state()))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


def test_mixed_radix_systems_are_unchanged():
    # Tiles cover qubit statevectors only; a qutrit in the system must leave
    # results exactly as the per-gate kernels compute them.
    dims = (2,) * 6 + (3,)
    rng = np.random.default_rng(2)
    steps = _random_steps(rng, 6, 40)
    steps.append(ApplyMatrixStep(_unitary(rng, 3), (6,)))
    initial = rng.normal(size=192) + 1j * rng.normal(size=192)
    initial /= np.linalg.norm(initial)
    results = []
    for engine_cls in (_Tiled, _PerGate):
        engine = engine_cls()
        engine.initialize(dims, initial_state=initial)
        for step in steps:
            engine.apply(step)
        results.append(engine.export_state())
    np.testing.assert_array_equal(results[0], results[1])


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


def _controlled(matrix, value=1):
    """``matrix`` on the later targets where the first target equals ``value``."""
    size = len(matrix)
    out = np.eye(2 * size, dtype=np.complex128)
    out[value * size : (value + 1) * size, value * size : (value + 1) * size] = matrix
    return out


def _insular_steps(rng, n, depth):
    """Gates whose controls and diagonal targets need no tile bit."""
    steps = []
    for _ in range(depth):
        kind = int(rng.integers(10))
        width = 3 if kind in (0, 1, 2, 5) else 2
        targets = tuple(int(q) for q in rng.choice(n, width, replace=False))
        if kind >= 8:
            # Diagonals with controls and a moving target (CRZ, open-control
            # and doubly controlled phases): some entries exactly 1.
            phases = np.exp(1j * rng.normal(size=4))
            matrix = [
                np.diag([1, 1, phases[0], phases[1]]),
                np.diag([phases[0], phases[1], 1, 1]),
                np.diag([1] * 6 + list(phases[:2])),
                np.diag([1] * 4 + list(phases)),
                np.diag([1, phases[0]]),
                np.diag([phases[0], 1]),
            ][int(rng.integers(6))].astype(np.complex128)
            width = matrix.shape[0].bit_length() - 1
            targets = tuple(int(q) for q in rng.choice(n, width, replace=False))
            steps.append(ApplyMatrixStep(matrix, targets))
            continue
        if kind == 0:
            matrix = _controlled(_CX)  # Toffoli
        elif kind == 1:
            matrix = _controlled(_SWAP)  # Fredkin: two active targets
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


@pytest.mark.parametrize("n", [7, 9, 11])
@pytest.mark.parametrize("seed", range(5))
def test_controls_and_diagonals_anywhere_stay_bit_identical(n, seed):
    rng = np.random.default_rng(1000 * n + seed)
    initial = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    initial /= np.linalg.norm(initial)
    steps = _insular_steps(rng, n, 80)
    np.testing.assert_array_equal(
        _run(_Tiled, n, steps, initial), _run(_PerGate, n, steps, initial)
    )


def test_a_fourier_transform_needs_one_pass_per_tile_of_hadamards():
    # Controlled phases are diagonal and take no tile bit, so only the
    # Hadamards' qubits fill tiles: three free bits per pass here, against one
    # pass per gate if every target needed a bit.
    n = 12
    program = fq.Program(n)
    for q in range(n):
        program.add(ops.H, q)
        for k, r in enumerate(range(q + 1, n), start=2):
            program.add(ops.CPhase(2 * np.pi / 2**k), (r, q))
    plan, _ = Simulator("statevector", runtime="numpy")._lower_program(program)
    batches = []

    class Counted(_Tiled):
        def _apply_tile_batch(self, pending):
            batches.append(len(pending))
            super()._apply_tile_batch(pending)

    tiled = _run(Counted, n, plan, None)
    np.testing.assert_array_equal(tiled, _run(_PerGate, n, plan, None))
    assert len(batches) <= 4


def test_tile_forms_and_descriptors_place_controls_and_targets_exactly():
    from fatqat.simulator._engine.base import _tile_form

    crz = np.diag([1, 1, np.exp(-0.2j), np.exp(0.2j)])
    form = _tile_form(crz)
    assert form.diagonal and form.controls == ((0, 1),) and form.active == (1,)
    np.testing.assert_array_equal(form.matrix, crz.diagonal()[2:])
    open_control = np.diag([np.exp(-0.2j), np.exp(0.2j), 1, 1])
    assert _tile_form(open_control).controls == ((0, 0),)
    assert _tile_form(np.diag([1, 1, 1, -1])).controls == ((0, 1), (1, 1))
    assert _tile_form(np.diag([np.exp(0.1j), np.exp(0.2j)])).controls == ()

    engine = _Tiled()
    engine.initialize((2,) * 9)
    # Tile bits {0, 1, 2, 5, 6, 7}: subsystem 0 sits at tile position 0,
    # subsystem 8 is outside the tile.
    position = {bit: i for i, bit in enumerate((0, 1, 2, 5, 6, 7))}
    ccrz = ApplyMatrixStep(np.diag([1] * 6 + [np.exp(-0.2j), np.exp(0.2j)]), (0, 8, 5))
    gate = engine._tile_gate(ccrz, position)
    assert gate.fixed == ((0, 1), (3, 0))  # control at position 0; target at 3
    assert (gate.rest_mask, gate.rest_value) == (1 << 8, 1 << 8)
    assert gate.targets == (3,)
    outside = ApplyMatrixStep(np.diag([np.exp(0.1j), np.exp(0.2j)]), (8,))
    assert engine._tile_gate(outside, position).targets == (-1 - 8,)
    toffoli = ApplyMatrixStep(_controlled(_CX), (1, 2, 6))
    assert engine._tile_gate(toffoli, position).fixed == ((1, 1), (2, 1), (4, 0))
