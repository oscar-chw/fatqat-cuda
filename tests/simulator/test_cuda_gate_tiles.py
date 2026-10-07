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

    # pylint: disable-next=too-many-ancestors,abstract-method
    class TiledUnitary(CupyUnitaryEngine):
        _TILE_MIN_BYTES = 0

    # pylint: disable-next=too-many-ancestors,abstract-method
    class PerGateUnitary(CupyUnitaryEngine):
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


def test_only_exact_three_qubit_gates_join_a_cuda_tile():
    # Runs without a device: the rule is decided on the host. A three-qubit
    # gate that rounds would take the cuBLAS path alone, so it stays out.
    from fatqat.simulator._engine.cupy import CupySVEngine

    engine = CupySVEngine(device_id=0)
    toffoli = ApplyMatrixStep(_controlled(_CX), (0, 1, 2))
    ccz = ApplyMatrixStep(np.diag([1, 1, 1, 1, 1, 1, 1, -1]).astype(complex), (0, 1, 2))
    phases = ApplyMatrixStep(np.diag(np.exp(1j * np.arange(8.0))), (0, 1, 2))
    dense = ApplyMatrixStep(
        _controlled(_unitary(np.random.default_rng(0), 4)), (0, 1, 2)
    )
    assert engine._tile_form_of(toffoli) is not None
    assert engine._tile_form_of(ccz) is not None
    assert engine._tile_form_of(phases) is None
    assert engine._tile_form_of(dense) is None
    # Two-qubit gates tile whatever they hold.
    assert engine._tile_form_of(
        ApplyMatrixStep(_unitary(np.random.default_rng(1), 4), (0, 1))
    )


@pytest.mark.parametrize("seed", range(4))
def test_simplify_keeps_cuda_values_when_it_rewrites_only_exact_gates(engines, seed):
    # Unit gates (S and Y bring +-i) and rotations conjugated by +-1
    # permutations are the rewrites claimed to leave CUDA values unchanged;
    # H and T are left out, since cancelling them changes values on purpose.
    del engines
    rng = np.random.default_rng(seed)
    n = 14
    program = fq.Program(n)
    for q in range(n):
        program.add(ops.RY(float(rng.uniform(0, np.pi))), q)
    for _ in range(160):
        q, r, s = (int(v) for v in rng.choice(n, 3, replace=False))
        pick = int(rng.integers(9))
        if pick == 0:
            program.add(ops.CX, (q, r))
            program.add(ops.RZ(float(rng.normal())), r)
            program.add(ops.CX, (q, r))
        elif pick == 1:
            program.add(ops.CCX, (q, r, s))
        elif pick == 2:
            program.add(ops.CPhase(float(rng.normal())), (q, r))
        else:
            name = ["S", "Sdg", "Y", "X", "Z", "CZ"][pick - 3]
            program.add(getattr(ops, name), (q, r) if name == "CZ" else q)
    request = {"counts": False, "final_state": True}
    backend = Simulator("statevector", runtime="cuda")
    plain, simple = (
        backend.run(
            program, shots=0, result_config=request, simulation_config={"simplify": s}
        )
        .result()
        .get_statevector()
        for s in (False, True)
    )
    np.testing.assert_array_equal(simple, plain)
    plan, _ = Simulator("statevector", runtime="numpy")._lower_program(program)
    from fatqat._backends.simplify import simplify_plan

    assert len(simplify_plan(plan, (2,) * n, zero_start=True)) < len(plan)


def _run_with(engine_cls, products, n, steps, initial):
    engine = engine_cls(device_id=0)
    engine.gpu_products = products
    engine.initialize((2,) * n, initial_state=initial)
    for step in steps:
        engine.apply(step)
    return engine.export_state()


@pytest.mark.parametrize("seed", range(6))
def test_plain_products_keep_tiles_and_per_gate_kernels_identical(engines, seed):
    # gpu_products="plain": still spelled out, so the two paths still agree
    # bit for bit; it differs from the compensated products only by rounding.
    tiled, per_gate, _ = engines
    n = 14
    rng = np.random.default_rng(7000 + seed)
    initial = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    initial /= np.linalg.norm(initial)
    steps = _random_steps(rng, n, 80)
    plain = _run_with(per_gate, "plain", n, steps, initial)
    np.testing.assert_array_equal(_run_with(tiled, "plain", n, steps, initial), plain)
    compensated = _run_with(per_gate, "compensated", n, steps, initial)
    assert not np.array_equal(plain, compensated)  # the option takes effect
    np.testing.assert_allclose(plain, compensated, atol=1e-13, rtol=0)


def test_public_plain_products_match_the_cpu(engines):
    del engines
    rng = np.random.default_rng(3)
    program = fq.Program(14)
    for q in range(14):
        program.add(ops.RY(float(rng.uniform(0, 3))), q)
        program.add(ops.RZ(float(rng.uniform(0, 3))), q)
    for q in range(13):
        program.add(ops.CX, (q, q + 1))
    request = {"counts": False, "final_state": True}
    cuda = Simulator("statevector", runtime="cuda").run(
        program,
        shots=0,
        result_config=request,
        simulation_config={"gpu_products": "plain"},
    )
    numba = Simulator("statevector", runtime="numba").run(
        program, shots=0, result_config=request
    )
    assert cuda.result().metadata["simulation_config"]["gpu_products"] == "plain"
    compensated = Simulator("statevector", runtime="cuda").run(
        program, shots=0, result_config=request
    )
    # The setting takes effect through the public API, not only on an engine.
    assert not np.array_equal(
        cuda.result().get_statevector(), compensated.result().get_statevector()
    )
    np.testing.assert_allclose(
        cuda.result().get_statevector(),
        numba.result().get_statevector(),
        atol=1e-13,
        rtol=0,
    )
