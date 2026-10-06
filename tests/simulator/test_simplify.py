"""Opt-in circuit simplification: exact rewrites, barriers, and public routing.

Unit cases pin each rule on hand-built plans; a seeded random-circuit check
compares simplified and original plans on the reference NumPy engines,
including mixed radix, reordered targets, channels, measurements and resets;
public tests confirm every runtime accepts the option.
"""

from math import prod

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.simplify import simplify_plan
from fatqat._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    MeasurementStep,
    ResetStep,
)
from fatqat.errors import BackendValidationError
from fatqat.implementation import default_matrix_implementation_map
from fatqat.parameters import Parameter
from fatqat.simulator import Simulator
from fatqat.simulator._engine.np import NumpyDMEngine, NumpySVEngine

try:
    from fatqat.simulator._engine.nb import NumbaDMEngine, NumbaSVEngine

    numba_engines = (NumbaSVEngine, NumbaDMEngine)
except ImportError:  # numba is optional in some test lanes
    numba_engines = None

_ATOL = 1e-12
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_CX = np.eye(4, dtype=np.complex128)[[0, 1, 3, 2]]
_CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)
_Z = np.diag([1, -1]).astype(np.complex128)
_S = np.diag([1, 1j])


def _rz(theta):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _gate(matrix, *targets):
    return ApplyMatrixStep(np.asarray(matrix, dtype=np.complex128), targets)


def _random_unitary(rng, size):
    q, _ = np.linalg.qr(
        rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    )
    return q


def test_self_inverse_pair_is_removed_exactly():
    assert simplify_plan((_gate(_X, 0), _gate(_X, 0)), (2,)) == ()


def test_exact_inverse_pairs_are_removed():
    s = np.diag([1, 1j])
    assert simplify_plan((_gate(s, 0), _gate(s.conj().T, 0)), (2,)) == ()
    assert simplify_plan((_gate(_CX, 0, 1), _gate(_CX, 0, 1)), (2, 2)) == ()


def test_rounding_gates_are_never_merged_and_phase_is_kept():
    # H.H, rotations and an exactly representable but dense product (RY.Z)
    # would each change rounding; they are left exactly as written.
    for plan in (
        (_gate(_H, 0), _gate(_H, 0)),
        (_gate(_rx(0.3), 0), _gate(_rz(0.7), 0)),
        (_gate(_rz(0.3), 0), _gate(_Z, 0)),
        tuple(_gate(_rz(1.7e-15), 0) for _ in range(50)),
    ):
        assert simplify_plan(plan, (2,)) == plan
    kept = simplify_plan((_gate(-np.eye(2), 0), _gate(np.eye(2), 0)), (2,))
    assert len(kept) == 1
    np.testing.assert_array_equal(kept[0].matrix, -np.eye(2))


def test_three_cnots_become_one_swap():
    reversed_cx = np.eye(4, dtype=np.complex128)[[0, 3, 2, 1]]
    plan = (_gate(_CX, 0, 1), _gate(reversed_cx, 0, 1), _gate(_CX, 0, 1))
    simplified = simplify_plan(plan, (2, 2))
    assert len(simplified) == 1
    np.testing.assert_array_equal(
        simplified[0].matrix, np.eye(4, dtype=np.complex128)[[0, 2, 1, 3]]
    )


def test_s_s_is_z():
    simplified = simplify_plan((_gate(_S, 0), _gate(_S, 0)), (2,))
    assert len(simplified) == 1
    np.testing.assert_array_equal(simplified[0].matrix, _Z)


def test_runs_merge_past_gates_on_other_subsystems_in_order():
    plan = (_gate(_X, 0), _gate(_H, 1), _gate(_Z, 0))
    simplified = simplify_plan(plan, (2, 2))
    assert [s.target_indices for s in simplified] == [(0,), (1,)]
    np.testing.assert_array_equal(simplified[0].matrix, _Z @ _X)
    # A product loses the built-in identity; engines then inspect its content.
    assert simplified[0].kernel_key is None


def test_non_diagonal_gate_does_not_pass_an_overlapping_gate():
    plan = (_gate(_X, 1), _gate(_CZ, 0, 1), _gate(_X, 1))
    assert len(simplify_plan(plan, (2, 2))) == 3
    plan = (_gate(_S, 1), _gate(_CX, 0, 1), _gate(_S.conj().T, 1))
    assert len(simplify_plan(plan, (2, 2))) == 3  # CX is not diagonal


def test_diagonal_gates_merge_across_diagonal_gates():
    plan = (_gate(_S, 1), _gate(_CZ, 0, 1), _gate(_S.conj().T, 1))
    simplified = simplify_plan(plan, (2, 2))
    # The later Sdg moves back across the diagonal CZ and cancels S exactly.
    assert [s.target_indices for s in simplified] == [(0, 1)]


def test_reordered_targets_are_matched_by_axis_permutation():
    # CX with control 0, then CX written with targets (1, 0) but the same
    # meaning: their product is the identity only if the reorder is right.
    reversed_cx = np.eye(4, dtype=np.complex128)[[0, 3, 2, 1]]
    assert simplify_plan((_gate(_CX, 0, 1), _gate(reversed_cx, 1, 0)), (2, 2)) == ()
    assert len(simplify_plan((_gate(_CX, 0, 1), _gate(_CX, 1, 0)), (2, 2))) == 1


@pytest.mark.parametrize(
    "barrier",
    [
        MeasurementStep((0,), (0,)),
        ResetStep((0,)),
        ApplyChannelStep((np.eye(2, dtype=np.complex128),), (0,)),
    ],
    ids=["measurement", "reset", "channel"],
)
def test_normalizing_steps_block_every_subsystem(barrier):
    # Their renormalization sums over the whole state in memory order, so
    # even a cancellation on another subsystem would change the last bit.
    for q in (0, 1):
        plan = (_gate(_X, q), barrier, _gate(_X, q))
        assert simplify_plan(plan, (2, 2)) == plan


def test_conditioned_gates_block_only_their_subsystems():
    conditioned = ApplyMatrixStep(
        np.eye(2, dtype=np.complex128), (1,), condition=((0, 1),)
    )
    blocked = (_gate(_X, 1), conditioned, _gate(_X, 1))
    assert simplify_plan(blocked, (2, 2)) == blocked
    assert simplify_plan((_gate(_X, 0), conditioned, _gate(_X, 0)), (2, 2)) == (
        conditioned,
    )


@pytest.mark.parametrize("seed", range(20))
def test_mid_circuit_measurement_results_are_bit_identical(seed):
    # Regression: X.X cancelled across a measurement of another qubit used to
    # permute the state the measurement renormalized, changing the last bit.
    if numba_engines is None:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(seed)
    n = 6
    program = fq.Program(n, 1)
    for q in range(n):
        program.add(ops.RY(float(rng.uniform(0, np.pi))), q)
    for q in range(n - 1):
        program.add(ops.CX, (q, q + 1))
    program.add(ops.X, 0)
    program.measure((n - 1,), (0,))
    program.add(ops.X, 0)
    program.add(ops.RZ(0.3), n - 1)
    backend = Simulator("statevector", runtime="numba")
    options = {
        "shots": 1,
        "result_config": {"counts": True, "final_state": True},
    }
    plain = backend.run(program, simulation_config={"seed": seed}, **options).result()
    simple = backend.run(
        program, simulation_config={"seed": seed, "simplify": True}, **options
    ).result()
    assert plain.get_counts() == simple.get_counts()
    np.testing.assert_array_equal(simple.get_statevector(), plain.get_statevector())


def test_conditioned_gates_are_never_merged():
    conditioned = ApplyMatrixStep(_X, (0,), condition=((0, 1),))
    assert len(simplify_plan((_gate(_X, 0), conditioned), (2,))) == 2
    assert len(simplify_plan((conditioned, _gate(_X, 0)), (2,))) == 2


def test_an_unknown_step_is_a_global_barrier():
    sentinel = object()
    plan = (_gate(_X, 0), sentinel, _gate(_X, 0))
    assert simplify_plan(plan, (2,)) == plan


def _random_plan(rng, dims, depth, *, noisy):
    n = len(dims)
    plan = []
    for _ in range(depth):
        kind = rng.choice(
            ["dense", "diag", "perm", "inverse", "barrier", "rotation"],
            p=[0.15, 0.2, 0.2, 0.2, 0.1, 0.15],
        )
        width = int(rng.integers(1, 3))
        targets = tuple(int(q) for q in rng.choice(n, width, replace=False))
        size = prod(dims[q] for q in targets)
        if kind == "dense":
            plan.append(_gate(_random_unitary(rng, size), *targets))
        elif kind == "diag":
            # Phases from {1, i, -1, -i}: products stay exactly representable.
            phases = 1j ** rng.integers(4, size=size)
            plan.append(_gate(np.diag(phases), *targets))
        elif kind == "perm":
            phases = 1j ** rng.integers(4, size=size)
            plan.append(
                _gate(np.diag(phases) @ np.eye(size)[rng.permutation(size)], *targets)
            )
        elif kind == "rotation":
            # Rounding diagonal gates: obstacles that must never be crossed.
            plan.append(_gate(np.diag(np.exp(1j * rng.normal(size=size))), *targets))
        elif kind == "inverse" and plan and isinstance(plan[-1], ApplyMatrixStep):
            last = plan[-1]
            plan.append(ApplyMatrixStep(last.matrix.conj().T, last.target_indices))
        elif kind == "barrier" and noisy:
            q = int(rng.integers(n))
            p = 0.2
            kraus = (
                np.sqrt(1 - p) * np.eye(dims[q], dtype=np.complex128),
                np.sqrt(p) * _random_unitary(rng, dims[q]),
            )
            plan.append(ApplyChannelStep(kraus, (q,)))
    return tuple(plan)


def _evolve(engine, dims, plan, initial):
    engine.initialize(dims, initial_state=initial)
    rng = np.random.default_rng(0)
    for step in plan:
        if isinstance(step, ApplyChannelStep):
            engine.apply_channel(step, rng)
        else:
            engine.apply(step)
    return engine.export_state()


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize(
    "dims", [(2, 2, 2, 2), (3, 2, 3)], ids=["qubits", "mixed_radix"]
)
def test_random_plans_give_identical_values_and_are_never_longer(seed, dims):
    rng = np.random.default_rng(seed)
    size = prod(dims)
    ket = rng.normal(size=size) + 1j * rng.normal(size=size)
    ket /= np.linalg.norm(ket)
    plan = _random_plan(rng, dims, 30, noisy=False)
    simplified = simplify_plan(plan, dims)
    assert len(simplified) <= len(plan)
    noisy = _random_plan(rng, dims, 30, noisy=True)
    rho = np.outer(ket, ket.conj())
    cases = [
        (NumpySVEngine, simplified, plan, ket),
        (NumpyDMEngine, simplify_plan(noisy, dims), noisy, rho),
    ]
    if numba_engines is not None:
        cases += [
            (numba_engines[0], simplified, plan, ket),
            (numba_engines[1], simplify_plan(noisy, dims), noisy, rho),
        ]
    for engine, short, full, initial in cases:
        actual = _evolve(engine(), dims, short, initial)
        expected = _evolve(engine(), dims, full, initial)
        if engine.__module__.endswith(".np") and set(dims) != {2}:
            # The NumPy engine contracts through BLAS, and some BLAS builds
            # (OpenBLAS on x86) round an element through a fused or unfused
            # path depending on its position. Unit gates never round, but
            # moving amplitudes can move an unrelated gate's rounding; the
            # change is unbiased and at the last-bit level.
            np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
        else:
            # Exact equality, not a tolerance: unit gates never round.
            np.testing.assert_array_equal(actual, expected)


def test_random_plans_actually_simplify():
    # Guards against a pass that is equivalent only because it does nothing.
    removed = 0
    for seed in range(40):
        plan = _random_plan(np.random.default_rng(seed), (2, 2, 2, 2), 30, noisy=False)
        removed += len(plan) - len(simplify_plan(plan, (2, 2, 2, 2)))
    assert removed > 150


def _layered(n):
    program = fq.Program(n)
    for layer in range(2):
        for q in range(n):
            program.add(ops.RY(0.31 + q * 0.013 + layer * 0.071), q)
            program.add(ops.RZ(-0.27 + q * 0.011), q)
        for q in range(layer % 2, n - 1, 2):
            program.add(ops.CX, (q, q + 1))
    program.add(ops.X, 0)
    program.add(ops.X, 0)
    return program


@pytest.mark.parametrize(
    "method", ["statevector", "density_matrix", "unitary", "superop"]
)
@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_public_runs_match_with_and_without_simplify(method, runtime):
    backend = Simulator(method, runtime=runtime)
    request = {"counts": False, "final_state": True}
    program = _layered(3)
    plain = backend.run(program, shots=0, result_config=request).result()
    simple = backend.run(
        program, shots=0, result_config=request, simulation_config={"simplify": True}
    ).result()
    # Only X.X cancels here (exactly); rotations are not merged, so the
    # result is bit-identical: simplification never costs accuracy.
    np.testing.assert_array_equal(
        getattr(simple, f"get_{method}")(), getattr(plain, f"get_{method}")()
    )
    assert simple.metadata["simulation_config"]["simplify"] is True


def test_public_mixed_radix_reordered_inverse_cancels():
    # A permutation on (qubit, qutrit), then its inverse written on
    # (qutrit, qubit): the pair cancels only if the reorder keeps each
    # target's own dimension.
    unitary = np.eye(6, dtype=np.complex128)[np.random.default_rng(3).permutation(6)]
    inverse = unitary.conj().T.reshape(2, 3, 2, 3).transpose(1, 0, 3, 2).reshape(6, 6)

    class Pair(ops.Operation):
        name = "SimplifyPair"
        num_subsystems = 2

    class PairInverse(ops.Operation):
        name = "SimplifyPairInverse"
        num_subsystems = 2

    implementations = default_matrix_implementation_map()
    implementations.add(Pair, unitary)
    implementations.add(PairInverse, inverse)
    registers = [fq.QuantumRegister(1, dim=d) for d in (3, 2)]
    qutrit, qubit = registers[0][0], registers[1][0]
    program = fq.Program(registers)
    program.add(ops.Shift(1), qutrit)
    program.add(Pair(), (qubit, qutrit))
    program.add(PairInverse(), (qutrit, qubit))
    program.add(ops.Clock(2), qutrit)
    backend = Simulator("unitary", runtime="numpy", implementation_map=implementations)
    request = {"counts": False, "final_state": True}
    plain = backend.run(program, shots=0, result_config=request).result().get_unitary()
    simple = (
        backend.run(
            program,
            shots=0,
            result_config=request,
            simulation_config={"simplify": True},
        )
        .result()
        .get_unitary()
    )
    np.testing.assert_array_equal(simple, plain)
    plan, _ = backend._lower_program(program)
    dims = backend._allocate_engine_indices(
        program, backend._resolve_resource_layout(program)
    ).system_dims
    # The pair cancels exactly. Shift and Clock then meet, but Clock's phases
    # (cube roots of unity) round, so they stay two gates.
    assert len(plan) == 4
    assert [s.kernel_key for s in simplify_plan(plan, dims)] == [
        plan[0].kernel_key,
        plan[3].kernel_key,
    ]


def test_estimator_accepts_simplify():
    observable = fq.Observable([("ZZZ", 0.8), ("XXX", 0.3)])
    estimator = fq.Estimator(Simulator("statevector", runtime="numpy"))
    plain = estimator.run(_layered(3), observable, shots=0).result().get_expectation()
    simple = (
        estimator.run(
            _layered(3), observable, shots=0, simulation_config={"simplify": True}
        )
        .result()
        .get_expectation()
    )
    assert simple == pytest.approx(plain, abs=_ATOL)


def test_sweeps_reject_simplify_instead_of_ignoring_it():
    theta = Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RY(theta), 0)
    with pytest.raises(
        BackendValidationError, match="not supported by parameter sweeps"
    ):
        Simulator("statevector", runtime="numpy").run_sweep(
            program, {theta: [0.1, 0.2]}, shots=0, simulation_config={"simplify": True}
        ).result()


def test_cuda_accepts_simplify():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device available")
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device discovery unavailable: {error}")
    request = {"counts": False, "final_state": True}
    program = _layered(4)
    cpu = Simulator("statevector", runtime="numpy").run(
        program, shots=0, result_config=request
    )
    gpu = Simulator("statevector", runtime="cuda").run(
        program, shots=0, result_config=request, simulation_config={"simplify": True}
    )
    np.testing.assert_allclose(
        gpu.result().get_statevector(),
        cpu.result().get_statevector(),
        atol=_ATOL,
        rtol=0,
    )
