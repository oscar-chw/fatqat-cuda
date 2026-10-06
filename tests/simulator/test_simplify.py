"""Opt-in circuit simplification: exact rewrites, barriers, and public routing.

Unit cases pin each rule on hand-built plans; a seeded random-circuit check
compares simplified and original plans on the reference NumPy engines,
including mixed radix, reordered targets, channels, measurements and resets;
public tests confirm every runtime accepts the option.
"""

from fractions import Fraction
from math import isqrt, prod

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.simplify import (
    _Block,
    _matmul,
    _Merger,
    _reorder,
    simplify_plan,
)
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
    # A custom H (no declared identity) and rotations round; merging any two
    # would round their product, so they are left exactly as written.
    for plan in (
        (_gate(_H, 0), _gate(_H, 0)),
        (_gate(_rx(0.3), 0), _gate(_rz(0.7), 0)),
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


def test_one_qubit_unit_gates_are_absorbed_into_a_two_qubit_block():
    # X on the target of CZ does not commute with it, but all three are
    # unit gates, so their exact product replaces them.
    plan = (_gate(_X, 1), _gate(_CZ, 0, 1), _gate(_X, 1))
    simplified = simplify_plan(plan, (2, 2))
    assert len(simplified) == 1
    xi = np.kron(np.eye(2), _X)
    np.testing.assert_array_equal(simplified[0].matrix, xi @ _CZ @ xi)


def test_rounding_gates_are_never_crossed_by_a_non_commuting_gate():
    dense = _gate(_H, 1)  # no declared identity: an ordinary rounding gate
    plan = (_gate(_X, 1), dense, _gate(_X, 1))
    assert simplify_plan(plan, (2, 2)) == plan


@pytest.mark.parametrize(
    ("plan", "dims", "expected"),
    [
        ((_gate(_rz(0.3), 0), _gate(_Z, 0)), (2,), _Z @ _rz(0.3)),
        ((_gate(_X, 0), _gate(_rz(0.3), 0), _gate(_X, 0)), (2,), _X @ _rz(0.3) @ _X),
        (
            (_gate(_CX, 0, 1), _gate(_rz(0.3), 1), _gate(_CX, 0, 1)),
            (2, 2),
            _CX @ np.kron(np.eye(2), _rz(0.3)) @ _CX,
        ),
    ],
    ids=["RZ.Z", "X.RZ.X", "CX.RZ.CX"],
)
def test_a_rotation_conjugated_by_sign_permutations_is_one_exact_gate(
    plan, dims, expected
):
    # Entries of +-1 only move and negate the rotation's entries: the product
    # is exact, and applying it multiplies each amplitude exactly as before.
    (step,) = simplify_plan(plan, dims)
    targets = tuple(range(len(dims)))
    np.testing.assert_array_equal(
        _reorder(step.matrix, step.target_indices, targets, dims), expected
    )


def test_a_rotation_does_not_move_past_an_i_phase_to_merge():
    # RZ commutes with CS, and CZ.RZ would be exact, but reaching CZ means RZ
    # moving past CS: its i swaps real and imaginary parts, so with fused
    # multiply-adds (CUDA) RZ's products would pair, and round, differently.
    # CS cannot merge with CZ (their subsystems only overlap), so RZ would
    # have to pass it to reach CZ.
    cs = np.diag([1, 1, 1, 1j])
    plan = (_gate(_CZ, 0, 1), _gate(cs, 1, 2), _gate(_rz(0.3), 1))
    assert simplify_plan(plan, (2, 2, 2)) == plan


def test_a_rotation_is_not_merged_with_an_i_phase():
    # S.RZ is exact too, but S swaps real and imaginary parts, so with fused
    # multiply-adds the rotation's products would pair differently.
    plan = (_gate(_rz(0.3), 0), _gate(_S, 0))
    assert simplify_plan(plan, (2,)) == plan


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


def _lowered(build, n):
    """The engine plan and dims of a public program built by ``build``."""
    program = fq.Program(n)
    build(program)
    backend = Simulator("statevector", runtime="numpy")
    plan, _ = backend._lower_program(program)
    return plan, (2,) * n


def _sqrt2_fraction(numerator, power):
    """``numerator * sqrt(2) / 2**power`` to 300 bits, as a Fraction."""
    return Fraction(numerator * isqrt(2 << 600), 1 << (300 + power))


@pytest.mark.parametrize(
    ("build", "n", "expected", "targets"),
    [
        (lambda p: [p.add(ops.H, 0), p.add(ops.X, 0), p.add(ops.H, 0)], 1, _Z, (0,)),
        (lambda p: [p.add(ops.H, 0), p.add(ops.Z, 0), p.add(ops.H, 0)], 1, _X, (0,)),
        (lambda p: [p.add(ops.T, 0), p.add(ops.T, 0)], 1, _S, (0,)),
        (lambda p: [p.add(ops.S, 0), p.add(ops.S, 0)], 1, _Z, (0,)),
        (
            lambda p: [
                p.add(ops.H, 0), p.add(ops.H, 1), p.add(ops.CX, (0, 1)),
                p.add(ops.H, 0), p.add(ops.H, 1),
            ],  # fmt: skip
            2,
            _CX,
            "reversed",
        ),
        (
            lambda p: [
                p.add(ops.CX, (0, 1)),
                p.add(ops.CX, (0, 2)),
                p.add(ops.CX, (0, 1)),
            ],
            3,
            _CX,
            "control 0, target 2",
        ),
    ],
    ids=["HXH=Z", "HZH=X", "TT=S", "SS=Z", "HH.CX.HH=reversed CX", "CX commutation"],
)
def test_lecture_identities_hold_exactly(build, n, expected, targets):
    plan, dims = _lowered(build, n)
    simplified = simplify_plan(plan, dims)
    assert len(simplified) == 1
    (step,) = simplified
    # Engine indices are reversed: public qubit q is engine subsystem n-1-q.
    if targets == "reversed":
        assert set(step.target_indices) == {0, 1}
        # Control is public qubit 1 (engine 0) and target public 0 (engine 1).
        matrix = _reorder(step.matrix, step.target_indices, (0, 1), dims)
    elif targets == "control 0, target 2":
        matrix = _reorder(step.matrix, step.target_indices, (2, 0), dims)
    else:
        assert step.target_indices == targets
        matrix = step.matrix
    np.testing.assert_array_equal(matrix, expected)


def test_h_h_cancels_and_a_rounding_run_becomes_its_correctly_rounded_product():
    plan, dims = _lowered(lambda p: [p.add(ops.H, 0), p.add(ops.H, 0)], 1)
    assert simplify_plan(plan, dims) == ()
    plan, dims = _lowered(
        lambda p: [p.add(ops.H, 0), p.add(ops.T, 0), p.add(ops.H, 0)], 1
    )
    (step,) = simplify_plan(plan, dims)
    # H T H = [[1 + w, 1 - w], [1 - w, 1 + w]] / 2 with w = (1 + i) / sqrt(2).
    plus = complex(
        float(Fraction(1, 2) + _sqrt2_fraction(1, 2)),
        float(_sqrt2_fraction(1, 2)),
    )
    minus = complex(
        float(Fraction(1, 2) - _sqrt2_fraction(1, 2)),
        -float(_sqrt2_fraction(1, 2)),
    )
    np.testing.assert_array_equal(step.matrix, [[plus, minus], [minus, plus]])


def test_a_merge_across_a_rounding_gate_must_remove_rounding():
    rz = _gate(_rz(0.3), 0)
    # S and Sdg would cancel across RZ, but RZ would then round in another
    # order with nothing gained, so the unit gates stay where they are.
    plan = (_gate(_S, 0), rz, _gate(_S.conj().T, 0))
    assert simplify_plan(plan, (2,)) == plan
    # T and Tdg round; cancelling them across RZ removes two roundings.
    t, tdg = _lowered(lambda p: [p.add(ops.T, 0), p.add(ops.Tdg, 0)], 1)[0]
    assert simplify_plan((t, rz, tdg), (2,)) == (rz,)


def test_a_block_keeps_its_parts_when_the_product_rounds_more():
    # Two gates of one rounding each, whose product would need four: the
    # block must emit the parts, not the denser product.
    t = _Block((0,), 0, (1, 1), leaf=object())
    block = _Block((0, 1), 1, (4, 1), children=(t, t))
    assert not block.use_product
    assert block.cost == (2, 2)
    assert _Block((0, 1), 1, (1, 1), children=(t, t)).use_product


def test_commutation_is_decided_exactly_not_to_a_tolerance():
    # x = (sqrt(2) - 1)**30 is about 3e-12: [[1, x], [0, 1]] and Z fail to
    # commute by 2x, below any float tolerance a quick check could use.
    merger = _Merger((2,))
    root = np.array([-1, 1, 0, -1])  # sqrt(2) - 1 = w - w**3 - 1
    x = np.array([1, 0, 0, 0])
    for _ in range(30):
        x = _matmul(x.reshape(1, 1, 4), root.reshape(1, 1, 4)).reshape(4)
    shear = np.zeros((2, 2, 4), dtype=np.int64)
    shear[0, 0, 0] = shear[1, 1, 0] = 1
    shear[0, 1] = x
    z = np.zeros((2, 2, 4), dtype=np.int64)
    z[0, 0, 0], z[1, 1, 0] = 1, -1
    a = merger._block((0,), merger._intern(shear, 0))
    b = merger._block((0,), merger._intern(z, 0))
    assert not merger._commutes(a, b)
    assert merger._commutes(b, merger._block((0,), merger._intern(z, 0)))


def test_unknown_steps_forget_every_known_input():
    sentinel = object()
    cx = _gate(_CX, 0, 1)
    simplified = simplify_plan((sentinel, cx), (2, 2), zero_start=True)
    assert simplified == (sentinel, cx)


def test_real_valued_gate_matrices_are_accepted():
    # Regression: a float64 rule crashed the unit-gate check.
    x = ApplyMatrixStep(np.array([[0.0, 1.0], [1.0, 0.0]]), (0,))
    assert x.matrix.dtype == np.float64
    assert simplify_plan((x, x), (2,)) == ()


def test_known_inputs_specialise_gates_only_from_the_zero_start():
    cx = _gate(_CX, 0, 1)
    # Control still |0>: the CX acts as the identity.
    assert simplify_plan((cx,), (2, 2), zero_start=True) == ()
    assert simplify_plan((cx,), (2, 2)) == (cx,)
    # Control known |1>, target unknown: the CX is an X on the target.
    h = _gate(_H, 1)
    plan = (_gate(_X, 0), h, cx)
    simplified = simplify_plan(plan, (2, 2), zero_start=True)
    assert simplified[:2] == plan[:2]
    assert simplified[2].target_indices == (1,)
    np.testing.assert_array_equal(simplified[2].matrix, _X)
    # A phase on a known state is kept; a phase of 1 is dropped.
    (kept,) = simplify_plan((_gate(_X, 0), _gate(_Z, 0)), (2,), zero_start=True)
    np.testing.assert_array_equal(kept.matrix, _Z @ _X)
    assert simplify_plan((_gate(_S, 0),), (2,), zero_start=True) == ()


def test_known_inputs_are_forgotten_at_every_non_gate_step():
    cx = _gate(_CX, 0, 1)
    channel = ApplyChannelStep((np.eye(2, dtype=np.complex128),), (0,))
    measure = MeasurementStep((0,), (0,))
    for step in (channel, measure, ResetStep((0,))):
        assert simplify_plan((step, cx), (2, 2), zero_start=True) == (step, cx)
    # A gate conditioned on a classical bit makes its own targets unknown.
    conditioned = ApplyMatrixStep(_X, (0,), condition=((0, 1),))
    assert simplify_plan((conditioned, cx), (2, 2), zero_start=True) == (
        conditioned,
        cx,
    )


def _clifford_t_program(rng, n, depth):
    program = fq.Program(n)
    names = ["H", "X", "Z", "S", "Sdg", "T", "Tdg", "SX"]
    for _ in range(depth):
        if rng.random() < 0.3 and n > 1:
            a, b = (int(q) for q in rng.choice(n, 2, replace=False))
            program.add(getattr(ops, ["CX", "CZ", "Swap"][rng.integers(3)]), (a, b))
        elif rng.random() < 0.15:
            program.add(ops.RZ(float(rng.normal())), int(rng.integers(n)))
        else:
            program.add(
                getattr(ops, names[rng.integers(len(names))]), int(rng.integers(n))
            )
    return program


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("method", ["statevector", "unitary"])
def test_clifford_t_circuits_stay_equivalent_and_get_shorter(seed, method):
    program = _clifford_t_program(np.random.default_rng(seed), 4, 60)
    backend = Simulator(method, runtime="numpy")
    request = {"counts": False, "final_state": True}
    plain = backend.run(program, shots=0, result_config=request).result()
    simple = backend.run(
        program, shots=0, result_config=request, simulation_config={"simplify": True}
    ).result()
    np.testing.assert_allclose(
        getattr(simple, f"get_{method}")(),
        getattr(plain, f"get_{method}")(),
        atol=1e-13,
        rtol=0,
    )
    plan, _ = backend._lower_program(program)
    assert len(simplify_plan(plan, (2,) * 4)) < len(plan)


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
    # X.X cancels and CX.RZ.CX-style products are exact: Numba gives the
    # same values. NumPy contracts through BLAS, whose last bit can depend on
    # an element's position (OpenBLAS on x86), so it is held to that bit.
    actual, expected = (
        getattr(result, f"get_{method}")() for result in (simple, plain)
    )
    if runtime == "numpy":
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
    else:
        np.testing.assert_array_equal(actual, expected)
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
    # The pair cancels exactly. Shift and Clock then meet: Clock's phases
    # (cube roots of unity) round, but Shift only moves them, so their
    # product is one exact gate.
    assert len(plan) == 4
    (merged,) = simplify_plan(plan, dims)
    np.testing.assert_array_equal(merged.matrix, plan[3].matrix @ plan[0].matrix)


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


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_a_product_never_takes_a_conditioned_gates_place(runtime):
    # Regression: H Z H equals the matrix of a later X conditioned on a bit
    # that is never 1; the product reused that step, condition and all, so
    # the X never ran.
    program = fq.Program(2, 2)
    program.add(ops.H, 0)
    program.add(ops.Z, 0)
    program.add(ops.H, 0)
    program.measure((1,), (0,))
    program.add(ops.X, 0, condition=(0, 1))
    program.measure((0,), (1,))
    backend = Simulator("statevector", runtime=runtime)
    counts = [
        backend.run(
            program, shots=20, simulation_config={"seed": 1, "simplify": simplify}
        )
        .result()
        .get_counts()
        for simplify in (False, True)
    ]
    assert counts[1] == counts[0]


def test_rounding_gates_never_change_width_class():
    # CUDA applies rounding gates on up to two qubits with its own kernels and
    # wider ones through cuBLAS: an exact rewrite must not move one across.
    ccx = _gate(np.eye(8, dtype=np.complex128)[[0, 1, 2, 3, 4, 5, 7, 6]], 0, 1, 2)
    rz = _gate(_rz(0.3), 2)
    assert simplify_plan((ccx, rz, ccx), (2, 2, 2)) == (ccx, rz, ccx)
    # A doubly controlled rotation whose first control is known |1>: dropping
    # would be exact, but shrinking it to a singly controlled one is not done.
    ccrz = _gate(np.diag(np.r_[np.ones(6), np.diag(_rz(0.3))]), 0, 1, 2)
    plan = (_gate(_X, 0), _gate(_H, 1), ccrz)
    assert simplify_plan(plan, (2, 2, 2), zero_start=True)[-1] is ccrz


def _exact_matmul(a, b):
    """The Z[w] product of two coefficient arrays in Python integers."""
    from fatqat._backends.simplify import (
        _SHIFT,
        _SIGN,
    )  # pylint: disable=import-outside-toplevel

    rows, inner, columns = a.shape[0], a.shape[1], b.shape[1]
    out = np.zeros((rows, columns, 4), dtype=object)
    for r in range(rows):
        for c in range(columns):
            for m in range(inner):
                for i in range(4):
                    for j in range(4):
                        out[r, c, j] += (
                            int(a[r, m, i])
                            * int(_SIGN[i, j])
                            * int(b[m, c, _SHIFT[i, j]])
                        )
    return out


@pytest.mark.parametrize("bits", [20, 30, 40, 50, 60])
def test_exact_products_never_overflow_silently(bits):
    # Entries near the merge limit can multiply past int64. Every product
    # returned must equal the exact integer one; one that cannot be computed
    # exactly is refused (None), never wrapped.

    rng = np.random.default_rng(bits)
    high = 1 << bits
    for size in (2, 4, 16):
        a = rng.integers(-high, high, size=(size, size, 4), dtype=np.int64)
        b = rng.integers(-high, high, size=(size, size, 4), dtype=np.int64)
        got = _matmul(a, b)
        if got is None:
            assert bits > 20, "small entries must never be refused"
        else:
            assert np.array_equal(got.astype(object), _exact_matmul(a, b))


def test_deep_clifford_t_runs_simplify_exactly():
    # Alternating H and T grow the numerators fast; the merge must stop
    # before they overflow, and the result stays the circuit's own.
    program = fq.Program(2)
    for _ in range(60):
        for q in range(2):
            program.add(ops.H, q)
            program.add(ops.T, q)
    program.add(ops.CX, (0, 1))
    request = {"counts": False, "final_state": True}
    plain, simple = (
        Simulator("statevector", runtime="numpy")
        .run(program, shots=0, result_config=request, simulation_config={"simplify": s})
        .result()
        .get_statevector()
        for s in (False, True)
    )
    np.testing.assert_allclose(simple, plain, rtol=0, atol=1e-12)


def test_blocks_that_keep_their_parts_emit_them_in_order(monkeypatch):
    # A merged block whose product would round more emits its original gates
    # instead; emitted in the wrong order, a circuit comes out wrong (0.65 in
    # an amplitude on one such case). Random Clifford+T with Toffolis reach
    # that path often; each must match its plain run.
    from fatqat._backends import simplify as module

    kept_parts = []
    shipped = module._Merger._emit

    def emit(self):
        stack = [e for e in self.out if isinstance(e, module._Block)]
        while stack:  # every block, nested ones included, as _emit visits them
            block = stack.pop()
            if block.children:
                kept_parts.append(not block.use_product)
                stack.extend(block.children)
        return shipped(self)

    monkeypatch.setattr(module._Merger, "_emit", emit)
    names = [
        ops.H,
        ops.T,
        ops.Tdg,
        ops.SX,
        ops.S,
        ops.X,
        ops.CX,
        ops.CZ,
        ops.Swap,
        ops.CCX,
    ]
    widths = {ops.CX: 2, ops.CZ: 2, ops.Swap: 2, ops.CCX: 3}
    request = {"counts": False, "final_state": True}
    for seed in range(400):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 5))
        program = fq.Program(n)
        for _ in range(int(rng.integers(2, 14))):
            gate = names[int(rng.integers(len(names)))]
            width = widths.get(gate, 1)
            if width > n:
                continue
            targets = tuple(int(q) for q in rng.choice(n, width, replace=False))
            program.add(gate, targets if width > 1 else targets[0])
        plain, simple = (
            Simulator("statevector", runtime="numba")
            .run(
                program,
                shots=0,
                result_config=request,
                simulation_config={"simplify": s},
            )
            .result()
            .get_statevector()
            for s in (False, True)
        )
        np.testing.assert_allclose(
            simple, plain, rtol=0, atol=1e-12, err_msg=f"seed {seed}"
        )
    assert sum(kept_parts) >= 10, "the kept-parts path was hardly reached"


def test_rounding_cost_is_the_densest_row():
    # The rounding cost of a gate is the largest number of products summed
    # into one amplitude: a controlled H rounds like H (2), not like the
    # identity rows it also has (1); unit gates cost nothing.
    from fatqat._backends.simplify import _from_unit, _rounding, _TABLE
    from fatqat._backends.steps import BuiltinKernelKey

    h, h_k = _TABLE[BuiltinKernelKey.H]
    controlled_h = np.zeros((4, 4, 4), dtype=np.int64)
    controlled_h[0, 0, 0] = controlled_h[1, 1, 0] = 1  # sqrt(2) * identity rows
    controlled_h[0, 0, :] = 0
    controlled_h[0, 0, 1], controlled_h[0, 0, 3] = 1, -1  # w - w**3 = sqrt(2)
    controlled_h[1, 1, :] = controlled_h[0, 0, :]
    controlled_h[2:, 2:, :] = h
    assert _rounding(h, h_k) == 2
    assert _rounding(controlled_h, 1) == 2
    assert _rounding(_from_unit(np.eye(4, dtype=np.complex128)[[1, 0, 2, 3]]), 0) == 0
    t, t_k = _TABLE[BuiltinKernelKey.T]
    assert _rounding(t, t_k) == 1
