"""Predeclared 4096-gate round-off comparisons on one or two qubits.

The independent 60-digit basis-sum oracle and exact binary64 embedding come
from test_cupy_accuracy. Long evolution can expose error differences hidden by
the eight-epsilon comparison floor in short circuits. These cases retain the
same 1e-12 absolute amplitude bound; they neither normalize states nor
replace stored gates with idealized ones.

Each case is one deterministic sample, and at this depth the CPU engines
themselves disagree by more than eight epsilons (on one GPU run: NumPy
2.72e-15 against Numba 6.78e-15 over 4096 gates and their adjoints), so no
third arithmetic can be held within eight epsilons of each of them. The GPU
is held to the range the CPU engines span: no worse than the less accurate
of them, plus eight epsilons. Whether it is more or less accurate on
average is measured by perf/precision.py, paired over many circuits.
"""

import json

import numpy as np
import pytest

from fatqat._backends.steps import ApplyMatrixStep
from tests.simulator.test_cupy_accuracy import (
    _ATOL,
    _EQUIVALENCE_FLOOR,
    _errors,
    _oracle_evolve,
)

_DEPTH = 4096
_CYCLE_TARGETS = ((0,), (1, 0), (1,), (0, 1), (0,), (1,), (0, 1), (1, 0))


@pytest.fixture(scope="module", name="depth_dependencies")
def _depth_dependencies():
    mpmath = pytest.importorskip("mpmath")
    cupy = pytest.importorskip("cupy")
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device discovery unavailable: {error}")
    if count == 0:
        pytest.skip("No CUDA device available")
    return mpmath


def _repeated_rotation():
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    pauli_axis = np.array(
        [[axis[2], axis[0] - 1j * axis[1]], [axis[0] + 1j * axis[1], -axis[2]]],
        dtype=np.complex128,
    )
    angle = 0.731
    matrix = (
        np.cos(angle / 2) * np.eye(2, dtype=np.complex128)
        - 1j * np.sin(angle / 2) * pauli_axis
    )
    initial = np.array([1 + 2j, -0.3 + 0.7j], dtype=np.complex128)
    initial /= np.linalg.norm(initial)
    step = ApplyMatrixStep(matrix, (0,))
    return (2,), initial, (step,) * _DEPTH


def _random_cycle(seed):
    rng = np.random.default_rng(seed)
    initial = np.array(
        rng.normal(size=4) + 1j * rng.normal(size=4), dtype=np.complex128
    )
    initial /= np.linalg.norm(initial)
    steps = []
    for targets in _CYCLE_TARGETS:
        size = 2 ** len(targets)
        matrix, _ = np.linalg.qr(
            rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        )
        steps.append(ApplyMatrixStep(np.array(matrix, dtype=np.complex128), targets))
    return initial, tuple(steps)


def _deep_case(case):
    if case == "repeated_rotation":
        return _repeated_rotation()
    if case == "random_cycle":
        initial, cycle = _random_cycle(5280)
        return (2, 2), initial, cycle * (_DEPTH // len(cycle))
    assert case == "forward_adjoint"
    initial, cycle = _random_cycle(73)
    forward = cycle * (_DEPTH // (2 * len(cycle)))
    inverse_cycle = tuple(
        ApplyMatrixStep(step.matrix.conj().T, step.target_indices)
        for step in reversed(cycle)
    )
    # Rounded matrices are only approximately unitary. The reference evolves
    # the same stored adjoints; returning exactly to the input is not assumed.
    backward = inverse_cycle * (_DEPTH // (2 * len(inverse_cycle)))
    return (2, 2), initial, forward + backward


@pytest.mark.parametrize(
    "case", ["repeated_rotation", "random_cycle", "forward_adjoint"]
)
def test_deep_evolution_against_stored_coefficient_oracle(
    depth_dependencies, record_property, case
):
    from fatqat.simulator._engine.cupy import CupySVEngine
    from fatqat.simulator._engine.nb import NumbaSVEngine
    from fatqat.simulator._engine.np import NumpySVEngine

    dims, initial, steps = _deep_case(case)
    assert len(steps) == _DEPTH
    engines = {
        "cupy": CupySVEngine(device_id=0),
        "numpy": NumpySVEngine(),
        "numba": NumbaSVEngine(),
    }
    states = {}
    for name, engine in engines.items():
        engine.initialize(dims, initial_state=initial)
        for step in steps:
            engine.apply(step)
        states[name] = engine.export_state()

    mp = depth_dependencies.mp
    with mp.workdps(60):
        reference = _oracle_evolve(dims, initial, steps, mp)
        errors = {name: _errors(state, reference, mp) for name, state in states.items()}
        reference_norm = float(mp.sqrt(mp.fsum(abs(value) ** 2 for value in reference)))

    gpu_error = errors["cupy"]["linf"]
    comparisons = {}
    for cpu in ("numpy", "numba"):
        cpu_error = errors[cpu]["linf"]
        comparisons[cpu] = {
            "gpu_to_cpu_linf_ratio": gpu_error / cpu_error if cpu_error else None,
            "gpu_minus_cpu_linf": gpu_error - cpu_error,
            "cpu_linf_in_floor_units": cpu_error / _EQUIVALENCE_FLOOR,
            "ratio_interpretation": (
                "above_equivalence_floor"
                if cpu_error > _EQUIVALENCE_FLOOR
                else "unresolved_at_equivalence_floor"
            ),
            # Informational: the CPU engines themselves differ by more
            # than the floor at this depth, so this is not the check below.
            "gpu_within_this_cpu_plus_floor_informational": bool(
                gpu_error <= cpu_error + _EQUIVALENCE_FLOOR
            ),
        }
    worse_cpu = max(errors["numpy"]["linf"], errors["numba"]["linf"])
    metrics = {
        "gpu_within_worse_cpu_plus_floor": bool(
            gpu_error <= worse_cpu + _EQUIVALENCE_FLOOR
        ),
        "case": case,
        "dims": dims,
        "gate_count": len(steps),
        "oracle_decimal_digits": 60,
        "coefficient_source": "exact stored complex128 components",
        "absolute_amplitude_tolerance": _ATOL,
        "additive_equivalence_floor": _EQUIVALENCE_FLOOR,
        "reference_norm": reference_norm,
        "errors": errors,
        "comparisons": comparisons,
    }
    serialized = json.dumps(metrics, sort_keys=True)
    # Emit all engine errors and both comparisons before any numerical gate;
    # a failure must leave the negative result available in logs and JUnit.
    record_property("cupy_depth_accuracy", serialized)
    print(serialized, flush=True)
    for name, state in states.items():
        assert state.dtype == np.complex128, f"{name}: {serialized}"
        assert np.all(np.isfinite(state)), f"{name}: {serialized}"
        assert errors[name]["linf"] <= _ATOL, f"{name}: {serialized}"
    for cpu in ("numpy", "numba"):
        np.testing.assert_allclose(states["cupy"], states[cpu], atol=_ATOL, rtol=0)
    assert metrics["gpu_within_worse_cpu_plus_floor"], serialized


def test_deep_cycles_are_not_systematically_less_accurate_than_numba(
    depth_dependencies,
):
    # The single-sample bound above is a sanity bound and moves with the CPU
    # engines. This is the regression detector: over many seeded deep
    # circuits, the GPU's error minus Numba's (the compiled CPU engine) must
    # not be measurably positive.
    from fatqat.simulator._engine.cupy import CupySVEngine
    from fatqat.simulator._engine.nb import NumbaSVEngine

    mp = depth_dependencies.mp
    depth = 1024
    differences = []
    for seed in range(16):
        initial, cycle = _random_cycle(10_000 + seed)
        steps = cycle * (depth // len(cycle))
        states = {}
        for name, engine in (
            ("cupy", CupySVEngine(device_id=0)),
            ("numba", NumbaSVEngine()),
        ):
            engine.initialize((2, 2), initial_state=initial)
            for step in steps:
                engine.apply(step)
            states[name] = engine.export_state()
        with mp.workdps(60):
            reference = _oracle_evolve((2, 2), initial, steps, mp)
            errors = {
                name: _errors(state, reference, mp)["linf"]
                for name, state in states.items()
            }
        differences.append(errors["cupy"] - errors["numba"])
    mean = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1)) / len(differences) ** 0.5
    assert mean <= 2 * standard_error, (mean, standard_error, differences)
