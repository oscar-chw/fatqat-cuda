"""Compare GPU and CPU round-off with an independent 60-digit oracle.

The oracle evolves the *stored complex128 coefficients*, not idealized gates
recomputed at high precision. It uses explicit basis-state sums, without any
FATQAT indexing, contraction, or matrix-application helpers. All engines must
meet the existing absolute 1e-12 amplitude contract. The additional comparison
allows the GPU at most eight float64 epsilons above each CPU's maximum error.
Raw errors and ratios are recorded; sub-floor ratios cannot establish a
backend accuracy ranking. These finite cases do not prove universal superiority.
"""

from itertools import product
import json
from math import prod

import numpy as np
import pytest

from fatqat._backends.steps import ApplyMatrixStep

_ATOL = 1e-12
_EQUIVALENCE_FLOOR = 8 * np.finfo(np.float64).eps


@pytest.fixture(scope="module", name="accuracy_dependencies")
def _accuracy_dependencies():
    mpmath = pytest.importorskip("mpmath")
    cupy = pytest.importorskip("cupy")
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device discovery unavailable: {error}")
    if count == 0:
        pytest.skip("No CUDA device available")
    return mpmath


def _exact_complex(value, mp):
    """Embed each binary64 component by its exact rational value."""
    real_num, real_den = float(value.real).as_integer_ratio()
    imag_num, imag_den = float(value.imag).as_integer_ratio()
    return mp.mpc(mp.mpf(real_num) / real_den, mp.mpf(imag_num) / imag_den)


def _flat_index(digits, dims):
    # Engine subsystem zero has unit place value; this is a scalar definition,
    # independent of the production engines' reshape and coset-walk approaches.
    return sum(digit * prod(dims[:index]) for index, digit in enumerate(digits))


def _local_index(digits, targets, dims):
    result = 0
    for target in targets:
        result = result * dims[target] + digits[target]
    return result


def _oracle_evolve(dims, initial_state, steps, mp):
    """Explicitly sum all source basis states compatible with each output."""
    basis = list(product(*(range(dim) for dim in dims)))
    state = [_exact_complex(value, mp) for value in initial_state]
    for step in steps:
        targets = step.target_indices
        spectators = tuple(index for index in range(len(dims)) if index not in targets)
        matrix = [[_exact_complex(value, mp) for value in row] for row in step.matrix]
        evolved = [mp.mpc(0) for _ in state]
        for output_digits in basis:
            row = _local_index(output_digits, targets, dims)
            terms = []
            for input_digits in basis:
                if all(input_digits[q] == output_digits[q] for q in spectators):
                    column = _local_index(input_digits, targets, dims)
                    terms.append(
                        matrix[row][column] * state[_flat_index(input_digits, dims)]
                    )
            evolved[_flat_index(output_digits, dims)] = mp.fsum(terms)
        state = evolved
    return state


def _random_case(dims, seed, depth, *, reverse=False):
    rng = np.random.default_rng(seed)
    initial = np.asarray(
        rng.normal(size=prod(dims)) + 1j * rng.normal(size=prod(dims)),
        dtype=np.complex128,
    )
    initial /= np.linalg.norm(initial)
    steps = []
    for index in range(depth):
        width = 1 + index % min(3, len(dims))
        targets = tuple(int(q) for q in rng.choice(len(dims), width, replace=False))
        size = prod(dims[q] for q in targets)
        matrix, _ = np.linalg.qr(
            rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        )
        # This frozen array is the numerical problem for every backend and the
        # oracle. Recomputing QR or normalizing in mpmath would change it.
        matrix = np.array(matrix, dtype=np.complex128)
        matrix.flags.writeable = False
        steps.append(ApplyMatrixStep(matrix, targets))
    if reverse:
        # This is cancellation-rich but not an exact identity: the adjoints
        # invert stored rounded matrices, which are only approximately unitary.
        for step in reversed(steps.copy()):
            adjoint = np.array(step.matrix.conj().T, dtype=np.complex128)
            adjoint.flags.writeable = False
            steps.append(ApplyMatrixStep(adjoint, step.target_indices))
    return initial, steps


def _entangled_case(size):
    initial = np.zeros(2**size, dtype=np.complex128)
    initial[0] = 1
    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    controlled_x = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )
    steps = [ApplyMatrixStep(hadamard, (0,))]
    steps.extend(ApplyMatrixStep(controlled_x, (0, q)) for q in range(1, size))
    return initial, steps


def _errors(actual, reference, mp):
    differences = [
        abs(_exact_complex(value, mp) - expected)
        for value, expected in zip(actual, reference, strict=True)
    ]
    return {
        "linf": float(max(differences)),
        "l2": float(mp.sqrt(mp.fsum(error**2 for error in differences))),
    }


@pytest.mark.parametrize(
    "case,dims,seed,depth,reverse",
    [
        ("bell", (2, 2), 0, 0, False),
        ("ghz", (2, 2, 2, 2), 0, 0, False),
        ("dense_ordered", (2, 2, 2, 2), 4150, 64, False),
        ("mixed_radix", (3, 2, 3), 5280, 48, False),
        ("cancellation", (2, 2, 2, 2), 73, 32, True),
    ],
    ids=["bell", "ghz", "dense_ordered", "mixed_radix", "cancellation"],
)
def test_accuracy_against_stored_coefficient_oracle(
    accuracy_dependencies, record_property, case, dims, seed, depth, reverse
):
    from fatqat.simulator._engine.cupy import CupySVEngine
    from fatqat.simulator._engine.nb import NumbaSVEngine
    from fatqat.simulator._engine.np import NumpySVEngine

    if case in ("bell", "ghz"):
        initial, steps = _entangled_case(len(dims))
    else:
        initial, steps = _random_case(dims, seed, depth, reverse=reverse)
    mp = accuracy_dependencies.mp
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
        assert states[name].dtype == np.complex128
        assert np.all(np.isfinite(states[name]))

    with mp.workdps(60):
        reference = _oracle_evolve(dims, initial, steps, mp)
        errors = {name: _errors(state, reference, mp) for name, state in states.items()}
        reference_norm = float(mp.sqrt(mp.fsum(abs(value) ** 2 for value in reference)))

    comparisons = {}
    for cpu in ("numpy", "numba"):
        cpu_error = errors[cpu]["linf"]
        gpu_error = errors["cupy"]["linf"]
        comparisons[cpu] = {
            "gpu_to_cpu_linf_ratio": gpu_error / cpu_error if cpu_error else None,
            "ratio_interpretation": (
                "above_equivalence_floor"
                if cpu_error > _EQUIVALENCE_FLOOR
                else "unresolved_at_equivalence_floor"
            ),
            "gpu_minus_cpu_linf": gpu_error - cpu_error,
        }
    metrics = {
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
    record_property("cupy_accuracy", serialized)
    print(serialized)
    for name, error in errors.items():
        assert error["linf"] <= _ATOL, f"{name}: {serialized}"
    for cpu in ("numpy", "numba"):
        np.testing.assert_allclose(states["cupy"], states[cpu], atol=_ATOL, rtol=0)
        assert (
            errors["cupy"]["linf"] <= errors[cpu]["linf"] + _EQUIVALENCE_FLOOR
        ), serialized
