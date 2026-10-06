"""The simplify check's ideal reference must be right before its verdict means anything.

Hand cases pin the reference's qubit order and control convention; the adder
must add, or the speed workload is not the circuit it claims to be; the
verdict must fail on each kind of regression it exists to catch.
"""

import numpy as np
import pytest

mpmath = pytest.importorskip("mpmath")

# pylint: disable=wrong-import-position  # imports require the guard above

from perf.simplify_check import (
    MAX_PASS_FRACTION,
    MIN_SPEEDUP,
    adder,
    build_program,
    error_eps,
    ideal_state,
    run,
    verdict,
)


def _basis(state):
    with mpmath.mp.workdps(60):
        values = np.array([complex(v) for v in state])
    (index,) = np.flatnonzero(np.abs(values) > 0.5)
    return index


def test_qubit_zero_is_the_most_significant_bit():
    assert _basis(ideal_state(3, [("X", None, (0,))])) == 0b100


def test_the_first_cx_target_is_the_control():
    gates = [("X", None, (2,)), ("CX", None, (2, 0))]
    assert _basis(ideal_state(3, gates)) == 0b101


@pytest.mark.parametrize(("a", "b"), [(0, 0), (1, 2), (3, 3), (2, 1)])
def test_the_adder_adds(a, b):
    n, gates = adder(2, a, b)
    index = _basis(ideal_state(n, gates))
    # Sum is written into the b register (qubits 1, 3), carry-out into z.
    bits = [(index >> (n - 1 - q)) & 1 for q in range(n)]
    total = bits[1] + 2 * bits[3] + 4 * bits[n - 1]
    assert total == a + b
    # a is restored, carry-in returns to 0.
    assert bits[2] + 2 * bits[4] == a and bits[0] == 0


def test_fatqat_agrees_with_the_reference_to_the_rounding_floor():
    gates = [
        ("H", None, (0,)),
        ("T", None, (0,)),
        ("CX", None, (0, 1)),
        ("RZ", 0.3, (1,)),
    ]
    with mpmath.mp.workdps(60):
        error = error_eps(
            run(build_program(2, gates), "statevector", "numpy", False),
            ideal_state(2, gates),
        )
    assert error < 8


def _row(workload, speedup=2.0, fraction=0.0, runtime="numba"):
    return {
        "workload": workload,
        "runtime": runtime,
        "speedup": speedup,
        "pass_fraction_of_plain": fraction,
    }


def test_verdict_fails_on_each_regression_and_passes_otherwise():
    fine = {"numba": {"mean_simplify_minus_plain_eps": -1.0, "standard_error_eps": 0.2}}
    rows = [_row("adder"), _row("rotated_qft_control")]
    assert verdict(fine, rows) == []
    worse = {"numba": {"mean_simplify_minus_plain_eps": 0.5, "standard_error_eps": 0.1}}
    assert verdict(worse, rows)
    assert verdict(fine, [_row("adder", speedup=MIN_SPEEDUP - 0.01)])
    assert verdict(fine, [_row("rotated_qft_control", fraction=MAX_PASS_FRACTION * 2)])
    # The GPU's wall time is reported, not gated: it runs where load varies.
    assert verdict(fine, [_row("adder", speedup=1.0, runtime="cuda")]) == []
