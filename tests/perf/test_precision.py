"""The precision harness's oracle must be right before any GPU number means anything.

Hand-derived cases pin the oracle's public index conventions for every method;
the CPU runtimes are the control; the verdict logic is checked on synthetic
errors where the right answer is known.
"""

import json
from math import prod
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

mpmath = pytest.importorskip("mpmath")

# pylint: disable=wrong-import-position  # imports require the guard above

from perf.precision import (
    CASES,
    EPS,
    Case,
    build_steps,
    cuda_available,
    error_eps,
    paired_comparisons,
    initial_state,
    oracle,
    run_arm,
    summarise,
)

_ROOT = Path(__file__).resolve().parents[2]
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SHIFT3 = np.roll(np.eye(3, dtype=np.complex128), 1, axis=0)  # |k> -> |k+1 mod 3>


def _as_array(reference):
    with mpmath.mp.workdps(60):
        if isinstance(reference[0], list):
            return np.array([[complex(v) for v in row] for row in reference])
        return np.array([complex(v) for v in reference])


def _oracle(method, dims, steps, initial=None):
    with mpmath.mp.workdps(60):
        return _as_array(oracle(Case(method, "hand", dims, 0), steps, initial))


def test_statevector_targets_use_public_most_significant_first_order():
    # X on subsystem 0 of |00> gives |10>, flat index 2 in public order.
    initial = np.array([1, 0, 0, 0], dtype=np.complex128)
    result = _oracle("statevector", (2, 2), [("gate", (_X,), (0,))], initial)
    np.testing.assert_array_equal(result, [0, 0, 1, 0])


def test_first_target_is_the_most_significant_local_digit():
    # CX with control = first target. Control on subsystem 1, target 0:
    # |01> (subsystem 1 set) -> |11>.
    cx = np.eye(4, dtype=np.complex128)[[0, 1, 3, 2]]
    initial = np.array([0, 1, 0, 0], dtype=np.complex128)
    result = _oracle("statevector", (2, 2), [("gate", (cx,), (1, 0))], initial)
    np.testing.assert_array_equal(result, [0, 0, 0, 1])


def test_mixed_radix_digits():
    # dims (2, 3): shift the qutrit of |0,2> to |0,0>; flat = 2*... public order.
    initial = np.zeros(6, dtype=np.complex128)
    initial[2] = 1
    result = _oracle("statevector", (2, 3), [("gate", (_SHIFT3,), (1,))], initial)
    expected = np.zeros(6)
    expected[0] = 1
    np.testing.assert_array_equal(result, expected)


def test_unitary_is_the_embedded_operator():
    rng = np.random.default_rng(3)
    matrix, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    result = _oracle("unitary", (2, 2), [("gate", (matrix,), (0,))])
    np.testing.assert_allclose(result, np.kron(matrix, np.eye(2)), atol=1e-15)


def test_density_channel_matches_amplitude_damping_closed_form():
    gamma = 0.3
    kraus = (
        np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128),
        np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128),
    )
    rho = np.array([[0.4, 0.2 + 0.1j], [0.2 - 0.1j, 0.6]], dtype=np.complex128)
    result = _oracle("density_matrix", (2,), [("channel", kraus, (0,))], rho)
    expected = sum(k @ rho @ k.conj().T for k in kraus)
    np.testing.assert_allclose(result, expected, atol=1e-15)


def test_superop_uses_public_column_stacking():
    rng = np.random.default_rng(4)
    matrix, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    result = _oracle("superop", (2,), [("gate", (matrix,), (0,))])
    np.testing.assert_allclose(result, np.kron(matrix.conj(), matrix), atol=1e-15)
    rho = np.array([[0.3, 0.1j], [-0.1j, 0.7]])
    evolved = (result @ rho.reshape(-1, order="F")).reshape((2, 2), order="F")
    np.testing.assert_allclose(evolved, matrix @ rho @ matrix.conj().T, atol=1e-15)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.method}-{c.name}")
def test_cpu_control_sits_at_the_rounding_floor(case):
    # Control in the same run: the reference CPU runtime agrees with the oracle
    # to a few epsilons on every shipped case.
    steps = build_steps(case, 11)
    initial = initial_state(case, 11)
    with mpmath.mp.workdps(60):
        reference = oracle(case, steps, initial)
        errors = error_eps(
            run_arm(case, steps, initial, "numpy", simplify=False), reference
        )
    assert errors["linf_eps"] < 16, errors


def test_error_metric_resolves_a_planted_one_epsilon_scale_fault():
    case = Case("statevector", "planted", (2, 2, 2), 6)
    steps = build_steps(case, 2)
    initial = initial_state(case, 2)
    with mpmath.mp.workdps(60):
        reference = oracle(case, steps, initial)
        state = run_arm(case, steps, initial, "numpy", simplify=False)
        clean = error_eps(state, reference)["linf_eps"]
        state = state.copy()
        state[3] += 64 * EPS
        planted = error_eps(state, reference)["linf_eps"]
    assert clean < 8
    assert 56 < planted < 72 + clean


def test_build_steps_is_deterministic_and_frozen():
    case = CASES[4]
    first, second = build_steps(case, 9), build_steps(case, 9)
    assert len(first) == case.depth
    for (k1, m1, t1), (k2, m2, t2) in zip(first, second):
        assert (k1, t1) == (k2, t2)
        for a, b in zip(m1, m2):
            np.testing.assert_array_equal(a, b)
            assert not a.flags.writeable
    assert any(kind == "channel" for kind, _, _ in first)
    for kind, matrices, targets in first:
        if kind == "channel":
            size = prod(case.dims[q] for q in targets)
            completeness = sum(k.conj().T @ k for k in matrices)
            np.testing.assert_allclose(completeness, np.eye(size), atol=1e-12)


def _row(cpu, gpu):
    return {
        "seeds": [
            {
                "errors": {
                    "numpy": {"linf_eps": c},
                    "numba": {"linf_eps": c + 1},
                    "cuda": {"linf_eps": g},
                }
            }
            for c, g in zip(cpu, gpu)
        ]
    }


def test_verdicts():
    better, slightly, worse = (
        _row([2, 2], [1, 2]),
        _row([2], [2.5]),
        _row([2, 2], [1, 4.5]),
    )
    summarise([better, slightly, worse], with_gpu=True)
    assert better["verdict"] == "gpu_not_worse"
    # Half an epsilon worse is still worse: it must not round to "not worse".
    assert slightly["verdict"] == "gpu_within_1_eps"
    assert worse["verdict"] == "gpu_within_3_eps"
    assert worse["gpu_minus_best_cpu_linf_eps"]["seeds_gpu_not_worse"] == 1
    cpu_only = _row([1], [1])
    summarise([cpu_only], with_gpu=False)
    assert cpu_only["verdict"] is None


@pytest.mark.skipif(cuda_available(), reason="checks the no-GPU refusal")
def test_require_gpu_refuses_to_report_cpu_only(tmp_path):
    out = tmp_path / "p.json"
    completed = subprocess.run(
        [sys.executable, "perf/precision.py", "--require-gpu", "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
        cwd=_ROOT,
    )
    assert completed.returncode == 2
    assert not out.exists()


def test_paired_comparisons_have_the_right_sign_and_counts():
    rows = [_row([2, 2, 3], [1, 2, 4])]  # cuda - numpy = -1, 0, +1
    stats = paired_comparisons(rows)
    gap = stats["numpy_minus_cuda"]
    assert gap["mean_eps"] == pytest.approx(0.0)
    assert gap["circuits"] == 3
    assert gap["circuits_first_not_worse"] == 2  # numpy <= cuda on seeds 2 and 3
    assert gap["standard_error_eps"] == pytest.approx(1 / 3**0.5)
    # numba is numpy + 1 on every seed: a consistent gap with zero spread.
    assert stats["numpy_minus_numba"]["mean_eps"] == pytest.approx(-1.0)
    assert stats["numpy_minus_numba"]["standard_error_eps"] == pytest.approx(0.0)


def test_simplify_arm_stays_within_the_cpu_control(tmp_path):
    out = tmp_path / "p.json"
    completed = subprocess.run(
        [
            sys.executable,
            "perf/precision.py",
            "--simplify",
            "--seeds",
            "1",
            "--methods",
            "statevector",
            "unitary",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads(out.read_text())
    assert data["simplify"] is True
    assert {row["method"] for row in data["rows"]} == {"statevector", "unitary"}
