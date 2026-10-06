"""The scaling benchmark must time real work, refuse silent CPU-only runs and leak nothing."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from perf.scaling import WORKLOADS, _fingerprint, _quartiles, cuda_available, sizes_for
from perf.scrub_check import GENERIC_PATTERNS, scan

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "perf" / "scaling.py"


def test_sizes_respect_the_limit_and_halve_for_matrix_methods():
    assert sizes_for("sv_state", 16) == [12, 14, 16]
    # 4**n entries: a 9-qubit density matrix is as large as an 18-qubit state.
    assert max(sizes_for("dm_noise", 16)) == 9
    assert max(sizes_for("unitary", 26)) == 14


def test_fingerprint_moves_with_a_small_error_and_not_with_a_copy():
    rng = np.random.default_rng(0)
    state = rng.normal(size=1 << 12) + 1j * rng.normal(size=1 << 12)
    state /= np.linalg.norm(state)
    base = np.array(_fingerprint(np, state))
    np.testing.assert_array_equal(base, _fingerprint(np, state.copy()))
    perturbed = state.copy()
    perturbed[17] += 1e-7
    assert np.abs(np.array(_fingerprint(np, perturbed)) - base).max() > 1e-10


def _run(tmp_path, *extra):
    out = tmp_path / "scaling.json"
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--threads", "2", "--out", str(out), *extra],
        capture_output=True,
        text=True,
        check=False,
        cwd=_ROOT,
        timeout=600,
    )
    return completed, out


def test_end_to_end_small_run_is_complete_and_clean(tmp_path):
    completed, out = _run(tmp_path, "--max-qubits", "12", "--repeats", "2")
    assert completed.returncode == 0, completed.stderr
    data = json.loads(out.read_text())
    assert data["configurations"]["numba"] == "CPU (compiled Numba), 2 threads"
    assert {row["workload"] for row in data["rows"]} == set(WORKLOADS)
    for row in data["rows"]:
        timing = row["numba"]
        assert len(timing["warm_ms"]) == 2
        assert 0 < timing["q1_ms"] <= timing["median_ms"] <= timing["q3_ms"]
        assert timing["first_call_ms"] > 0
        if sys.platform == "win32":
            assert timing["peak_rss_mib"] is None  # no resource module
        else:
            assert timing["peak_rss_mib"] > 0
        assert timing["gpu_pool_reserved_mib"] is None
    # The output must not carry machine details by construction.
    assert not scan(out.read_text(), GENERIC_PATTERNS)
    text = out.read_text().lower()
    for field in (
        "platform",
        "cpu_count",
        "logical_cpus",
        "affinity",
        "hostname",
        "memory_total",
    ):
        assert field not in text


@pytest.mark.skipif(cuda_available(), reason="checks the no-GPU refusal")
def test_require_gpu_refuses_to_report_cpu_only(tmp_path):
    completed, out = _run(tmp_path, "--require-gpu", "--max-qubits", "12")
    assert completed.returncode == 2
    assert not out.exists()


def test_threads_must_be_positive(tmp_path):
    completed, _ = _run(tmp_path, "--threads", "0")
    assert completed.returncode != 0


def test_simplify_flag_reaches_every_call_and_the_output(tmp_path):
    completed, out = _run(
        tmp_path,
        "--max-qubits",
        "12",
        "--repeats",
        "1",
        "--workloads",
        "sv_state",
        "--simplify",
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads(out.read_text())
    assert data["simplify"] is True
    assert [row["qubits"] for row in data["rows"]] == [12]


def test_cpu_only_run_can_serve_as_a_reference(tmp_path):
    first, out = _run(
        tmp_path,
        "--max-qubits",
        "12",
        "--repeats",
        "1",
        "--workloads",
        "sv_state",
        "--runtimes",
        "numba",
    )
    assert first.returncode == 0, first.stderr
    data = json.loads(out.read_text())
    assert data["configurations"]["cuda"] is None
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second, out2 = _run(
        second_dir,
        "--max-qubits",
        "12",
        "--repeats",
        "1",
        "--workloads",
        "sv_state",
        "--reference",
        str(out),
        "--runtimes",
        "numba",
    )
    assert second.returncode == 0, second.stderr
    assert json.loads(out2.read_text())["reference_file"] == "scaling.json"


def test_quartiles_stay_within_the_samples():
    # Two very different warm calls: the default method would put Q1 below 0.
    q1, median, q3 = _quartiles([1.0, 50.0])
    assert 1.0 <= q1 <= median <= q3 <= 50.0
    assert _quartiles([7.0]) == [7.0, 7.0, 7.0]
