"""The publication scrub must catch machine details and pass run configuration.

Both directions matter: a pattern that misses a GPU model leaks hardware, and a
pattern that flags "4 threads" would block the benchmark tables already public.
"""

from pathlib import Path
import subprocess
import sys

import pytest

from perf.scrub_check import GENERIC_PATTERNS, main, scan

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "perf" / "scrub_check.py"
# Path and address examples are assembled at run time so that no literal
# lookalike sits in the repository for history scanners to flag.
_SEP = "/"


@pytest.mark.parametrize(
    "text",
    [
        "measured on an RTX 4090",
        "GeForce card",
        "one A100 card",
        "an H100 node",
        "NVIDIA L40S",
        "dual Xeon Gold",
        "AMD EPYC 9654",
        "Core i7 laptop",
        "Apple M1 Pro",
        "an M4 Ultra",
        "a 12-core CPU",
        "6 physical cores",
        "512 GB RAM",
        "24 GB of memory",
        "8x GPUs",
        "reached at " + ".".join(["10", "0", "0", "12"]),
        "cd " + _SEP + "home" + _SEP + "alice/run",
        _SEP + "Users" + _SEP + "bob/Desktop",
        r"C:\Users\carol\data",
        "venv in " + _SEP + "home" + _SEP + "dana/env",
        "GPU-3f2a1b4c-aaaa",
        "bus 0000:41:00.0",
        "driver 535.104",
        '"hostname": "gpu-box"',
    ],
)
def test_machine_details_are_found(text):
    assert scan(text, GENERIC_PATTERNS), text


@pytest.mark.parametrize(
    "text",
    [
        "one NVIDIA GPU",
        "compiled Numba on 4 threads and on all cores",
        "16 workers",
        "live GPU memory 128.065 MiB",
        "complex128 state, CUDA 13 wheels",
        "numpy==2.3.5 numba==0.67.0",
        "24 qubits, 281.7 ms vs 46.95 ms",
        "src/fatqat/simulator/_engine/cupy.py",
        "results/benchmarks.json",
        "within 8 machine epsilons",
    ],
)
def test_run_configuration_is_not_flagged(text):
    assert not scan(text, GENERIC_PATTERNS), scan(text, GENERIC_PATTERNS)


@pytest.mark.parametrize(
    "relative",
    [
        "README.md",
        "NOTICE",
        "results/README.md",
        "results/benchmarks.json",
        "docs/README.md",
        "docs/benchmarks.md",
        "docs/cuda-coverage.md",
    ],
)
def test_published_files_are_clean(relative):
    # Control: these were audited clean before this check existed. If one
    # fails, either a leak was published or a pattern is over-broad.
    path = _ROOT / relative
    assert path.is_file(), relative
    assert not scan(path.read_text(encoding="utf-8"), GENERIC_PATTERNS)


def test_missing_input_fails(tmp_path, capsys):
    assert main([str(tmp_path / "absent.json")]) == 2
    assert "missing file" in capsys.readouterr().out


def test_missing_private_file_fails_instead_of_scanning_less(tmp_path):
    target = tmp_path / "clean.md"
    target.write_text("one NVIDIA GPU\n", encoding="utf-8")
    assert main([str(target), "--private", str(tmp_path / "nope.txt")]) == 2


def test_private_patterns_match_without_echoing_the_name(tmp_path, capsys):
    private = tmp_path / "private.txt"
    private.write_text("# comment\n\nsecret-host-\\d+\n", encoding="utf-8")
    target = tmp_path / "out.json"
    target.write_text('{"note": "ran on secret-host-7"}\n', encoding="utf-8")
    assert main([str(target), "--private", str(private)]) == 1
    output = capsys.readouterr().out
    assert "private pattern" in output
    assert "secret-host" not in output


def test_command_line_exit_codes(tmp_path):
    clean = tmp_path / "clean.md"
    clean.write_text("one NVIDIA GPU, compiled Numba\n", encoding="utf-8")
    dirty = tmp_path / "dirty.md"
    dirty.write_text("one NVIDIA GPU (an RTX 4090)\n", encoding="utf-8")

    def run(*files):
        return subprocess.run(
            [sys.executable, str(_SCRIPT), *map(str, files)],
            capture_output=True,
            text=True,
            check=False,
        ).returncode

    assert run(clean) == 0
    assert run(clean, dirty) == 1
    assert run(clean, tmp_path / "absent.md") == 2
