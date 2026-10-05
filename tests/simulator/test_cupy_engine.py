"""Built-in CUDA statevector behavior, with optional hardware tests.

Hardware availability is checked only for tests that execute a device plan.
Host-side validation and optional-dependency tests still run without CUDA.
"""

import builtins
import subprocess
import sys

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.steps import ApplyChannelStep, ApplyMatrixStep
from fatqat.errors import (
    BackendValidationError,
    ResultFieldUnavailableError,
)
from fatqat.implementation import MatrixImplementationMap
from fatqat.implementation.matrices import shift_matrix
from fatqat.job import Job
from fatqat.result import Result
from fatqat.simulator import Simulator

_STATE_ONLY = {"counts": False, "final_state": True}
_ATOL = 1e-12
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)


def _cuda_simulator(*, device_id=None, implementation_map=None):
    """Exercise the standard public CUDA route in every public behavior test."""
    return Simulator(
        method="SV",
        runtime="cuda",
        device_id=device_id,
        implementation_map=implementation_map,
    )


@pytest.fixture(scope="module", name="gpu_available")
def _gpu_available():
    """Skip absent CuPy/devices; computation failures after this check fail."""
    cupy = pytest.importorskip("cupy")
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device discovery unavailable: {error}")
    if count == 0:
        pytest.skip("No CUDA device available")
    return cupy


@pytest.fixture(name="gpu_engine")
def _gpu_engine(gpu_available):
    from fatqat.simulator._engine.cupy import CupySVEngine

    return CupySVEngine(device_id=0)


@pytest.fixture(name="no_cupy_import")
def _no_cupy_import(monkeypatch):
    """A host-side rejection must not need the optional device package."""
    from fatqat.simulator._engine import cupy as cupy_engine_module

    original_import = builtins.__import__
    original_import_module = cupy_engine_module.import_module

    def guarded_import(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise AssertionError("Validation attempted to import CuPy")
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise AssertionError("Validation attempted to load CuPy")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(cupy_engine_module, "import_module", guarded_import_module)


def _state(backend, program, initial_state=None):
    job = backend.run(
        program, shots=0, initial_state=initial_state, result_config=_STATE_ONLY
    )
    assert isinstance(job, Job)
    assert job.status == "DONE"
    result = job.result()
    assert isinstance(result, Result)
    state = result.get_statevector()
    assert isinstance(state, np.ndarray)
    assert state.dtype == np.complex128
    return state


def _ghz(size, *, measure=False):
    program = fq.Program(size, size if measure else 0)
    program.add(ops.H, 0)
    for target in range(1, size):
        program.add(ops.CX, (0, target))
    if measure:
        program.measure_all()
    return program


def test_optional_backend_import_preserves_cpu_default():
    # A fresh interpreter avoids sys.modules hiding an eager CuPy import.
    script = """
import sys
class NoCupy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cupy' or fullname.startswith('cupy.'):
            raise AssertionError('CPU import attempted to load CuPy')
sys.meta_path.insert(0, NoCupy())
import fatqat as fq
from fatqat.simulator import Simulator
Simulator(method="SV", runtime="cuda")
result = Simulator().run(fq.Program(1)).result()
assert result.get_statevector().tolist() == [1 + 0j, 0j]
assert result.metadata["runtime"] == "numba"
assert 'cupy' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("shots", [1, 64])
@pytest.mark.parametrize("sweep", [False, True])
def test_measurement_basis_change_rejected_before_process_dispatch(
    no_cupy_import, monkeypatch, shots, sweep
):
    from fatqat.simulator import simulator as simulator_module

    def forbidden_dispatch(*args, **kwargs):
        raise AssertionError("CUDA pilot attempted process dispatch")

    monkeypatch.setattr(simulator_module, "_run_shots_in_processes", forbidden_dispatch)
    theta = fq.Parameter("theta")
    program = fq.Program(1, 1)
    program.add(ops.RY(theta if sweep else 0.4), 0)
    program.measure_all()
    estimator = fq.Estimator(_cuda_simulator())
    observable = fq.Observable([("X", 1)])
    with pytest.raises(
        BackendValidationError, match="measurement has no well-defined expectation"
    ):
        if sweep:
            estimator.run_sweep(program, observable, {theta: [0.2, 0.7]}, shots=shots)
        else:
            estimator.run(program, observable, shots=shots)


def test_zero_norm_sampling_matches_cpu_error(gpu_available):
    program = fq.Program(1, 1)
    program.measure_all()
    for backend, error in (
        (_cuda_simulator(), ValueError),
        (Simulator(runtime="numpy"), ValueError),
        (Simulator(runtime="numba"), ZeroDivisionError),
    ):
        job = backend.run(program, shots=10, initial_state=np.zeros(2, dtype=complex))
        assert job.status == "ERROR"
        with pytest.raises(error):
            job.result()


@pytest.mark.parametrize("targets", [(0, 2), (2, 0)])
def test_public_nonadjacent_ordered_control_targets(gpu_available, targets):
    program = fq.Program(3)
    program.add(ops.X, targets[0])
    program.add(ops.CX, targets)
    # Public |q0 q1 q2> = |101>, independent of which outer qubit controls.
    expected = np.zeros(8, dtype=np.complex128)
    expected[5] = 1
    np.testing.assert_allclose(
        _state(_cuda_simulator(), program), expected, atol=_ATOL, rtol=0
    )


def test_public_mixed_dimensions_keep_operand_order(gpu_available):
    class MixedGate(ops.Operation):
        name = "CupyMixedGate"
        num_subsystems = 2

    qubit = fq.QuantumRegister(1, dim=2)
    qutrit = fq.QuantumRegister(1, dim=3)
    program = fq.Program([qubit, qutrit])
    program.add(MixedGate(), (qutrit[0], qubit[0]))
    matrix = np.kron(shift_matrix(3, 1), _X)
    implementations = MatrixImplementationMap()
    implementations.add(MixedGate, matrix)
    start = np.zeros(6, dtype=np.complex128)
    start[1] = 1  # public |qubit=0, qutrit=1>
    expected = np.zeros(6, dtype=np.complex128)
    expected[5] = 1  # public |qubit=1, qutrit=2>
    np.testing.assert_allclose(
        _state(_cuda_simulator(implementation_map=implementations), program, start),
        expected,
        atol=_ATOL,
        rtol=0,
    )


@pytest.mark.parametrize("size", [2, 5], ids=["bell", "ghz"])
def test_entangled_state_matches_both_cpu_runtimes(gpu_available, size):
    program = _ghz(size)
    actual = _state(_cuda_simulator(), program)
    expected = np.zeros(2**size, dtype=np.complex128)
    expected[[0, -1]] = 1 / np.sqrt(2)
    np.testing.assert_allclose(actual, expected, atol=_ATOL, rtol=0)
    for runtime in ("numpy", "numba"):
        reference = _state(Simulator(runtime=runtime), program)
        np.testing.assert_allclose(actual, reference, atol=_ATOL, rtol=0)


@pytest.mark.parametrize("dims", [(2, 2, 2, 2), (3, 2, 2)])
def test_random_local_gates_match_numpy_and_numba(gpu_engine, dims):
    from fatqat.simulator._engine.nb import NumbaSVEngine
    from fatqat.simulator._engine.np import NumpySVEngine

    rng = np.random.default_rng(4150)
    start = rng.normal(size=np.prod(dims)) + 1j * rng.normal(size=np.prod(dims))
    start /= np.linalg.norm(start)
    engines = (gpu_engine, NumpySVEngine(), NumbaSVEngine())
    for engine in engines:
        engine.initialize(dims, initial_state=start)
    for _ in range(40):
        targets = tuple(int(q) for q in rng.choice(len(dims), size=2, replace=False))
        size = int(np.prod([dims[q] for q in targets]))
        matrix, _ = np.linalg.qr(
            rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        )
        step = ApplyMatrixStep(matrix, targets)
        for engine in engines:
            engine.apply(step)
    actual = gpu_engine.export_state()
    assert actual.dtype == np.complex128
    for engine in engines[1:]:
        np.testing.assert_allclose(actual, engine.export_state(), atol=_ATOL, rtol=0)


@pytest.mark.parametrize("targets", [(2,), (2, 0)], ids=["one_qubit", "two_qubit"])
@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_gate_matrix_layout_preserves_complex_coefficients(gpu_engine, targets, layout):
    from fatqat.simulator._engine.np import NumpySVEngine

    rng = np.random.default_rng(917)
    size = 2 ** len(targets)
    canonical, _ = np.linalg.qr(
        rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    )
    if layout == "fortran":
        matrix = np.asfortranarray(canonical)
    else:
        backing = np.zeros((2 * size, 2 * size), dtype=np.complex128)
        matrix = backing[::2, ::2]
        matrix[:] = canonical
    # ApplyMatrixStep intentionally preserves already-read-only coefficients.
    # This makes the actual non-C layout reach the device upload boundary.
    matrix.flags.writeable = False
    assert not matrix.flags.c_contiguous
    initial = rng.normal(size=8) + 1j * rng.normal(size=8)
    initial /= np.linalg.norm(initial)
    reference = NumpySVEngine()
    for engine in (gpu_engine, reference):
        engine.initialize((2, 2, 2), initial_state=initial)
    gpu_engine.apply(ApplyMatrixStep(matrix, targets))
    reference.apply(ApplyMatrixStep(canonical, targets))
    np.testing.assert_allclose(
        gpu_engine.export_state(), reference.export_state(), atol=_ATOL, rtol=0
    )
    np.testing.assert_array_equal(matrix, canonical)
    # An adjoint is a common source of Fortran-order gate storage. Apply it
    # after the dense complex gate so conjugation and layout both matter.
    adjoint = canonical.conj().T
    adjoint.flags.writeable = False
    gpu_engine.apply(ApplyMatrixStep(adjoint, targets))
    np.testing.assert_allclose(gpu_engine.export_state(), initial, atol=_ATOL, rtol=0)


def test_engine_input_and_export_are_owned_copies(gpu_engine):
    source = np.array([0, 1, 0, 0], dtype=np.complex128)
    original = source.copy()
    gpu_engine.initialize((2, 2), initial_state=source)
    source[:] = 0
    np.testing.assert_array_equal(gpu_engine.export_state(), original)
    exported = gpu_engine.export_state()
    exported[:] = 999
    gpu_engine.apply(ApplyMatrixStep(_X, (1,)))
    np.testing.assert_array_equal(gpu_engine.export_state(), [0, 0, 0, 1])
    gpu_engine.initialize((2, 2))
    np.testing.assert_array_equal(gpu_engine.export_state(), [1, 0, 0, 0])


def test_probabilities_and_partial_collapse_mixed_dimensions(gpu_engine):
    # Little-endian digits: qutrit q0=1; qubit q1 remains in superposition.
    start = np.zeros(6, dtype=np.complex128)
    start[[1, 4]] = [1, 1j]
    gpu_engine.initialize((3, 2), initial_state=start)
    expected_probabilities = np.zeros(6)
    expected_probabilities[[1, 4]] = 0.5
    np.testing.assert_allclose(
        gpu_engine.probabilities(), expected_probabilities, atol=_ATOL, rtol=0
    )
    before = gpu_engine.export_state()
    sampled = gpu_engine.collapse((0,), np.random.default_rng(123))
    assert sampled in (1, 4)
    np.testing.assert_array_equal(before, start)
    np.testing.assert_allclose(
        gpu_engine.export_state(), start / np.sqrt(2), atol=_ATOL, rtol=0
    )


def test_terminal_counts_repeat_and_follow_born_probabilities(gpu_available):
    backend = _cuda_simulator()
    program = _ghz(3, measure=True)
    options = {"shots": 4096, "simulation_config": {"seed": 73}}
    first = backend.run(program, **options).result()
    second = backend.run(program, **options).result()
    counts = first.get_counts()
    assert counts == second.get_counts()
    assert set(counts) == {"000", "111"}
    assert sum(counts.values()) == 4096
    assert 0.46 < counts["000"] / 4096 < 0.54
    with pytest.raises(ResultFieldUnavailableError):
        first.get_statevector()


def test_terminal_counts_and_final_state_share_one_shot(gpu_available):
    result = (
        _cuda_simulator()
        .run(
            _ghz(3, measure=True),
            shots=1,
            simulation_config={"seed": 91},
            result_config={"counts": True, "final_state": True},
        )
        .result()
    )
    counts = result.get_counts()
    assert len(counts) == 1
    outcome, count = next(iter(counts.items()))
    assert count == 1
    expected = np.zeros(8, dtype=np.complex128)
    expected[int(outcome, 2)] = 1
    np.testing.assert_allclose(result.get_statevector(), expected, atol=_ATOL, rtol=0)


def test_terminal_counts_follow_classical_destination_order(gpu_available):
    program = fq.Program(3, 3)
    program.add(ops.X, 0)
    program.measure((0, 2, 1), (2, 0, 1))
    result = _cuda_simulator().run(program, shots=17).result()
    assert result.get_counts() == {"001": 17}


def test_supported_serial_controls_and_execution_error_job(gpu_available):
    backend = _cuda_simulator()
    config = {
        "shot_parallelism": "serial",
        "kernel_parallelism": "auto",
        "fusion": False,
    }
    result = backend.run(_ghz(2), simulation_config=config).result()
    np.testing.assert_allclose(
        result.get_statevector(), [1, 0, 0, 1] / np.sqrt(2), atol=_ATOL, rtol=0
    )
    job = backend.run(_ghz(2, measure=True), simulation_config={"seed": -1})
    assert isinstance(job, Job)
    assert job.status == "ERROR"
    with pytest.raises(ValueError):
        job.result()


def test_sweep_preserves_result_order_and_initial_input(gpu_available):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RY(theta), 0)
    angles = [0.0, 0.7, np.pi]
    source = np.array([0, 1], dtype=np.complex128)
    job = _cuda_simulator().run_sweep(
        program,
        {theta: angles},
        shots=0,
        initial_state=source,
        result_config=_STATE_ONLY,
    )
    assert isinstance(job, Job)
    assert job.status == "DONE"
    results = job.result()
    assert isinstance(results, list)
    assert len(results) == len(angles)
    np.testing.assert_array_equal(source, [0, 1])
    for angle, result in zip(angles, results, strict=True):
        expected = [-np.sin(angle / 2), np.cos(angle / 2)]
        np.testing.assert_allclose(
            result.get_statevector(), expected, atol=_ATOL, rtol=0
        )


@pytest.mark.parametrize(
    "config",
    [
        {"shot_parallelism": "threads"},
        {"shot_parallelism": "processes"},
        {"kernel_parallelism": "threads"},
        {"max_workers": 2},
        {"fusion": True},
    ],
)
def test_cpu_execution_controls_rejected_before_device_use(no_cupy_import, config):
    with pytest.raises(BackendValidationError):
        _cuda_simulator().run(_ghz(2, measure=True), simulation_config=config)


@pytest.mark.parametrize("feature", ["reset", "mid_measurement", "condition"])
@pytest.mark.parametrize("sweep", [False, True])
def test_dynamic_programs_rejected_before_device_use(no_cupy_import, feature, sweep):
    theta = fq.Parameter("theta")
    program = fq.Program(2, 2)
    program.add(ops.RY(theta if sweep else 0.3), 0)
    if feature == "reset":
        program.add(ops.Reset, 0)
    elif feature == "mid_measurement":
        program.measure(0, 0)
        program.add(ops.H, 0)
    else:
        program.add(ops.X, 1, condition=(0, 1))
    program.measure_all()
    backend = _cuda_simulator()
    with pytest.raises(BackendValidationError):
        if sweep:
            backend.run_sweep(program, {theta: [0.1, 0.7]})
        else:
            backend.run(program)


def test_invalid_state_shape_rejected_before_device_use(no_cupy_import):
    with pytest.raises(BackendValidationError, match="initial_state has shape"):
        _cuda_simulator().run(fq.Program(2), initial_state=np.zeros(3))


def test_stochastic_final_state_requires_one_shot(no_cupy_import):
    with pytest.raises(BackendValidationError):
        _cuda_simulator().run(
            _ghz(2, measure=True),
            shots=2,
            result_config={"counts": True, "final_state": True},
        )


@pytest.mark.parametrize("feature", ["channel", "reset"])
def test_unsupported_engine_operations_reject_before_allocation(
    no_cupy_import, feature
):
    from fatqat.simulator._engine.cupy import CupySVEngine

    engine = CupySVEngine(device_id=0)
    rng = np.random.default_rng(0)
    with pytest.raises(BackendValidationError):
        if feature == "channel":
            engine.apply_channel(ApplyChannelStep((_X,), (0,)), rng)
        else:
            engine.reset_subsystems((0,), rng)


@pytest.mark.parametrize("sweep", [False, True])
@pytest.mark.parametrize("shots", [0, 64], ids=["exact", "sampled"])
@pytest.mark.parametrize("label", ["Z", "I"], ids=["observable", "identity"])
@pytest.mark.parametrize(
    "config",
    [
        {"shot_parallelism": "threads"},
        {"shot_parallelism": "processes"},
        {"kernel_parallelism": "threads"},
        {"max_workers": 2},
        {"fusion": True},
    ],
)
def test_expectation_controls_reject_directly_without_device_use(
    no_cupy_import, sweep, shots, label, config
):
    angle = fq.Parameter("angle")
    program = fq.Program(1)
    program.add(ops.RY(angle if sweep else 0.7), 0)
    estimator = fq.Estimator(_cuda_simulator())
    observable = fq.Observable([(label, 1.0)])
    # Calling result() here would also accept a late failed Job. Validation
    # must instead raise from the public request itself, including identity
    # observables whose value requires no numerical execution.
    with pytest.raises(BackendValidationError):
        if sweep:
            estimator.run_sweep(
                program,
                observable,
                {angle: [0.2, 0.7]},
                shots=shots,
                simulation_config=config,
            )
        else:
            estimator.run(program, observable, shots=shots, simulation_config=config)


@pytest.mark.parametrize("sweep", [False, True])
@pytest.mark.parametrize("shots", [0, 64], ids=["exact", "sampled"])
@pytest.mark.parametrize("feature", ["reset", "condition", "measurement"])
def test_expectation_dynamic_programs_reject_directly_without_device_use(
    no_cupy_import, sweep, shots, feature
):
    angle = fq.Parameter("angle")
    program = fq.Program(1, 1)
    program.add(ops.RY(angle if sweep else 0.7), 0)
    if feature == "reset":
        program.add(ops.Reset, 0)
    elif feature == "condition":
        program.add(ops.X, 0, condition=(0, 1))
    else:
        program.measure(0, 0)
        program.add(ops.H, 0)
    estimator = fq.Estimator(_cuda_simulator())
    observable = fq.Observable([("Z", 1.0)])
    # CUDA restrictions are checked during common preparation, before the
    # estimator's generic statevector capability hooks.
    with pytest.raises(BackendValidationError):
        if sweep:
            estimator.run_sweep(program, observable, {angle: [0.2, 0.7]}, shots=shots)
        else:
            estimator.run(program, observable, shots=shots)


@pytest.mark.parametrize("sweep", [False, True])
@pytest.mark.parametrize("shots", [0, 4096], ids=["exact", "sampled"])
def test_ideal_expectations_preserve_public_factors_and_result_order(
    gpu_available, sweep, shots
):
    angle = fq.Parameter("angle")
    program = fq.Program(2)
    program.add(ops.RY(angle if sweep else 0.9), 0)
    program.add(ops.CX, (0, 1))
    observables = [fq.Observable([(label, 1.0)]) for label in ("ZZ", "ZI", "XX")]
    estimator = fq.Estimator(_cuda_simulator())
    options = {"shots": shots, "simulation_config": {"seed": 31}}
    angles = [0.2, 0.9, 1.7] if sweep else [0.9]
    if sweep:
        job = estimator.run_sweep(program, observables, {angle: angles}, **options)
    else:
        job = estimator.run(program, observables, **options)
    assert isinstance(job, Job)
    assert job.status == "DONE"
    results = job.result() if sweep else [job.result()]
    assert len(results) == len(angles)
    for value, result in zip(angles, results, strict=True):
        expected = [1, np.cos(value), np.sin(value)]
        np.testing.assert_allclose(
            result.get_expectation(), expected, atol=0.06 if shots else _ATOL, rtol=0
        )
        if not shots:
            np.testing.assert_array_equal(result.get_standard_error(), [0, 0, 0])


@pytest.mark.parametrize(
    "device_id,caller_device,selected_device", [(None, 1, 0), (1, 0, 1)]
)
def test_selected_device_is_used_and_caller_device_restored(
    gpu_available, device_id, caller_device, selected_device
):
    cp = gpu_available

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("Cross-device execution needs at least two CUDA devices")
    backend = _cuda_simulator(device_id=device_id)
    with cp.cuda.Device(caller_device):
        actual = _state(backend, _ghz(3))
        assert cp.cuda.runtime.getDevice() == caller_device
        assert backend._engine.state.device.id == selected_device
        expected = np.zeros(8, dtype=np.complex128)
        expected[[0, -1]] = 1 / np.sqrt(2)
        np.testing.assert_allclose(actual, expected, atol=_ATOL, rtol=0)
        np.testing.assert_allclose(
            backend._engine.probabilities(), np.abs(expected) ** 2, atol=_ATOL, rtol=0
        )
        assert cp.cuda.runtime.getDevice() == caller_device
        result = backend.run(
            _ghz(3, measure=True),
            shots=1,
            simulation_config={"seed": 71},
            result_config={"counts": True, "final_state": True},
        ).result()
        assert cp.cuda.runtime.getDevice() == caller_device
        assert backend._engine.state.device.id == selected_device
        outcome = next(iter(result.get_counts()))
        collapsed = np.zeros(8, dtype=np.complex128)
        collapsed[int(outcome, 2)] = 1
        np.testing.assert_allclose(
            result.get_statevector(), collapsed, atol=_ATOL, rtol=0
        )


def test_sixty_four_row_sweep_retains_only_owned_host_results(
    gpu_available, record_property
):
    cp = gpu_available

    angle = fq.Parameter("angle")
    program = fq.Program(5)
    program.add(ops.RY(angle), 0)
    angles = np.linspace(-np.pi, np.pi, 64)
    backend = _cuda_simulator()
    pool = cp.cuda.MemoryPool()
    # An isolated allocator measures live allocations, not free blocks that
    # CuPy intentionally caches. The same-shape warmup accounts for the one
    # resident engine state and any fixed allocation rounding.
    with cp.cuda.Device(0), cp.cuda.using_allocator(pool.malloc):
        backend.run_sweep(
            program, {angle: [0.1]}, shots=0, result_config=_STATE_ONLY
        ).result()
        cp.cuda.get_current_stream().synchronize()
        baseline_live_bytes = pool.used_bytes()
        assert baseline_live_bytes > 0
        job = backend.run_sweep(
            program, {angle: angles}, shots=0, result_config=_STATE_ONLY
        )
        assert job.status == "DONE"
        results = job.result()
        cp.cuda.get_current_stream().synchronize()
        after_live_bytes = pool.used_bytes()
        assert after_live_bytes <= baseline_live_bytes
        assert not backend._engine._matrix_cache
    states = [result.get_statevector() for result in results]
    assert len(states) == 64
    for value, state in zip(angles, states, strict=True):
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.complex128
        expected = np.zeros(32, dtype=np.complex128)
        expected[[0, 16]] = [np.cos(value / 2), np.sin(value / 2)]
        np.testing.assert_allclose(state, expected, atol=_ATOL, rtol=0)
    for index, state in enumerate(states):
        assert all(not np.shares_memory(state, prior) for prior in states[:index])
    host_output_bytes = sum(state.nbytes for state in states)
    assert host_output_bytes == 64 * 32 * np.dtype(np.complex128).itemsize
    # The output list legitimately grows with its rows; the resident device
    # state and temporary gate cache must not grow alongside those outputs.
    record_property("sweep_owned_host_output_bytes", host_output_bytes)
    record_property("sweep_baseline_live_device_bytes", baseline_live_bytes)
    record_property("sweep_final_live_device_bytes", after_live_bytes)


def test_gpu_values_do_not_reach_implicit_numpy_coercion(gpu_available, monkeypatch):
    cp = gpu_available

    original_array = np.array
    original_asarray = np.asarray

    def guarded_array(value, *args, **kwargs):
        assert not isinstance(value, cp.ndarray), "Device array passed to np.array"
        return original_array(value, *args, **kwargs)

    def guarded_asarray(value, *args, **kwargs):
        assert not isinstance(value, cp.ndarray), "Device array passed to np.asarray"
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "array", guarded_array)
    monkeypatch.setattr(np, "asarray", guarded_asarray)
    backend = _cuda_simulator()
    result = backend.run(
        _ghz(2, measure=True),
        shots=1,
        simulation_config={"seed": 4},
        result_config={"counts": True, "final_state": True},
    ).result()
    assert isinstance(result.get_statevector(), np.ndarray)
    assert sum(result.get_counts().values()) == 1
    expectation = (
        fq.Estimator(backend).run(_ghz(2), fq.Observable([("XX", 1.0)])).result()
    )
    assert expectation.get_expectation() == pytest.approx(1.0, abs=_ATOL, rel=0)
