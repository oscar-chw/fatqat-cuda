"""Standard CUDA runtime selection, preparation, and execution contracts.

Constructor and failure-path tests run without CuPy or CUDA hardware. Device
execution is limited to the routing and compatibility tests at the end; the
existing GPU behavior and oracle modules cover numerical correctness.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError
from fatqat.job import Job
from fatqat.simulator import Simulator
from fatqat.simulator.experimental import CupySimulator

_ENTRYPOINTS = ("run", "run_sweep", "estimator", "estimator_sweep")


@pytest.fixture(name="forbid_cupy")
def _forbid_cupy(monkeypatch):
    from fatqat.simulator._engine import cupy as cupy_engine_module

    def forbidden_import(*args, **kwargs):
        raise AssertionError("Host preparation attempted to load CuPy")

    monkeypatch.setattr(cupy_engine_module, "import_module", forbidden_import)


@pytest.fixture(scope="module", name="gpu_available")
def _gpu_available():
    cupy = pytest.importorskip("cupy")
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device discovery unavailable: {error}")
    if count == 0:
        pytest.skip("No CUDA device available")
    return cupy


def _request(backend, entrypoint, *, feature=None):
    angle = fq.Parameter("angle")
    sweep = entrypoint.endswith("sweep")
    program = fq.Program(1, 1 if feature in ("condition", "measurement") else 0)
    program.add(ops.RY(angle if sweep else 0.3), 0)
    if feature == "reset":
        program.add(ops.Reset, 0)
    elif feature == "condition":
        program.add(ops.X, 0, condition=(0, 1))
    elif feature == "measurement":
        program.measure(0, 0)
        program.add(ops.H, 0)
    options = {"simulation_config": {"seed": 19}}
    if feature == "workers":
        options["simulation_config"]["max_workers"] = 2
    if entrypoint.startswith("estimator"):
        estimator = fq.Estimator(backend)
        observable = fq.Observable([("Z", 1.0)])
        if sweep:
            return estimator.run_sweep(
                program, observable, {angle: [0.3, 0.7]}, **options
            )
        return estimator.run(program, observable, **options)
    options.update(shots=0, result_config={"counts": False, "final_state": True})
    if sweep:
        return backend.run_sweep(program, {angle: [0.3, 0.7]}, **options)
    return backend.run(program, **options)


@pytest.mark.parametrize(
    "runtime,method,device_id",
    [("cuda", "SV", None), ("CUDA", "sv", 0), ("CuDa", "statevector", 3)],
)
def test_cuda_constructor_is_lazy_and_normalizes_method(
    forbid_cupy, runtime, method, device_id
):
    backend = Simulator(method=method, runtime=runtime, device_id=device_id)
    assert backend.method == "statevector"


@pytest.mark.parametrize(
    "method,canonical",
    [
        ("DM", "density_matrix"),
        ("density_matrix", "density_matrix"),
        ("unitary", "unitary"),
        ("superop", "superop"),
    ],
)
def test_cuda_constructor_selects_other_supported_methods_lazily(
    forbid_cupy, method, canonical
):
    backend = Simulator(method=method, runtime="cuda")
    assert backend.method == canonical


@pytest.mark.parametrize("method", ["mps", "unknown"])
def test_cuda_constructor_still_rejects_unknown_methods(forbid_cupy, method):
    with pytest.raises(BackendValidationError, match="unsupported method"):
        Simulator(method=method, runtime="cuda")


@pytest.mark.parametrize("device_id", [-1, True, False, 0.5, "0", np.int64(0)])
def test_cuda_constructor_rejects_invalid_device_ordinal(forbid_cupy, device_id):
    with pytest.raises(BackendValidationError, match="device_id"):
        Simulator(runtime="cuda", device_id=device_id)


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize("device_id", [0, 1, False])
def test_cpu_constructor_rejects_device_selection(forbid_cupy, runtime, device_id):
    with pytest.raises(BackendValidationError, match="device_id"):
        Simulator(runtime=runtime, device_id=device_id)


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_cpu_runtime_accepts_none_without_loading_cupy(forbid_cupy, runtime):
    result = Simulator(runtime=runtime, device_id=None).run(fq.Program(1)).result()
    assert result.metadata["runtime"] == runtime
    np.testing.assert_array_equal(result.get_statevector(), [1, 0])


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_missing_cupy_is_an_execution_error_job_without_cpu_fallback(
    monkeypatch, entrypoint
):
    from fatqat.simulator._engine import cupy as cupy_engine_module

    missing = ModuleNotFoundError("CuPy deliberately unavailable")

    def missing_import(*args, **kwargs):
        raise missing

    monkeypatch.setattr(cupy_engine_module, "import_module", missing_import)
    backend = Simulator(runtime="cuda")
    job = _request(backend, entrypoint)
    assert isinstance(job, Job)
    assert job.status == "ERROR"
    with pytest.raises(BackendValidationError, match="CuPy") as caught:
        job.result()
    assert caught.value.__cause__ is missing


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_unavailable_device_preserves_execution_error_and_requested_ordinal(
    monkeypatch, entrypoint
):
    from fatqat.simulator._engine import cupy as cupy_engine_module

    failure = RuntimeError("CUDA device deliberately unavailable")
    requested_devices = []

    class UnavailableDevice:
        def __init__(self, device_id):
            requested_devices.append(device_id)

        def __enter__(self):
            raise failure

        def __exit__(self, *args):
            return False

    unavailable_cupy = SimpleNamespace(cuda=SimpleNamespace(Device=UnavailableDevice))
    monkeypatch.setattr(
        cupy_engine_module, "import_module", lambda *_args, **_kwargs: unavailable_cupy
    )
    backend = Simulator(runtime="cuda", device_id=3)
    assert not requested_devices
    job = _request(backend, entrypoint)
    assert isinstance(job, Job)
    assert job.status == "ERROR"
    with pytest.raises(RuntimeError) as caught:
        job.result()
    assert caught.value is failure
    assert requested_devices and set(requested_devices) == {3}


class _PermissiveSubclass(Simulator):
    def _validate_additional_config(self, *, config, simulation, shots, facts):
        # A hardware-profile subclass owns its extra constraints, not the
        # authority to disable restrictions of the selected numerical runtime.
        return None


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
@pytest.mark.parametrize(
    "feature", ["reset", "condition", "measurement", "channel", "workers"]
)
def test_subclass_hook_cannot_bypass_common_cuda_preparation(
    forbid_cupy, entrypoint, feature
):
    noise = None
    if feature == "channel":
        noise = fq.NoiseModel()
        noise.add(fq.noise.Depolarizing(p=0.1), operation=ops.RY)
    backend = _PermissiveSubclass(runtime="cuda", noise=noise)
    # A failed Job would indicate validation happened too late. This must
    # raise directly, despite the subclass hook declining additional checks.
    with pytest.raises(BackendValidationError):
        _request(backend, entrypoint, feature=feature)


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_standard_cuda_entrypoints_report_runtime_and_select_default_device(
    gpu_available, entrypoint
):
    backend = Simulator(method="SV", runtime="cuda", device_id=None)
    job = _request(backend, entrypoint)
    assert job.status == "DONE"
    results = job.result() if entrypoint.endswith("sweep") else [job.result()]
    angles = [0.3, 0.7] if entrypoint.endswith("sweep") else [0.3]
    assert len(results) == len(angles)
    for angle, result in zip(angles, results, strict=True):
        assert result.metadata["backend_name"] == "Simulator"
        assert result.metadata["method"] == "statevector"
        assert result.metadata["runtime"] == "cuda"
        if entrypoint.startswith("estimator"):
            assert result.get_expectation() == pytest.approx(
                np.cos(angle), abs=1e-12, rel=0
            )
        else:
            np.testing.assert_allclose(
                result.get_statevector(),
                [np.cos(angle / 2), np.sin(angle / 2)],
                atol=1e-12,
                rtol=0,
            )
    assert isinstance(backend._engine.state, gpu_available.ndarray)
    assert backend._engine.state.device.id == 0


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_experimental_wrapper_preserves_results_with_canonical_runtime_metadata(
    gpu_available, entrypoint
):
    standard_job = _request(Simulator(runtime="cuda", device_id=0), entrypoint)
    compatible_job = _request(CupySimulator(device_id=0), entrypoint)
    assert standard_job.status == compatible_job.status == "DONE"
    if entrypoint.endswith("sweep"):
        standard_results = standard_job.result()
        compatible_results = compatible_job.result()
    else:
        standard_results = [standard_job.result()]
        compatible_results = [compatible_job.result()]
    for standard, compatible in zip(standard_results, compatible_results, strict=True):
        assert compatible.metadata["runtime"] == "cuda"
        assert compatible.metadata["backend_name"] == "CupySimulator"
        if entrypoint.startswith("estimator"):
            assert compatible.get_expectation() == pytest.approx(
                standard.get_expectation(), abs=1e-12, rel=0
            )
        else:
            np.testing.assert_allclose(
                compatible.get_statevector(),
                standard.get_statevector(),
                atol=1e-12,
                rtol=0,
            )


def test_superconducting_native_gates_use_selected_cuda_device(gpu_available):
    from fatqat.simulator import SCQubitSimulator

    program = fq.Program(3)
    program.add(ops.SX, 0)
    program.add(ops.SX, 2)
    program.add(ops.CZ, (2, 0))
    program.add(ops.RZ(0.31), 0)
    program.add(ops.X, 1)
    profile = {"num_qubits": 3, "couplings": ((0, 2), (2, 1))}
    backend = SCQubitSimulator(runtime="cuda", device_id=0, **profile)
    actual = backend.run(program).result()
    expected = SCQubitSimulator(runtime="numpy", **profile).run(program).result()
    np.testing.assert_allclose(
        actual.get_statevector(), expected.get_statevector(), atol=1e-12, rtol=0
    )
    assert actual.metadata["runtime"] == "cuda"
    assert actual.metadata["backend_name"] == "SCQubitSimulator"
    assert isinstance(backend._engine.state, gpu_available.ndarray)
    assert backend._engine.state.device.id == 0


@pytest.mark.parametrize("runtime", ["cuda", "CUDA"])
def test_atom_array_rejects_cuda_at_construction(forbid_cupy, runtime):
    from fatqat.simulator import AtomArraySimulator

    with pytest.raises(BackendValidationError, match="AtomArraySimulator"):
        AtomArraySimulator(runtime=runtime)


@pytest.mark.parametrize("entrypoint", ["counts", "estimator"])
def test_cuda_classical_readout_confusion_preserves_physical_state(
    gpu_available, entrypoint
):
    from fatqat.noise import ReadoutConfusion

    noise = fq.NoiseModel()
    noise.add(ReadoutConfusion(np.array([[0.0, 1.0], [1.0, 0.0]])), targets=1)
    backend = Simulator(runtime="cuda", noise=noise)
    program = fq.Program(2, 2 if entrypoint == "counts" else 0)
    program.add(ops.X, 0)
    if entrypoint == "counts":
        program.measure_all()
        result = backend.run(
            program,
            shots=1,
            simulation_config={"seed": 15},
            result_config={"counts": True, "final_state": True},
        ).result()
        assert result.get_counts() == {"11": 1}
        np.testing.assert_array_equal(result.get_statevector(), [0, 0, 1, 0])
    else:
        result = (
            fq.Estimator(backend)
            .run(
                program,
                [fq.Observable([("ZI", 1.0)]), fq.Observable([("IZ", 1.0)])],
                shots=32,
                simulation_config={"seed": 15},
            )
            .result()
        )
        np.testing.assert_array_equal(result.get_expectation(), [-1, -1])
        np.testing.assert_array_equal(result.get_standard_error(), [0, 0])


def test_qiskit_adapter_cuda_counts_use_qiskit_bit_order(gpu_available):
    qiskit = pytest.importorskip("qiskit")
    from fatqat.qiskit import FatqatBackend

    circuit = qiskit.QuantumCircuit(3, 3)
    circuit.x(0)
    circuit.h(1)
    circuit.cx(1, 2)
    circuit.s(2)
    circuit.measure([0, 1, 2], [0, 1, 2])
    backend = FatqatBackend(method="SV", runtime="cuda")
    options = {"shots": 2048, "seed_simulator": 61}
    first = backend.run(circuit, **options).result()
    second = backend.run(circuit, **options).result()
    assert first.success
    counts = first.get_counts()
    assert counts == second.get_counts()
    assert set(counts) == {"001", "111"}
    assert sum(counts.values()) == 2048
    assert 0.45 < counts["001"] / 2048 < 0.55
