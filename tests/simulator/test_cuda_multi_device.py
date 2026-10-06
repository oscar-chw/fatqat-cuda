"""CUDA parameter sweeps spread over several devices.

Constructor validation runs without CuPy. Device tests compare a multi-device
sweep with the same sweep on one device by exact equality: every row runs the
same engine code on the same plan, so the results must be bit-identical.
"""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError, MatrixImplementationError
from fatqat.parameters import Parameter
from fatqat.simulator import Simulator


@pytest.mark.parametrize(
    "device_id", [(), (0, 0), (0, -1), (0, 1.0), (0, True), [1, "2"]]
)
def test_invalid_device_lists_are_rejected_at_construction(device_id):
    with pytest.raises(BackendValidationError, match="distinct ordinals"):
        Simulator("statevector", runtime="cuda", device_id=device_id)


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_cpu_runtimes_reject_a_device_list(runtime):
    with pytest.raises(
        BackendValidationError, match="only supported by runtime='cuda'"
    ):
        Simulator("statevector", runtime=runtime, device_id=(0, 1))


def test_a_device_list_is_accepted_without_touching_cuda():
    # Construction validates the list without importing CuPy or touching a
    # device; availability is checked when execution starts.
    Simulator("statevector", runtime="cuda", device_id=[2, 0])


def _device_count():
    cupy = pytest.importorskip("cupy")
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device discovery unavailable: {error}")
    return count


def _sweep_program(n):
    theta = [Parameter(f"t{q}") for q in range(n)]
    program = fq.Program(n, n)
    for q in range(n):
        program.add(ops.RY(theta[q]), q)
    for q in range(n - 1):
        program.add(ops.CX, (q, q + 1))
    for q in range(n):
        program.add(ops.RZ(theta[q]), q)
    program.measure_all()
    return program, theta


@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_multi_device_sweep_is_bit_identical_and_ordered(method):
    count = _device_count()
    if count < 2:
        pytest.skip("needs at least two CUDA devices")
    devices = tuple(range(min(count, 4)))
    n = 6
    program, theta = _sweep_program(n)
    rng = np.random.default_rng(3)
    bindings = {p: rng.uniform(0, np.pi, size=9) for p in theta}
    options = {
        "shots": 500,
        "simulation_config": {"seed": 11},
        "result_config": {"counts": True, "final_state": False},
    }
    single = Simulator(method, runtime="cuda", device_id=0).run_sweep(
        program, bindings, **options
    )
    spread = Simulator(method, runtime="cuda", device_id=devices).run_sweep(
        program, bindings, **options
    )
    single_counts = [r.get_counts() for r in single.result()]
    spread_counts = [r.get_counts() for r in spread.result()]
    assert spread_counts == single_counts
    # Rows really differ, so equal lists also prove the order is kept.
    assert len({tuple(sorted(c.items())) for c in single_counts}) > 1


def test_multi_device_states_are_bit_identical():
    count = _device_count()
    if count < 2:
        pytest.skip("needs at least two CUDA devices")
    n = 12
    theta = Parameter("theta")
    program = fq.Program(n, n)
    for q in range(n):
        program.add(ops.RY(theta), q)
    for q in range(n - 1):
        program.add(ops.CX, (q, q + 1))
    bindings = {theta: np.linspace(0.1, 1.0, 7)}
    request = {"counts": False, "final_state": True}
    single = Simulator("statevector", runtime="cuda").run_sweep(
        program, bindings, shots=0, result_config=request
    )
    spread = Simulator("statevector", runtime="cuda", device_id=(0, 1)).run_sweep(
        program, bindings, shots=0, result_config=request
    )
    for a, b in zip(single.result(), spread.result(), strict=True):
        np.testing.assert_array_equal(a.get_statevector(), b.get_statevector())


def test_a_failing_device_fails_the_sweep_job():
    if _device_count() < 1:
        pytest.skip("No CUDA device available")
    theta = Parameter("theta")
    program = fq.Program(2)
    program.add(ops.RY(theta), 0)
    backend = Simulator("statevector", runtime="cuda", device_id=(0, 4096))
    job = backend.run_sweep(
        program,
        {theta: [0.1, 0.2, 0.3]},
        shots=0,
        result_config={"counts": False, "final_state": True},
    )
    assert job.status == "ERROR"
    with pytest.raises(Exception):
        job.result()


def _two_fake_devices():
    """A Numba backend that takes the multi-device sweep path.

    The device-spreading logic is runtime-independent; giving a CPU backend two
    "devices" (one Numba engine each) exercises it without CUDA hardware.
    """
    from fatqat.simulator._engine.nb import NumbaSVEngine

    backend = Simulator("statevector", runtime="numba")
    backend._device_ids = (0, 1)
    backend._engine_cls = lambda device_id: NumbaSVEngine()
    return backend


def _failing_rule_program(bad):
    """RX whose matrix rule fails for the angles in ``bad``."""
    from fatqat.implementation import default_matrix_implementation_map

    def rule(op):
        if float(op.theta) in bad:
            raise ValueError(f"bad theta {float(op.theta)}")
        c, s = np.cos(op.theta / 2), np.sin(op.theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])

    implementations = default_matrix_implementation_map()
    implementations.add(ops.RX, rule)
    theta = Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RX(theta), 0)
    return program, theta, implementations


def test_multi_device_sweep_raises_the_earliest_failing_row_like_serial():
    program, theta, implementations = _failing_rule_program({1.0, 2.0})
    bindings = {theta: [0.0, 1.0, 2.0, 3.0]}
    request = {"shots": 0, "result_config": {"counts": False, "final_state": True}}
    serial = Simulator(
        "statevector", runtime="numba", implementation_map=implementations
    )
    with pytest.raises(MatrixImplementationError, match=r"bad theta 1\.0"):
        serial.run_sweep(program, bindings, **request)
    spread = _two_fake_devices()
    spread._impl_map = implementations.copy()
    with pytest.raises(MatrixImplementationError, match=r"bad theta 1\.0"):
        spread.run_sweep(program, bindings, **request)


def test_extra_devices_release_their_state_after_a_sweep():
    # Their engines are reused by later sweeps, but holding the last row's
    # state would keep a full state allocated on each extra device.
    theta = Parameter("theta")
    program = fq.Program(2)
    program.add(ops.RY(theta), 0)
    program.add(ops.CX, (0, 1))
    spread = _two_fake_devices()
    request = {"shots": 0, "result_config": {"counts": False, "final_state": True}}
    results = spread.run_sweep(program, {theta: [0.1, 0.2, 0.3]}, **request).result()
    assert len(results) == 3
    assert spread._sweep_engines[1]._state is None
