"""CUDA sweeps and shot runs spread over several devices.

Constructor validation runs without CuPy. Device tests compare a multi-device
run with the same run on one device by exact equality: every row or shot runs
the same engine code on the same plan and seed, so the results must be
bit-identical.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import (
    BackendExecutionError,
    BackendValidationError,
    MatrixImplementationError,
)
from fatqat.parameters import Parameter
from fatqat.noise import AmplitudeDamping, Depolarizing, NoiseModel
from fatqat.simulator import Simulator
from fatqat.simulator._engine.np import NumpyDMEngine, NumpySVEngine


@pytest.mark.parametrize(
    "device_id", [(), (0, 0), (0, -1), (0, 1.0), (0, True), [1, "2"], "gpu", "ALL"]
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
    Simulator("statevector", runtime="cuda", device_id="all")


class _FakeCudaRuntimeError(RuntimeError):
    pass


def _fake_cupy(count):
    """A CuPy stand-in whose device count is ``count``; ``None`` raises, as
    CUDA does when there is no driver or no device."""

    def device_count():
        if count is None:
            raise _FakeCudaRuntimeError("cudaErrorNoDevice")
        return count

    runtime = SimpleNamespace(
        getDeviceCount=device_count, CUDARuntimeError=_FakeCudaRuntimeError
    )
    return SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))


def test_all_devices_resolve_to_every_visible_device_once():
    backend = Simulator("statevector", runtime="cuda", device_id="all")
    backend._engine.__dict__["_cp"] = _fake_cupy(3)
    assert backend._devices() == (0, 1, 2)
    # Resolved once: a later change in the count does not move the devices.
    backend._engine.__dict__["_cp"] = _fake_cupy(1)
    assert backend._devices() == (0, 1, 2)


@pytest.mark.parametrize("count", [0, None], ids=["zero", "raises"])
def test_all_devices_with_none_visible_fail_the_job(count):
    backend = Simulator("statevector", runtime="cuda", device_id="all")
    backend._engine.__dict__["_cp"] = _fake_cupy(count)
    program = fq.Program(1, 1)
    program.measure(0, 0)
    job = backend.run(program, shots=4)
    assert job.status == "ERROR"
    with pytest.raises(BackendValidationError, match="no CUDA device") as caught:
        job.result()
    if count is None:  # the driver's own message is kept, not hidden
        assert "cudaErrorNoDevice" in str(caught.value)
        assert isinstance(caught.value.__cause__, _FakeCudaRuntimeError)


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
    assert spread._device_engines[1]._state is None


def _trajectory_program():
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=ops.H)
    noise.add(AmplitudeDamping(p=0.3), operation=ops.RY)
    theta = Parameter("theta")
    program = fq.Program(3, 3)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.measure(0, 0)
    program.add(ops.X, 2, condition=(0, 1))
    program.add(ops.Reset, 1)
    program.add(ops.RY(theta), 1)
    program.add(ops.H, 2)
    program.measure((1, 2), (1, 2))
    return program, theta, noise


class _FakeDevice:
    """A NumPy engine standing in for one GPU (picklable: GPU workers get it).

    With ``FATQAT_FAKE_DEVICE_LOG`` set, each batch it runs leaves a file
    naming its process and device. The devices listed (comma-separated) in
    ``FATQAT_FAKE_LOST_DEVICE`` raise; those in ``FATQAT_FAKE_CRASH_DEVICE``
    kill their worker process outright (never list device 0: it runs here).
    """

    def __init__(self, device_id=0):
        super().__init__()
        self.device_id = device_id

    def execute_shot_batch(self, context, payload, seed_batch, policy):
        if str(self.device_id) in os.environ.get("FATQAT_FAKE_CRASH_DEVICE", "").split(
            ","
        ):
            os._exit(1)
        if str(self.device_id) in os.environ.get("FATQAT_FAKE_LOST_DEVICE", "").split(
            ","
        ):
            raise RuntimeError(f"device {self.device_id} lost")
        log = os.environ.get("FATQAT_FAKE_DEVICE_LOG")
        if log:
            name = f"{os.getpid()}-{self.device_id}-{len(seed_batch)}"
            Path(log, name).touch()
        return super().execute_shot_batch(context, payload, seed_batch, policy)


class _FakeDeviceSV(_FakeDevice, NumpySVEngine):
    pass


class _FakeDeviceDM(_FakeDevice, NumpyDMEngine):
    pass


@pytest.fixture(name="device_log")
def _device_log(tmp_path, monkeypatch):
    """Fresh GPU workers that log each batch they run into ``tmp_path``."""
    from fatqat.simulator._engine import parallel

    monkeypatch.setenv("FATQAT_FAKE_DEVICE_LOG", str(tmp_path))
    monkeypatch.setattr(parallel, "_DEVICE_EXECUTORS", {})
    yield tmp_path
    for executor in parallel._DEVICE_EXECUTORS.values():
        executor.shutdown(wait=True)


def _logged(log):
    """(process, device, batch size) of every batch the fake devices ran."""
    return [tuple(int(x) for x in path.name.split("-")) for path in log.iterdir()]


def _fake_devices(count, noise, method="statevector"):
    """A NumPy backend with ``count`` fake devices, one engine each.

    Shot spreading is runtime-independent, so this exercises it, worker
    processes included, without CUDA hardware.
    """
    cls = _FakeDeviceSV if method == "statevector" else _FakeDeviceDM
    backend = Simulator(method, runtime="numpy", noise=noise)
    backend._device_ids = tuple(range(count))
    backend._engine_cls = cls
    backend._engine = cls(device_id=0)
    return backend


@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
@pytest.mark.parametrize("shots", [2, 7, 1000])
def test_shots_spread_over_devices_give_one_device_counts(device_log, method, shots):
    program, theta, noise = _trajectory_program()
    program = program.assign_parameters({theta: 0.7})
    options = {"shots": shots, "simulation_config": {"seed": 29}}
    single = Simulator(method, runtime="numpy", noise=noise)
    expected = single.run(program, **options).result().get_counts()
    spread = _fake_devices(3, noise, method)
    assert spread.run(program, **options).result().get_counts() == expected
    # The shots really were split, in batches as even as the count allows,
    # one device each, the further devices' in other processes.
    batches = _logged(device_log)
    sizes = sorted(size for _, _, size in batches)
    assert sorted(device for _, device, _ in batches) == list(range(min(3, shots)))
    assert sum(sizes) == shots and sizes[-1] - sizes[0] <= 1
    assert {pid for pid, device, _ in batches if device} - {os.getpid()} == {
        pid for pid, device, _ in batches if device
    }


def test_a_gpu_worker_frees_its_state_after_each_batch():
    from fatqat.simulator._engine import parallel

    program, theta, noise = _trajectory_program()
    program = program.assign_parameters({theta: 0.7})
    backend = Simulator("statevector", runtime="numpy", noise=noise)
    plan, _facts = backend._lower_program(program)
    engine = backend._engine
    payload = engine.materialize_execution(
        plan,
        system_dims=(2, 2, 2),
        n_clbits=3,
        deferred_measurements=(),
        policy=parallel._SERIAL,
    )
    from fatqat._backends.engine_contract import _StateVectorResultRequest
    from fatqat.simulator._execution_contract import _ExecutionContext

    context = _ExecutionContext(
        "per_shot",
        _StateVectorResultRequest(True, False),
        (2, 2, 2),
        3,
        4,
        1,
        None,
        None,
    )
    seeds = np.random.SeedSequence(1).spawn(4)
    rows = parallel._run_device_batch(_FakeDeviceSV, 5, context, payload, seeds)
    assert len(rows) == 4
    worker_engine = parallel._DEVICE_ENGINES.pop((_FakeDeviceSV, 5))
    assert worker_engine.device_id == 5 and worker_engine._state is None


def test_sweep_rows_on_devices_do_not_also_spread_their_shots(device_log):
    program, theta, noise = _trajectory_program()
    bindings = {theta: [0.1, 0.5, 0.9, 1.3]}
    # Serial shots: the CPU's own process workers, which auto may choose on
    # a many-core machine, also build the fake engine and would log batches.
    options = {
        "shots": 300,
        "simulation_config": {"seed": 31, "shot_parallelism": "serial"},
    }
    single = Simulator("statevector", runtime="numpy", noise=noise)
    expected = [
        r.get_counts() for r in single.run_sweep(program, bindings, **options).result()
    ]
    spread = _fake_devices(2, noise)
    got = [
        r.get_counts() for r in spread.run_sweep(program, bindings, **options).result()
    ]
    assert got == expected
    # Each row ran its shots on its own device's engine, by the one-device
    # route; spreading them as well would have logged shot batches.
    assert not _logged(device_log)


def test_trajectories_on_every_device_equal_one_device(monkeypatch):
    from fatqat.simulator import simulator as simulator_module

    count = _device_count()
    if count < 2:
        pytest.skip("needs at least two CUDA devices")
    program, theta, noise = _trajectory_program()
    program = program.assign_parameters({theta: 0.7})
    options = {"shots": 3000, "simulation_config": {"seed": 37}}
    used = []
    shipped = simulator_module._run_shots_on_device_workers

    def spy(engine_cls, devices, context, payload, batches, run_local):
        used.append(tuple(devices[: len(batches)]))
        return shipped(engine_cls, devices, context, payload, batches, run_local)

    monkeypatch.setattr(simulator_module, "_run_shots_on_device_workers", spy)
    counts = [
        Simulator("statevector", runtime="cuda", device_id=device_id, noise=noise)
        .run(program, **options)
        .result()
        .get_counts()
        for device_id in (0, "all")
    ]
    assert counts[1] == counts[0]
    # "all" really used every device, not just the first.
    assert used == [tuple(range(count))]
    assert len(counts[0]) >= 4


def test_a_device_failing_mid_shots_fails_the_job(device_log, monkeypatch):
    program, theta, noise = _trajectory_program()
    program = program.assign_parameters({theta: 0.7})
    monkeypatch.setenv("FATQAT_FAKE_LOST_DEVICE", "2")
    spread = _fake_devices(3, noise)
    job = spread.run(program, shots=30, simulation_config={"seed": 41})
    assert job.status == "ERROR"
    with pytest.raises(RuntimeError, match="device 2 lost") as caught:
        job.result()
    assert "in the GPU worker for CUDA device 2" in caught.value.__notes__
    # The other devices' batches still ran to the end before the job failed.
    assert sorted(device for _, device, _ in _logged(device_log)) == [0, 1]


@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_a_requested_final_state_keeps_shots_on_one_device(method):
    # A conditioned gate without a measurement is per-shot yet deterministic,
    # so its final state may be requested with many shots; spreading the
    # shots would drop that state.
    program = fq.Program(2, 1)
    program.add(ops.H, 0)
    program.add(ops.X, 1, condition=(0, 0))
    options = {
        "shots": 50,
        "result_config": {"counts": True, "final_state": True},
        "simulation_config": {"seed": 3},
    }
    single = Simulator(method, runtime="numpy").run(program, **options).result()
    spread = _fake_devices(2, None, method)
    result = spread.run(program, **options).result()
    getter = "get_statevector" if method == "statevector" else "get_density_matrix"
    expected = getattr(single, getter)()
    assert expected is not None
    np.testing.assert_array_equal(getattr(result, getter)(), expected)
    assert result.get_counts() == single.get_counts()


def test_the_first_devices_error_wins_and_every_batch_finishes(device_log, monkeypatch):
    program, theta, noise = _trajectory_program()
    program = program.assign_parameters({theta: 0.7})
    monkeypatch.setenv("FATQAT_FAKE_LOST_DEVICE", "0,2")
    spread = _fake_devices(3, noise)
    job = spread.run(program, shots=30, simulation_config={"seed": 41})
    with pytest.raises(RuntimeError, match="device 0 lost") as caught:
        job.result()
    assert "in the shot batch for CUDA device 0" in caught.value.__notes__
    # Device 1's batch ran to the end although device 0 had already failed.
    assert [device for _, device, _ in _logged(device_log)] == [1]


def test_a_crashed_gpu_worker_fails_its_run_and_the_next_run_recovers(
    device_log, monkeypatch
):
    program, theta, noise = _trajectory_program()
    program = program.assign_parameters({theta: 0.7})
    options = {"shots": 30, "simulation_config": {"seed": 43}}
    expected = (
        Simulator("statevector", runtime="numpy", noise=noise)
        .run(program, **options)
        .result()
        .get_counts()
    )
    spread = _fake_devices(3, noise)
    monkeypatch.setenv("FATQAT_FAKE_CRASH_DEVICE", "1")
    job = spread.run(program, **options)
    with pytest.raises(
        BackendExecutionError, match="GPU worker for CUDA device 1 exited"
    ) as caught:
        job.result()
    assert caught.value.__cause__ is not None  # the pool's own error, chained
    # A fresh worker replaces the dead one; the crash flag is gone by then.
    monkeypatch.delenv("FATQAT_FAKE_CRASH_DEVICE")
    assert spread.run(program, **options).result().get_counts() == expected


def test_each_device_keeps_one_worker_across_runs(device_log):
    program, theta, noise = _trajectory_program()
    program = program.assign_parameters({theta: 0.7})
    spread = _fake_devices(4, noise)
    for seed in range(4):
        spread.run(program, shots=40, simulation_config={"seed": seed}).result()
    workers = {}
    for pid, device, _ in _logged(device_log):
        workers.setdefault(device, set()).add(pid)
    assert sorted(workers) == [0, 1, 2, 3]
    assert all(len(pids) == 1 for pids in workers.values())
    assert len({pid for pids in workers.values() for pid in pids}) == 4


def test_spreading_and_device_choice_are_logged(device_log, caplog):
    import logging

    program, theta, noise = _trajectory_program()
    program = program.assign_parameters({theta: 0.7})
    spread = _fake_devices(3, noise)
    with caplog.at_level(logging.DEBUG, logger="fatqat"):
        spread.run(program, shots=30, simulation_config={"seed": 1}).result()
    assert (
        "spreading 30 shots over CUDA devices (0, 1, 2) in batches of [10, 10, 10]"
        in caplog.messages
    )
    assert any(
        m.startswith("shot branching, _FakeDeviceSV: 10 shots") for m in caplog.messages
    )


def test_threads_starting_runs_at_once_share_one_worker_per_device(monkeypatch):
    # Without the lock, threads racing on first use each started a worker
    # process (and, on a GPU, a CUDA context) for the same device.
    import threading
    import time

    from fatqat.simulator._engine import parallel

    monkeypatch.setattr(parallel, "_DEVICE_EXECUTORS", {})
    real_get = dict.get
    barrier = threading.Barrier(8)

    class Slow(dict):
        def get(self, key, default=None):  # widen the race window
            value = real_get(self, key, default)
            time.sleep(0.01)
            return value

    monkeypatch.setattr(parallel, "_DEVICE_EXECUTORS", Slow())
    got = []

    def start():
        barrier.wait()
        got.append(parallel._device_executor(1))

    threads = [threading.Thread(target=start) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert len({id(executor) for executor in got}) == 1
    finally:
        for executor in {id(e): e for e in got}.values():
            executor.shutdown(wait=True)
