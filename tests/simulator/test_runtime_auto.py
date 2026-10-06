"""runtime="auto": the runtime follows the hardware found and the state's size.

The choice is tested with stand-in hardware (the device lookups patched), so
it runs everywhere; a run on a real Apple GPU checks the Metal path end to end.
"""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError
from fatqat.simulator import Simulator
from fatqat.simulator import simulator as module
from fatqat.simulator.fake_atom_array import AtomArraySimulator

pytest.importorskip("numba")


def _hardware(monkeypatch, *, gpus=0, metal=False):
    monkeypatch.setattr(module, "_cuda_device_count", lambda: gpus)
    monkeypatch.setattr(module, "_metal_available", lambda: metal)


def _program(n):
    program = fq.Program(n, n)
    for q in range(n):
        program.add(ops.H, q)
    program.measure_all()
    return program


def test_small_states_stay_on_numba_whatever_the_hardware(monkeypatch):
    _hardware(monkeypatch, gpus=4, metal=True)
    backend = Simulator("statevector", runtime="auto")
    result = backend.run(_program(3), shots=20, simulation_config={"seed": 2})
    expected = Simulator("statevector", runtime="numba").run(
        _program(3), shots=20, simulation_config={"seed": 2}
    )
    assert result.result().metadata["runtime"] == "numba"
    assert result.result().get_counts() == expected.result().get_counts()


def test_a_large_state_goes_to_every_gpu_and_a_small_one_comes_back(monkeypatch):
    _hardware(monkeypatch, gpus=2, metal=True)
    backend = Simulator("statevector", runtime="auto")
    numba_engine = backend._engine
    assert backend._choose_runtime((2,) * 14)
    assert backend._runtime == "cuda"
    assert type(backend._engine).__name__ == "CupySVEngine"
    assert backend._all_devices  # resolved to every visible device on first run
    cuda_engine = backend._engine
    assert backend._choose_runtime((2,) * 13)
    assert backend._runtime == "numba" and backend._engine is numba_engine
    # Engines are kept, not rebuilt, so their caches survive the switch.
    assert backend._choose_runtime((2,) * 21)
    assert backend._engine is cuda_engine
    assert not backend._choose_runtime((2,) * 22)


def test_a_density_matrix_counts_its_squared_size(monkeypatch):
    _hardware(monkeypatch, gpus=1)
    backend = Simulator("density_matrix", runtime="auto")
    assert not backend._choose_runtime((2,) * 6)  # 2**12 amplitudes
    assert backend._choose_runtime((2,) * 7)  # 2**14
    assert backend._runtime == "cuda"


def test_without_a_gpu_statevectors_use_metal_and_other_methods_numba(monkeypatch):
    _hardware(monkeypatch, metal=True)
    sv = Simulator("statevector", runtime="auto")
    assert sv._choose_runtime((2,) * 21) and sv._runtime == "metal"
    assert sv._choose_runtime((2,) * 20) and sv._runtime == "numba"
    dm = Simulator("density_matrix", runtime="auto")
    assert not dm._choose_runtime((2,) * 12)
    assert dm._runtime == "numba"


def test_atom_arrays_choose_numba_only(monkeypatch):
    _hardware(monkeypatch, gpus=4, metal=True)
    backend = AtomArraySimulator(method="statevector", runtime="auto")
    assert not backend._choose_runtime((2,) * 24)
    for runtime in ("cuda", "metal"):
        with pytest.raises(BackendValidationError, match="does not support runtime"):
            AtomArraySimulator(method="statevector", runtime=runtime)


def test_auto_takes_no_device_id():
    with pytest.raises(BackendValidationError, match="takes no device_id"):
        Simulator("statevector", runtime="auto", device_id=0)


def test_the_metal_threshold_matches_the_engine():
    from fatqat.simulator._engine import metal

    assert module._METAL_MIN_BYTES == metal.MetalSVEngine._TILE_MIN_BYTES


def test_auto_sweeps_choose_too(monkeypatch):
    from fatqat.parameters import Parameter

    _hardware(monkeypatch)
    theta = Parameter("theta")
    program = fq.Program(2)
    program.add(ops.RY(theta), 0)
    rows = Simulator("statevector", runtime="auto").run_sweep(
        program,
        {theta: [0.1, 0.2]},
        shots=0,
        result_config={"counts": False, "final_state": True},
    )
    assert [r.metadata["runtime"] for r in rows.result()] == ["numba", "numba"]


@pytest.mark.skipif(not module._metal_available(), reason="needs an Apple GPU")
def test_auto_runs_a_large_statevector_on_metal(monkeypatch):
    from fatqat.simulator._engine import metal

    dispatches = []
    shipped = metal.MetalSVEngine._encode

    def encode(self, *args):
        dispatches.append(args[-1])
        return shipped(self, *args)

    monkeypatch.setattr(metal.MetalSVEngine, "_encode", encode)
    n = 21  # 32 MiB: above the Metal threshold
    program = fq.Program(n)
    for q in range(n):
        program.add(ops.H, q)
        program.add(ops.T, q)
    request = {"counts": False, "final_state": True}
    auto = Simulator("statevector", runtime="auto").run(
        program, shots=0, result_config=request
    )
    numba = Simulator("statevector", runtime="numba").run(
        program, shots=0, result_config=request
    )
    assert auto.result().metadata["runtime"] == "metal"
    assert dispatches, "the GPU took no tiles"
    np.testing.assert_allclose(
        auto.result().get_statevector(),
        numba.result().get_statevector(),
        atol=1e-15,
        rtol=0,
    )


def test_a_small_run_after_a_large_one_is_validated_on_its_own_engine(monkeypatch):
    _hardware(monkeypatch, gpus=1)
    backend = Simulator("statevector", runtime="auto")
    backend.run(_program(3), shots=4).result()
    numba_engine = backend._engine
    assert backend._choose_runtime((2,) * 16)  # as a large run would leave it
    assert numba_engine._state is None  # the engine left behind drops its state
    result = backend.run(
        _program(3),
        shots=4,
        simulation_config={
            "kernel_parallelism": "threads",
            "shot_parallelism": "serial",
        },
    )
    assert result.result().metadata["runtime"] == "numba"


def test_every_entry_point_chooses_its_runtime(monkeypatch):
    from fatqat.parameters import Parameter

    seen = []
    shipped = Simulator._choose_runtime

    def spy(self, system_dims):
        seen.append(tuple(system_dims))
        return shipped(self, system_dims)

    monkeypatch.setattr(Simulator, "_choose_runtime", spy)
    _hardware(monkeypatch)
    backend = Simulator("statevector", runtime="auto")
    backend.run(_program(2), shots=4).result()
    theta = Parameter("theta")
    swept = fq.Program(3)
    swept.add(ops.RY(theta), 0)
    backend.run_sweep(swept, {theta: [0.1]}, shots=0).result()
    fq.Estimator(backend).run(
        _no_measure(4),
        fq.Observable([("ZIII", 1.0)]),
        shots=0,
    ).result()
    assert seen == [(2, 2), (2, 2, 2), (2, 2, 2, 2)]
    # A backend that is not "auto" never chooses.
    seen.clear()
    Simulator("statevector", runtime="numba").run(_program(2), shots=4).result()
    assert not seen


def _no_measure(n):
    program = fq.Program(n)
    for q in range(n):
        program.add(ops.H, q)
    return program
