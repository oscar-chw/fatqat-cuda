"""CUDA statevector trajectories: channels, reset, measurement, feedforward.

The device engine draws every random choice from the shot's host seed stream,
in the order the NumPy engine draws it, so one seed selects the same branches
on both. Counts are therefore compared exactly, not statistically: a draw
could only differ if a branch boundary fell within round-off of a uniform.
"""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.noise import (
    AmplitudeDamping,
    Depolarizing,
    NoiseModel,
    PhaseDamping,
    TransitionRelaxation,
)
from fatqat.simulator import Simulator


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


def _qubit_case():
    # Depolarizing takes the scaled-unitary sampler and amplitude damping the
    # general quantum-jump path, so both channel routes run on the device.
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=ops.H)
    noise.add(AmplitudeDamping(p=0.3), operation=ops.RY)
    program = fq.Program(3, 3)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.measure(0, 0)
    program.add(ops.X, 2, condition=(0, 1))
    program.add(ops.Reset, 1)
    program.add(ops.RY(0.7), 1)
    program.add(ops.H, 2)
    program.measure((1, 2), (1, 2))
    return program, noise


def _qutrit_case():
    noise = NoiseModel()
    noise.add(
        TransitionRelaxation(p=0.2, coefficients={(1, 0): 1, (2, 1): 1}),
        operation=ops.Shift,
    )
    noise.add(PhaseDamping(p=0.15), operation=ops.Shift)
    qreg = fq.QuantumRegister(2, dim=3)
    creg = fq.ClassicalRegister(2, dim=3)
    program = fq.Program([qreg], [creg])
    program.add(ops.Shift(1), qreg[0])
    program.measure(qreg[0], creg[0])
    program.add(ops.Shift(1), qreg[1], condition=(creg[0], 1))
    program.add(ops.Reset, qreg[0])
    program.add(ops.Shift(2), qreg[0])
    program.measure(qreg[1], creg[1])
    return program, noise


@pytest.mark.parametrize("case", [_qubit_case, _qutrit_case])
def test_cuda_trajectory_counts_equal_numpy_for_the_same_seed(gpu_available, case):
    program, noise = case()
    counts = {
        runtime: Simulator("SV", runtime=runtime, noise=noise)
        .run(program, shots=2000, simulation_config={"seed": 23})
        .result()
        .get_counts()
        for runtime in ("numpy", "cuda")
    }
    # Several outcomes, so a sampler stuck on one branch cannot match.
    assert len(counts["numpy"]) >= 3
    assert counts["cuda"] == counts["numpy"]


@pytest.mark.parametrize("seed", range(6))
def test_cuda_single_trajectory_state_matches_numpy(gpu_available, seed):
    program, noise = _qubit_case()
    states = [
        Simulator("SV", runtime=runtime, noise=noise)
        .run(
            program,
            shots=1,
            result_config={"counts": True, "final_state": True},
            simulation_config={"seed": seed},
        )
        .result()
        .get_statevector()
        for runtime in ("numpy", "cuda")
    ]
    np.testing.assert_allclose(states[1], states[0], rtol=0, atol=1e-12)


def test_cuda_condition_without_measurement_runs_on_the_device(gpu_available):
    # A clbit that is never written reads 0, so the conditioned X never fires.
    program = fq.Program(1, 1)
    program.add(ops.RY(0.3), 0)
    program.add(ops.X, 0, condition=(0, 1))
    states = [
        Simulator(runtime=runtime)
        .run(program, shots=0, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
        for runtime in ("numpy", "cuda")
    ]
    np.testing.assert_allclose(states[1], states[0], rtol=0, atol=1e-15)
