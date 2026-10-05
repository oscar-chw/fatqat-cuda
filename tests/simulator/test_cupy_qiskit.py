"""Direct GPU cross-validation against independently built Qiskit circuits.

Like tests/test_against_qiskit.py, this uses quantum_info.Statevector without
transpilation and converts Qiskit's least-significant-first basis exactly once
at the comparison boundary. Neither library receives the other's matrices,
lowered plan, or circuit conversion. Amplitudes are compared at atol=1e-12,
rtol=0 without normalization or global-phase alignment.
"""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.simulator import Simulator

_ATOL = 1e-12


def _cuda_simulator():
    return Simulator(method="SV", runtime="cuda")


@pytest.fixture(scope="module", name="qiskit_reference")
def _qiskit_reference():
    qiskit = pytest.importorskip("qiskit")
    quantum_info = pytest.importorskip("qiskit.quantum_info")
    cupy = pytest.importorskip("cupy")
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device discovery unavailable: {error}")
    if count == 0:
        pytest.skip("No CUDA device available")
    return qiskit.QuantumCircuit, quantum_info.Statevector


def _gate_recipe(num_qubits, seed):
    """A seeded gate description, independent of either simulator's objects."""
    rng = np.random.default_rng(seed)
    last = num_qubits - 1
    recipe = [
        ("h", (0,), ()),
        ("cx", (0, last), ()),
        ("p", (last,), (0.37,)),
        ("rx", (0,), (-0.21,)),
        ("cy", (last, 0), ()),
        ("ry", (last,), (0.63,)),
    ]
    fixed_single = ("h", "x", "y", "s", "sdg", "t", "tdg")
    rotated_single = ("rx", "ry", "rz", "p")
    fixed_pair = ("cx", "cy", "cz", "swap")
    gate_count = 20 + seed % 21
    while len(recipe) < gate_count:
        family = len(recipe) % 5
        if family == 0:
            gate = str(rng.choice(fixed_single))
            targets = (int(rng.integers(num_qubits)),)
            parameters = ()
        elif family == 1:
            gate = str(rng.choice(rotated_single))
            targets = (int(rng.integers(num_qubits)),)
            parameters = (float(rng.uniform(-np.pi, np.pi)),)
        elif family == 2:
            gate = str(rng.choice(fixed_pair))
            targets = tuple(int(q) for q in rng.choice(num_qubits, 2, replace=False))
            parameters = ()
        elif family == 3:
            gate = "cp"
            targets = tuple(int(q) for q in rng.choice(num_qubits, 2, replace=False))
            parameters = (float(rng.uniform(-np.pi, np.pi)),)
        else:
            gate = "ccx" if num_qubits >= 3 else "cx"
            width = 3 if num_qubits >= 3 else 2
            targets = tuple(
                int(q) for q in rng.choice(num_qubits, width, replace=False)
            )
            parameters = ()
        recipe.append((gate, targets, parameters))
    return recipe


def _fatqat_program(num_qubits, recipe):
    fixed = {
        "h": ops.H,
        "x": ops.X,
        "y": ops.Y,
        "s": ops.S,
        "sdg": ops.Sdg,
        "t": ops.T,
        "tdg": ops.Tdg,
        "cx": ops.CX,
        "cy": ops.CY,
        "cz": ops.CZ,
        "swap": ops.Swap,
        "ccx": ops.CCX,
    }
    parameterized = {
        "rx": ops.RX,
        "ry": ops.RY,
        "rz": ops.RZ,
        "p": ops.Phase,
        "cp": ops.CPhase,
    }
    program = fq.Program(num_qubits)
    for gate, targets, parameters in recipe:
        operation = parameterized[gate](*parameters) if parameters else fixed[gate]
        program.add(operation, targets[0] if len(targets) == 1 else targets)
    return program


def _qiskit_circuit(num_qubits, recipe, circuit_type):
    circuit = circuit_type(num_qubits)
    for gate, targets, parameters in recipe:
        getattr(circuit, gate)(*parameters, *targets)
    return circuit


def _qiskit_state_in_public_basis(state, num_qubits):
    # Public digit q has place value 2**(n-1-q); Qiskit's same named
    # qubit has place value 2**q. This is a bit permutation, not phase repair.
    indices = [
        sum(((index >> q) & 1) << (num_qubits - 1 - q) for q in range(num_qubits))
        for index in range(1 << num_qubits)
    ]
    return np.asarray(state, dtype=np.complex128)[indices]


@pytest.mark.parametrize("num_qubits", [2, 5, 8], ids=["2q", "5q", "8q"])
@pytest.mark.parametrize("seed", [11, 29, 47, 83])
def test_gpu_public_state_matches_independent_qiskit_circuit(
    qiskit_reference, record_property, num_qubits, seed
):
    circuit_type, statevector_type = qiskit_reference
    recipe = _gate_recipe(num_qubits, seed)
    program = _fatqat_program(num_qubits, recipe)
    circuit = _qiskit_circuit(num_qubits, recipe, circuit_type)
    expected = _qiskit_state_in_public_basis(statevector_type(circuit), num_qubits)
    result = (
        _cuda_simulator()
        .run(program, shots=0, result_config={"counts": False, "final_state": True})
        .result()
    )
    actual = result.get_statevector()
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.complex128
    assert actual.shape == expected.shape == (2**num_qubits,)
    record_property("qiskit_oracle_seed", seed)
    record_property("qiskit_oracle_gate_count", len(recipe))
    record_property(
        "qiskit_oracle_max_amplitude_error", float(np.max(abs(actual - expected)))
    )
    np.testing.assert_allclose(actual, expected, atol=_ATOL, rtol=0)
