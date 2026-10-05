"""CUDA density and operator methods against explicit, independent references.

Public basis digits are enumerated directly, including ordered mixed-radix
targets. The high-precision case embeds the stored complex128 coefficients
exactly and retains the existing 1e-12 and each-CPU-plus-eight-epsilon gates.
"""

from itertools import product
import json
from math import prod

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.steps import ApplyChannelStep, ApplyMatrixStep
from fatqat.implementation import MatrixImplementationMap
from fatqat.noise import Channel, ChannelImplementationMap
from fatqat.simulator import Simulator
from tests.simulator.test_cupy_accuracy import _ATOL, _EQUIVALENCE_FLOOR, _exact_complex

_STATE_ONLY = {"counts": False, "final_state": True}
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_CX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)


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


def _index(digits, dims):
    index = 0
    for digit, dim in zip(digits, dims, strict=True):
        index = index * dim + digit
    return index


def _embed(local, targets, dims):
    """Construct a dense operator from the definition of public basis digits."""
    basis = list(product(*(range(dim) for dim in dims)))
    spectators = tuple(q for q in range(len(dims)) if q not in targets)
    target_dims = tuple(dims[q] for q in targets)
    full = np.zeros((prod(dims), prod(dims)), dtype=np.complex128)
    for row, output in enumerate(basis):
        local_row = _index(tuple(output[q] for q in targets), target_dims)
        for column, source in enumerate(basis):
            if all(output[q] == source[q] for q in spectators):
                local_column = _index(tuple(source[q] for q in targets), target_dims)
                full[row, column] = local[local_row, local_column]
    return full


def _reset_reference(rho, targets, dims):
    """Trace matching reset digits and reprepare them at zero, entry by entry."""
    basis = list(product(*(range(dim) for dim in dims)))
    output = np.zeros_like(rho)
    for row, row_digits in enumerate(basis):
        out_row = _index(
            tuple(0 if q in targets else digit for q, digit in enumerate(row_digits)),
            dims,
        )
        for column, column_digits in enumerate(basis):
            if all(row_digits[q] == column_digits[q] for q in targets):
                out_column = _index(
                    tuple(
                        0 if q in targets else digit
                        for q, digit in enumerate(column_digits)
                    ),
                    dims,
                )
                output[out_row, out_column] += rho[row, column]
    return output


def _unitary(rng, size):
    matrix, _ = np.linalg.qr(
        rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    )
    return np.asarray(matrix, dtype=np.complex128)


def _physical_input(size, seed=57):
    rng = np.random.default_rng(seed)
    kets = []
    for _ in range(2):
        ket = rng.normal(size=size) + 1j * rng.normal(size=size)
        kets.append(np.asarray(ket / np.linalg.norm(ket), dtype=np.complex128))
    rho = 0.6 * np.outer(kets[0], kets[0].conj())
    rho += 0.4 * np.outer(kets[1], kets[1].conj())
    return kets[0], rho


def _assert_physical(rho):
    np.testing.assert_allclose(np.trace(rho), 1, atol=_ATOL, rtol=0)
    np.testing.assert_allclose(rho, rho.conj().T, atol=_ATOL, rtol=0)
    assert np.linalg.eigvalsh(rho).min() >= -_ATOL


def _matrix_result(backend, program, initial_state=None):
    result = backend.run(
        program, shots=0, initial_state=initial_state, result_config=_STATE_ONLY
    ).result()
    accessor = {
        "density_matrix": result.get_density_matrix,
        "unitary": result.get_unitary,
        "superop": result.get_superop,
    }[backend.method]
    matrix = accessor()
    assert isinstance(matrix, np.ndarray)
    assert matrix.dtype == np.complex128
    assert np.all(np.isfinite(matrix))
    assert result.metadata["method"] == backend.method
    return matrix


def _assert_cpu_parity(actual, method, program, initial_state=None, **backend_options):
    for runtime in ("numpy", "numba"):
        if runtime == "numba":
            pytest.importorskip("numba")
        reference = _matrix_result(
            Simulator(method, runtime=runtime, **backend_options),
            program,
            initial_state,
        )
        np.testing.assert_allclose(actual, reference, atol=_ATOL, rtol=0)


def _ordered_program(dims):
    class OuterPair(ops.Operation):
        name = "CudaMatrixOuterPair"
        num_subsystems = 2

    class Middle(ops.Operation):
        name = "CudaMatrixMiddle"
        num_subsystems = 1

    class AdjacentPair(ops.Operation):
        name = "CudaMatrixAdjacentPair"
        num_subsystems = 2

    registers = [fq.QuantumRegister(1, dim=dim) for dim in dims]
    program = fq.Program(registers)
    rng = np.random.default_rng(5280)
    implementations = MatrixImplementationMap()
    expected = np.eye(prod(dims), dtype=np.complex128)
    for gate, targets in ((OuterPair, (2, 0)), (Middle, (1,)), (AdjacentPair, (1, 2))):
        matrix = _unitary(rng, prod(dims[q] for q in targets))
        implementations.add(gate, matrix)
        program.add(gate(), tuple(registers[q][0] for q in targets))
        expected = _embed(matrix, targets, dims) @ expected
    return program, implementations, expected


@pytest.mark.parametrize("method", ["DM", "unitary", "superop"])
@pytest.mark.parametrize("dims", [(2, 2, 2), (2, 3, 2)], ids=["qubits", "mixed_radix"])
@pytest.mark.parametrize("device_id", [0, 1])
def test_public_matrix_methods_preserve_order_dtype_and_device(
    gpu_available, method, dims, device_id
):
    cp = gpu_available
    if cp.cuda.runtime.getDeviceCount() <= device_id:
        pytest.skip("Requested CUDA device unavailable")
    program, implementations, unitary = _ordered_program(dims)
    ket, rho = _physical_input(prod(dims))
    original = rho.copy()
    backend = Simulator(
        method, runtime="cuda", device_id=device_id, implementation_map=implementations
    )
    with cp.cuda.Device(0):
        actual = _matrix_result(backend, program, rho if method == "DM" else None)
        assert cp.cuda.runtime.getDevice() == 0
        assert backend._engine.state.device.id == device_id
    evolved_rho = unitary @ rho @ unitary.conj().T
    if method == "DM":
        expected = evolved_rho
        _assert_physical(actual)
        np.testing.assert_array_equal(rho, original)
    elif method == "unitary":
        expected = unitary
        np.testing.assert_allclose(actual @ ket, unitary @ ket, atol=_ATOL, rtol=0)
        np.testing.assert_allclose(
            actual.conj().T @ actual, np.eye(len(ket)), atol=_ATOL, rtol=0
        )
    else:
        expected = np.kron(unitary.conj(), unitary)
        evolved = (actual @ rho.reshape(-1, order="F")).reshape(rho.shape, order="F")
        np.testing.assert_allclose(evolved, evolved_rho, atol=_ATOL, rtol=0)
        _assert_physical(evolved)
    np.testing.assert_allclose(actual, expected, atol=_ATOL, rtol=0)
    _assert_cpu_parity(
        actual,
        method,
        program,
        rho if method == "DM" else None,
        implementation_map=implementations,
    )


@pytest.mark.parametrize("input_kind", ["ket", "density"])
def test_density_engine_owns_input_and_every_export(gpu_available, input_kind):
    from fatqat.simulator._engine.cupy import CupyDMEngine

    ket, rho = _physical_input(4)
    source = ket.copy() if input_kind == "ket" else rho.copy()
    expected = np.outer(ket, ket.conj()) if input_kind == "ket" else rho.copy()
    with gpu_available.cuda.Device(0):
        engine = CupyDMEngine(device_id=0)
        engine.initialize((2, 2), initial_state=source)
        source[...] = 0
        np.testing.assert_allclose(engine.export_state(), expected, atol=_ATOL, rtol=0)
        engine.apply(ApplyMatrixStep(_X, (1,)))
        full = np.kron(_X, np.eye(2))
        expected = full @ expected @ full.conj().T
        exported = engine.export_state()
        np.testing.assert_allclose(exported, expected, atol=_ATOL, rtol=0)
        exported[...] = 0
        np.testing.assert_allclose(engine.export_state(), expected, atol=_ATOL, rtol=0)


def test_density_preserves_accepted_unnormalized_nonhermitian_input(gpu_available):
    program = fq.Program(2)
    program.add(ops.S, 0)
    program.add(ops.CX, (1, 0))
    rng = np.random.default_rng(7)
    source = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    original = source.copy()
    unitary = _embed(_CX, (1, 0), (2, 2)) @ np.kron(np.diag([1, 1j]), np.eye(2))
    actual = _matrix_result(Simulator("DM", runtime="cuda"), program, source)
    np.testing.assert_allclose(
        actual, unitary @ original @ unitary.conj().T, atol=_ATOL, rtol=0
    )
    np.testing.assert_array_equal(source, original)
    _assert_cpu_parity(actual, "DM", program, source)


@pytest.mark.parametrize("method", ["DM", "superop"])
def test_amplitude_and_phase_damping_follow_analytic_channel(gpu_available, method):
    probability, dephasing = 0.31, 0.27
    program = fq.Program(1)
    program.add(ops.I, 0)
    noise = fq.NoiseModel()
    noise.add(fq.noise.AmplitudeDamping(p=probability), operation=ops.I)
    noise.add(fq.noise.PhaseDamping(p=dephasing), operation=ops.I)
    rho = np.array([[0.35, 0.12 + 0.17j], [0.12 - 0.17j, 0.65]], dtype=np.complex128)
    coherence = np.sqrt(1 - probability) * (1 - dephasing)
    expected_channel = np.array(
        [
            [1, 0, 0, probability],
            [0, coherence, 0, 0],
            [0, 0, coherence, 0],
            [0, 0, 0, 1 - probability],
        ],
        dtype=np.complex128,
    )
    expected_rho = (expected_channel @ rho.reshape(-1, order="F")).reshape(
        (2, 2), order="F"
    )
    actual = _matrix_result(
        Simulator(method, runtime="cuda", noise=noise),
        program,
        rho if method == "DM" else None,
    )
    expected = expected_rho if method == "DM" else expected_channel
    np.testing.assert_allclose(actual, expected, atol=_ATOL, rtol=0)
    evolved = (
        actual
        if method == "DM"
        else (actual @ rho.reshape(-1, order="F")).reshape((2, 2), order="F")
    )
    _assert_physical(evolved)
    _assert_cpu_parity(
        actual, method, program, rho if method == "DM" else None, noise=noise
    )


def test_density_kraus_evolution_against_sixty_digit_stored_coefficients(
    gpu_available, record_property
):
    mpmath = pytest.importorskip("mpmath")
    pytest.importorskip("numba")
    from fatqat.simulator._engine.cupy import CupyDMEngine
    from fatqat.simulator._engine.nb import NumbaDMEngine
    from fatqat.simulator._engine.np import NumpyDMEngine

    _, initial = _physical_input(4, seed=91)
    rng = np.random.default_rng(192)
    pair = np.asfortranarray(_unitary(rng, 4))
    single = _unitary(rng, 2)
    amplitude = (
        np.diag([1, np.sqrt(0.87)]).astype(np.complex128),
        np.array([[0, np.sqrt(0.13)], [0, 0]], dtype=np.complex128),
    )
    phase = (
        np.sqrt(0.91) * np.eye(2, dtype=np.complex128),
        np.sqrt(0.09) * np.diag([1, -1]).astype(np.complex128),
    )
    # Public targets, explicit stored operators, and channel-vs-gate semantics.
    cycle = (
        (False, (pair,), (1, 0)),
        (True, amplitude, (0,)),
        (False, (single,), (1,)),
        (True, phase, (1,)),
    )
    operations = cycle * 16
    steps = []
    for channel, matrices, targets in operations:
        private_targets = tuple(1 - q for q in targets)
        steps.append(
            ApplyChannelStep(matrices, private_targets)
            if channel
            else ApplyMatrixStep(matrices[0], private_targets)
        )
    states = {}
    with gpu_available.cuda.Device(0):
        for name, engine in (
            ("cupy", CupyDMEngine()),
            ("numpy", NumpyDMEngine()),
            ("numba", NumbaDMEngine()),
        ):
            engine.initialize((2, 2), initial_state=initial)
            channel_rng = np.random.default_rng(82)
            before_rng = channel_rng.bit_generator.state
            for step in steps:
                if isinstance(step, ApplyChannelStep):
                    engine.apply_channel(step, channel_rng)
                else:
                    engine.apply(step)
            assert channel_rng.bit_generator.state == before_rng
            states[name] = engine.export_state()
    mp = mpmath.mp
    with mp.workdps(60):
        reference = mp.matrix(
            [[_exact_complex(value, mp) for value in row] for row in initial]
        )
        for _, matrices, targets in operations:
            evolved = mp.zeros(4)
            for local in matrices:
                dense = _embed(local, targets, (2, 2))
                exact = mp.matrix(
                    [[_exact_complex(value, mp) for value in row] for row in dense]
                )
                evolved += exact * reference * exact.H
            reference = evolved
        errors = {}
        for name, state in states.items():
            differences = [
                abs(_exact_complex(state[row, column], mp) - reference[row, column])
                for row in range(4)
                for column in range(4)
            ]
            errors[name] = {
                "linf": float(max(differences)),
                "frobenius": float(mp.sqrt(mp.fsum(value**2 for value in differences))),
            }
    metrics = {
        "oracle_decimal_digits": 60,
        "operation_count": len(operations),
        "coefficient_source": "exact stored complex128 components",
        "absolute_tolerance": _ATOL,
        "additive_equivalence_floor": _EQUIVALENCE_FLOOR,
        "errors": errors,
        "comparisons": {
            cpu: {
                "gpu_minus_cpu_linf": errors["cupy"]["linf"] - errors[cpu]["linf"],
                "interpretation": (
                    "above_equivalence_floor"
                    if errors[cpu]["linf"] > _EQUIVALENCE_FLOOR
                    else "unresolved_at_equivalence_floor"
                ),
            }
            for cpu in ("numpy", "numba")
        },
    }
    serialized = json.dumps(metrics, sort_keys=True)
    record_property("cuda_density_accuracy", serialized)
    print(serialized, flush=True)
    for name, state in states.items():
        assert state.dtype == np.complex128
        assert np.all(np.isfinite(state)), serialized
        assert errors[name]["linf"] <= _ATOL, serialized
        _assert_physical(state)
    for cpu in ("numpy", "numba"):
        np.testing.assert_allclose(states["cupy"], states[cpu], atol=_ATOL, rtol=0)
        assert (
            errors["cupy"]["linf"] <= errors[cpu]["linf"] + _EQUIVALENCE_FLOOR
        ), serialized


@pytest.mark.parametrize("method", ["DM", "superop"])
def test_grouped_reset_of_entangled_subsystems_is_exact_partial_trace(
    gpu_available, method
):
    dims = (2, 2, 2)
    program = fq.Program(3)
    unitary = np.eye(8, dtype=np.complex128)
    for gate, matrix, targets in (
        (ops.H, _H, (0,)),
        (ops.CX, _CX, (0, 1)),
        (ops.CX, _CX, (0, 2)),
        (ops.S, np.diag([1, 1j]), (0,)),
    ):
        program.add(gate, targets)
        unitary = _embed(matrix, targets, dims) @ unitary
    program.add(ops.Reset, (2, 0))
    actual = _matrix_result(Simulator(method, runtime="cuda"), program)
    if method == "DM":
        expected = np.zeros((8, 8), dtype=np.complex128)
        expected[0, 0] = expected[2, 2] = 0.5
        np.testing.assert_allclose(actual, expected, atol=_ATOL, rtol=0)
        _assert_physical(actual)
    else:
        # Determine every column by acting on E_(row,column), so this catches
        # output as well as input vectorization permutations independently.
        expected = np.zeros((64, 64), dtype=np.complex128)
        for column in range(8):
            for row in range(8):
                elementary = np.zeros((8, 8), dtype=np.complex128)
                elementary[row, column] = 1
                evolved = unitary @ elementary @ unitary.conj().T
                expected[:, row + 8 * column] = _reset_reference(
                    evolved, (2, 0), dims
                ).reshape(-1, order="F")
        np.testing.assert_allclose(actual, expected, atol=_ATOL, rtol=0)
        _, rho = _physical_input(8)
        evolved = (actual @ rho.reshape(-1, order="F")).reshape((8, 8), order="F")
        np.testing.assert_allclose(
            evolved,
            _reset_reference(unitary @ rho @ unitary.conj().T, (2, 0), dims),
            atol=_ATOL,
            rtol=0,
        )
        _assert_physical(evolved)
    _assert_cpu_parity(actual, method, program)


@pytest.mark.parametrize("shots", [1, 4096])
def test_density_dynamic_measurement_feedforward_and_seed_repeatability(
    gpu_available, shots
):
    program = fq.Program(2, 2)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.measure(0, 0)
    program.add(ops.X, 1, condition=(0, 1))
    program.measure(1, 1)
    options = {
        "shots": shots,
        "simulation_config": {"seed": 37},
        "result_config": {"counts": True, "final_state": shots == 1},
    }
    backend = Simulator("DM", runtime="cuda")
    first = backend.run(program, **options).result()
    second = backend.run(program, **options).result()
    counts = first.get_counts()
    assert counts == second.get_counts()
    assert sum(counts.values()) == shots
    assert set(counts) <= {"00", "10"}
    if shots == 1:
        outcome = next(iter(counts))
        expected = np.zeros((4, 4), dtype=np.complex128)
        expected[int(outcome, 2), int(outcome, 2)] = 1
        np.testing.assert_allclose(
            first.get_density_matrix(), expected, atol=_ATOL, rtol=0
        )
    else:
        assert 0.45 < counts.get("10", 0) / shots < 0.55


@pytest.mark.parametrize("method", ["DM", "unitary", "superop"])
def test_matrix_evolution_without_output_avoids_host_state_coercion(
    gpu_available, monkeypatch, method
):
    cp = gpu_available
    program = fq.Program(3)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 2))
    program.add(ops.RY(0.31), 1)
    noise = None
    if method != "unitary":
        program.add(ops.Reset, (2, 0))
        noise = fq.NoiseModel()
        noise.add(fq.noise.AmplitudeDamping(p=0.17), operation=ops.RY)
    backend = Simulator(method, runtime="cuda", noise=noise)
    original_asnumpy = cp.asnumpy

    def guarded_asnumpy(value, *args, **kwargs):
        if (
            isinstance(value, cp.ndarray)
            and value.ndim == 2
            and value.dtype.kind == "c"
        ):
            raise AssertionError(
                "Full complex matrix crossed to host during no-output execution"
            )
        return original_asnumpy(value, *args, **kwargs)

    def prohibit_device_coercion(original):
        def guarded(value, *args, **kwargs):
            if isinstance(value, cp.ndarray):
                raise AssertionError(
                    "A CUDA state reached a NumPy host-array constructor"
                )
            return original(value, *args, **kwargs)

        return guarded

    monkeypatch.setattr(cp, "asnumpy", guarded_asnumpy)
    monkeypatch.setattr(np, "asarray", prohibit_device_coercion(np.asarray))
    monkeypatch.setattr(np, "array", prohibit_device_coercion(np.array))
    job = backend.run(
        program, shots=0, result_config={"counts": False, "final_state": False}
    )
    assert job.status == "DONE"
    assert job.result().metadata["runtime"] == "cuda"
    assert isinstance(backend._engine.state, cp.ndarray)
    expected_size = 64 if method == "superop" else 8
    assert backend._engine.state.shape == (expected_size, expected_size)


@pytest.mark.parametrize("shots", [0, 4096], ids=["exact", "sampled"])
def test_density_estimator_handles_noisy_exact_and_sampled_observables(
    gpu_available, shots
):
    program = fq.Program(1)
    program.add(ops.H, 0)
    program.add(ops.S, 0)
    noise = fq.NoiseModel()
    noise.add(fq.noise.AmplitudeDamping(p=0.3), operation=ops.S)
    observables = [fq.Observable([(letter, 1.0)]) for letter in ("X", "Y", "Z")]
    observables.append(fq.Observable.from_sparse([(["ZERO"], [0], 1.0)], num_qubits=1))
    expected = [0.0, np.sqrt(0.7), 0.3, 0.65]
    estimator = fq.Estimator(Simulator("DM", runtime="cuda", noise=noise))
    options = {"shots": shots, "simulation_config": {"seed": 56}}
    first = estimator.run(program, observables, **options).result()
    second = estimator.run(program, observables, **options).result()
    np.testing.assert_array_equal(first.get_expectation(), second.get_expectation())
    np.testing.assert_allclose(
        first.get_expectation(), expected, atol=0.06 if shots else _ATOL, rtol=0
    )
    assert first.metadata["runtime"] == "cuda"
    assert first.metadata["method"] == "density_matrix"
    if not shots:
        np.testing.assert_array_equal(first.get_standard_error(), [0.0] * 4)
        for runtime in ("numpy", "numba"):
            if runtime == "numba":
                pytest.importorskip("numba")
            cpu = fq.Estimator(Simulator("DM", runtime=runtime, noise=noise))
            result = cpu.run(program, observables, **options).result()
            np.testing.assert_allclose(
                first.get_expectation(), result.get_expectation(), atol=_ATOL, rtol=0
            )


def _custom_kraus_case(num_qubits, width, family):
    """Freeze genuinely nonunitary Kraus terms with complex input/output phases."""
    size = 2**width
    rng = np.random.default_rng(1800 + width)
    if family == "dense":
        stacked, _ = np.linalg.qr(
            rng.normal(size=(3 * size, size)) + 1j * rng.normal(size=(3 * size, size))
        )
        kraus = tuple(stacked[index * size : (index + 1) * size] for index in range(3))
    else:
        weights = np.linspace(0.19, 0.73, size)
        phases = rng.uniform(-2.7, 2.7, size=(2, size))
        kraus = []
        permutations = ((1, 0), (0, 1)) if width == 1 else ((2, 0, 3, 1), (1, 3, 0, 2))
        for index, probabilities in enumerate((weights, 1 - weights)):
            matrix = np.zeros((size, size), dtype=np.complex128)
            rows = np.arange(size) if family == "diagonal" else permutations[index]
            matrix[rows, np.arange(size)] = np.sqrt(probabilities) * np.exp(
                1j * phases[index]
            )
            kraus.append(matrix)
    # Exercise stored noncontiguous/F-order inputs without deriving the oracle
    # from a simulator's lowered or device-resident copies.
    frozen = []
    for index, matrix in enumerate(kraus):
        if index % 2:
            stored = np.asfortranarray(matrix)
        else:
            padded = np.zeros((size, 2 * size), dtype=np.complex128)
            padded[:, ::2] = matrix
            stored = padded[:, ::2]
        stored.flags.writeable = False
        frozen.append(stored)
    kraus = tuple(frozen)
    np.testing.assert_allclose(
        sum(matrix.conj().T @ matrix for matrix in kraus),
        np.eye(size),
        atol=_ATOL,
        rtol=0,
    )

    class Carrier(ops.Operation):
        name = "CudaCustomKrausCarrier"
        num_subsystems = width

    class CustomChannel(Channel):
        num_subsystems = width

    implementation_map = MatrixImplementationMap()
    implementation_map.add(Carrier, np.eye(size, dtype=np.complex128))
    channel_map = ChannelImplementationMap()
    channel_map.add(CustomChannel, lambda _channel, *, targets: kraus)
    noise = fq.NoiseModel()
    noise.add(CustomChannel(), operation=Carrier)
    targets = (
        ((num_qubits - 1,), (0,), (num_qubits - 1,))
        if width == 1
        else ((num_qubits - 1, 0), (0, num_qubits - 1), (num_qubits - 1, 0))
    )
    program = fq.Program(num_qubits)
    for ordered_targets in targets:
        program.add(Carrier(), ordered_targets)
    options = {
        "implementation_map": implementation_map,
        "channel_implementation_map": channel_map,
        "noise": noise,
    }
    return program, options, kraus, targets


def _assert_custom_channel_precision(
    states, initial, kraus, targets, record_property, case
):
    mpmath = pytest.importorskip("mpmath")
    mp = mpmath.mp
    dims = (2,) * (len(initial).bit_length() - 1)
    with mp.workdps(60):
        reference = mp.matrix(
            [[_exact_complex(value, mp) for value in row] for row in initial]
        )
        for ordered_targets in targets:
            evolved = mp.zeros(len(initial))
            for local in kraus:
                full = _embed(local, ordered_targets, dims)
                exact = mp.matrix(
                    [[_exact_complex(value, mp) for value in row] for row in full]
                )
                evolved += exact * reference * exact.H
            reference = evolved
        errors = {}
        for name, state in states.items():
            differences = [
                abs(_exact_complex(state[row, column], mp) - reference[row, column])
                for row in range(len(initial))
                for column in range(len(initial))
            ]
            errors[name] = {
                "linf": float(max(differences)),
                "frobenius": float(mp.sqrt(mp.fsum(value**2 for value in differences))),
            }
    metrics = {
        "case": case,
        "oracle_decimal_digits": 60,
        "coefficient_source": "exact stored complex128 components",
        "absolute_tolerance": _ATOL,
        "additive_equivalence_floor": _EQUIVALENCE_FLOOR,
        "errors": errors,
        "comparisons": {
            cpu: {
                "gpu_minus_cpu_linf": errors["cuda"]["linf"] - errors[cpu]["linf"],
                "interpretation": (
                    "above_equivalence_floor"
                    if errors[cpu]["linf"] > _EQUIVALENCE_FLOOR
                    else "unresolved_at_equivalence_floor"
                ),
            }
            for cpu in ("numpy", "numba")
        },
    }
    serialized = json.dumps(metrics, sort_keys=True)
    record_property("cuda_custom_kraus_accuracy", serialized)
    print(serialized, flush=True)
    for error in errors.values():
        assert error["linf"] <= _ATOL, serialized
    for cpu in ("numpy", "numba"):
        assert (
            errors["cuda"]["linf"] <= errors[cpu]["linf"] + _EQUIVALENCE_FLOOR
        ), serialized


@pytest.mark.parametrize("method", ["DM", "superop"])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("family", ["dense", "diagonal", "phased_permutation"])
def test_public_complex_multikraus_channel_preserves_ordered_sandwiches(
    gpu_available, record_property, method, width, family
):
    pytest.importorskip("numba")
    program, options, kraus, targets = _custom_kraus_case(3, width, family)
    _, initial = _physical_input(8, seed=187)
    original = initial.copy()
    states = {}
    for runtime in ("cuda", "numpy", "numba"):
        backend = Simulator(method, runtime=runtime, **options)
        states[runtime] = _matrix_result(
            backend, program, initial if method == "DM" else None
        )
        if runtime == "cuda":
            repeated = _matrix_result(
                backend, program, initial if method == "DM" else None
            )
            np.testing.assert_array_equal(states[runtime], repeated)
    np.testing.assert_array_equal(initial, original)
    if method == "DM":
        # Emit the higher-precision error record before numerical assertions
        # against double-precision references can terminate this case.
        _assert_custom_channel_precision(
            states, initial, kraus, targets, record_property, f"{family}_{width}qubit"
        )
    for runtime in ("numpy", "numba"):
        np.testing.assert_allclose(states["cuda"], states[runtime], atol=_ATOL, rtol=0)
    expected = initial.copy() if method == "DM" else np.eye(64, dtype=np.complex128)
    for ordered_targets in targets:
        full = [_embed(matrix, ordered_targets, (2, 2, 2)) for matrix in kraus]
        if method == "DM":
            expected = sum(matrix @ expected @ matrix.conj().T for matrix in full)
        else:
            expected = sum(np.kron(matrix.conj(), matrix) for matrix in full) @ expected
    np.testing.assert_allclose(states["cuda"], expected, atol=_ATOL, rtol=0)
    if method == "DM":
        _assert_physical(states["cuda"])
    else:
        evolved = (states["cuda"] @ initial.reshape(-1, order="F")).reshape(
            initial.shape, order="F"
        )
        _assert_physical(evolved)


@pytest.mark.parametrize("width", [1, 2])
def test_density_multikraus_channel_preserves_populated_source_across_tiles(
    gpu_available, width
):
    # Enough independent spectator states to exercise multiple CUDA blocks for
    # both local widths. Every density entry is populated before the first
    # channel, so accidental writes to another term's source are observable.
    program, options, kraus, targets = _custom_kraus_case(7, width, "dense")
    _, initial = _physical_input(128, seed=188)
    original = initial.copy()
    expected = initial.copy()
    for ordered_targets in targets:
        full = [_embed(matrix, ordered_targets, (2,) * 7) for matrix in kraus]
        expected = sum(matrix @ expected @ matrix.conj().T for matrix in full)
    backend = Simulator("DM", runtime="cuda", **options)
    for _ in range(3):
        actual = _matrix_result(backend, program, initial)
        np.testing.assert_allclose(actual, expected, atol=_ATOL, rtol=0)
        _assert_physical(actual)
        np.testing.assert_array_equal(initial, original)
    _assert_cpu_parity(actual, "DM", program, initial, **options)


def _tiny_structure_case(kind):
    angle = 2.0**-40
    matrix = np.eye(4, dtype=np.complex128)
    if kind == "phase":
        # Non-diagonal permutation: its fixed row must retain the tiny phase.
        matrix[0, 0] = matrix[1, 1] = 0
        matrix[0, 1] = matrix[1, 0] = 1
        matrix[2, 2] = np.exp(1j * angle)
    else:
        cosine, sine = np.cos(angle), np.sin(angle)
        matrix[1:3, 1:3] = [[cosine, -sine], [sine, cosine]]
    matrix.flags.writeable = False

    class TinyGate(ops.Operation):
        name = "CudaTinyStructureGate"
        num_subsystems = 2

    implementations = MatrixImplementationMap()
    implementations.add(TinyGate, matrix)
    program = fq.Program(3)
    program.add(TinyGate(), (2, 0))
    expected = _embed(matrix, (2, 0), (2, 2, 2))
    return program, implementations, expected


@pytest.mark.parametrize("kind", ["phase", "rotation"])
def test_unitary_retains_tiny_nonidentity_matrix_coefficients(gpu_available, kind):
    program, implementations, expected = _tiny_structure_case(kind)
    # At 2^-40 the relevant stored coefficient is about 9.1e-13. The usual
    # 1e-12 gate could accept deleting it; 1e-15 leaves binary64 rounding room
    # while separating the stored matrix from identity by over 900 tolerances.
    tolerance = 1e-15
    without_tiny = expected.real if kind == "phase" else np.eye(8)
    assert np.max(np.abs(expected - without_tiny)) > 900 * tolerance
    for runtime in ("cuda", "numpy", "numba"):
        if runtime == "numba":
            pytest.importorskip("numba")
        backend = Simulator(
            "unitary", runtime=runtime, implementation_map=implementations
        )
        actual = _matrix_result(backend, program)
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
