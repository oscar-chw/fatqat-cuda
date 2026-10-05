"""Resident expectation precision, data movement, reuse, and eager completion."""

import json

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.steps import ApplyMatrixStep
from fatqat.simulator import Simulator
from tests.simulator.test_cupy_accuracy import (
    _ATOL,
    _EQUIVALENCE_FLOOR,
    _exact_complex,
    _oracle_evolve,
    _random_case,
)

_LOCAL_OPERATORS = {
    "I": ((1, 0), (0, 1)),
    "X": ((0, 1), (1, 0)),
    "Y": ((0, -1j), (1j, 0)),
    "Z": ((1, 0), (0, -1)),
    "ZERO": ((1, 0), (0, 0)),
    "ONE": ((0, 0), (0, 1)),
}


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


def _observable_specs(num_qubits):
    last = num_qubits - 1
    all_y = tuple((qubit, "Y") for qubit in range(num_qubits))
    return (
        (
            (0.75, ((0, "ZERO"), (last, "ONE"))),
            (-0.5, ((1, "Z"),)),
            (0.125, ()),
        ),
        (
            (0.3, ((last, "Y"), (0, "X"))),
            (-0.2, ((num_qubits - 2, "Y"), (0, "ONE"))),
            (0.4, ((last, "Z"), (1, "X"))),
            (0.0, ((0, "Y"),)),
        ),
        tuple(
            ((-1.0) ** width / (width + 1), tuple((q, "Y") for q in range(width)))
            for width in range(1, num_qubits + 1)
        ),
        (
            (1024.0, all_y),
            (0.375, ((0, "X"),)),
            (-1024.0, all_y),
            (-0.2, ((last, "ONE"),)),
            (0.2, ()),
        ),
    )


def _observables(specs, num_qubits):
    return [
        fq.Observable.from_sparse(
            [
                (
                    [letter for _, letter in factors] or ["I"],
                    [qubit for qubit, _ in factors] or [0],
                    coefficient,
                )
                for coefficient, factors in terms
            ],
            num_qubits=num_qubits,
        )
        for terms in specs
    ]


def _operator_element(row_digits, column_digits, by_qubit, mp):
    element = mp.mpc(1)
    for qubit, (row, column) in enumerate(zip(row_digits, column_digits, strict=True)):
        local = _LOCAL_OPERATORS[by_qubit.get(qubit, "I")]
        element *= local[row][column]
        if not element:
            break
    return element


def _expectations_mp(state, specs, mp):
    """Explicit <psi|O|psi> with local matrices, without masks or CPU kernels."""
    num_qubits = len(state).bit_length() - 1
    basis = [tuple(map(int, f"{index:0{num_qubits}b}")) for index in range(len(state))]
    results = []
    for terms in specs:
        weighted = []
        for coefficient, factors in terms:
            by_qubit = dict(factors)
            products = []
            for row, row_digits in enumerate(basis):
                for column, column_digits in enumerate(basis):
                    element = _operator_element(row_digits, column_digits, by_qubit, mp)
                    if element:
                        products.append(mp.conj(state[row]) * element * state[column])
            exact_coefficient = _exact_complex(complex(coefficient), mp).real
            weighted.append(exact_coefficient * mp.fsum(products))
        results.append(mp.fsum(weighted).real)
    return results


@pytest.mark.parametrize("num_qubits,seed", [(3, 19), (4, 47), (5, 73)])
def test_resident_reduction_matches_identical_state_sixty_digit_oracle(
    gpu_available, record_property, num_qubits, seed
):
    mpmath = pytest.importorskip("mpmath")
    from fatqat.simulator._engine.cupy import CupySVEngine
    from fatqat.simulator._engine.nb import NumbaSVEngine
    from fatqat.simulator._engine.np import NumpySVEngine

    dims = (2,) * num_qubits
    initial, steps = _random_case(dims, seed, 12)
    specs = _observable_specs(num_qubits)
    mp = mpmath.mp
    with mp.workdps(60):
        evolved = _oracle_evolve(dims, initial, steps, mp)
        snapshot = np.array([complex(value) for value in evolved], dtype=np.complex128)
        # Isolate reduction error: every engine receives the same rounded
        # snapshot, and the oracle embeds those stored components exactly.
        reference = _expectations_mp(
            [_exact_complex(value, mp) for value in snapshot], specs, mp
        )
        actual = {}
        errors = {}
        for name, engine in (
            ("cupy", CupySVEngine()),
            ("numpy", NumpySVEngine()),
            ("numba", NumbaSVEngine()),
        ):
            engine.initialize(dims, initial_state=snapshot)
            actual[name] = engine.expectation_values(engine.state, specs)
            errors[name] = [
                float(abs(_exact_complex(complex(value), mp).real - expected))
                for value, expected in zip(actual[name], reference, strict=True)
            ]
    metrics = {
        "num_qubits": num_qubits,
        "seed": seed,
        "evolution_gate_count": len(steps),
        "oracle_decimal_digits": 60,
        "comparison_scope": "reduction of identical stored complex128 states",
        "absolute_tolerance": _ATOL,
        "additive_equivalence_floor": _EQUIVALENCE_FLOOR,
        "values": actual,
        "absolute_errors_by_observable": errors,
        "gpu_minus_cpu_error": {
            cpu: [
                gpu - host
                for gpu, host in zip(errors["cupy"], errors[cpu], strict=True)
            ]
            for cpu in ("numpy", "numba")
        },
    }
    serialized = json.dumps(metrics, sort_keys=True)
    record_property("cuda_resident_accuracy", serialized)
    print(serialized, flush=True)
    for name, values in actual.items():
        assert np.all(np.isfinite(values)), serialized
        assert max(errors[name]) <= _ATOL, serialized
    for cpu in ("numpy", "numba"):
        np.testing.assert_allclose(actual["cupy"], actual[cpu], atol=_ATOL, rtol=0)
        assert all(
            gpu <= host + _EQUIVALENCE_FLOOR
            for gpu, host in zip(errors["cupy"], errors[cpu], strict=True)
        ), serialized


def _program_and_reference_steps(angle):
    program = fq.Program(3)
    program.add(ops.H, 0)
    program.add(ops.RY(angle), 1)
    program.add(ops.S, 1)
    program.add(ops.CX, (1, 2))
    program.add(ops.RZ(0.41), 0)
    program.add(ops.CY, (2, 0))
    if isinstance(angle, fq.Parameter):
        return program, None
    half = float(angle) / 2
    matrices_and_targets = (
        (np.array([[1, 1], [1, -1]]) / np.sqrt(2), (0,)),
        (np.array([[np.cos(half), -np.sin(half)], [np.sin(half), np.cos(half)]]), (1,)),
        (np.diag([1, 1j]), (1,)),
        (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), (1, 2)),
        (np.diag([np.exp(-0.205j), np.exp(0.205j)]), (0,)),
        (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]), (2, 0)),
    )
    return program, tuple(
        ApplyMatrixStep(
            np.asarray(matrix, dtype=np.complex128), tuple(2 - q for q in targets)
        )
        for matrix, targets in matrices_and_targets
    )


@pytest.mark.parametrize("shots", [0, 4096], ids=["exact", "sampled"])
@pytest.mark.parametrize("device_id", [0, 1])
def test_resident_estimator_sweep_reuses_base_and_preserves_observable_order(
    gpu_available, shots, device_id
):
    cp = gpu_available
    if cp.cuda.runtime.getDeviceCount() <= device_id:
        pytest.skip("Requested CUDA device unavailable")
    mpmath = pytest.importorskip("mpmath")
    angle = fq.Parameter("angle")
    program, _ = _program_and_reference_steps(angle)
    values = [0.2, 0.7, 1.1]
    specs = _observable_specs(3)[:3]
    observables = _observables(specs, 3)
    estimator = fq.Estimator(Simulator(runtime="cuda", device_id=device_id))
    options = {"shots": shots, "simulation_config": {"seed": 81}}
    with cp.cuda.Device(0):
        first = estimator.run_sweep(
            program, observables, {angle: values}, **options
        ).result()
        second = estimator.run_sweep(
            program, observables, {angle: values}, **options
        ).result()
        repeated = estimator.run(
            program.assign_parameters({angle: 0.7}), observables, **options
        ).result()
        assert cp.cuda.runtime.getDevice() == 0
    np.testing.assert_array_equal(
        first[1].get_expectation(), repeated.get_expectation()
    )
    mp = mpmath.mp
    for value, first_result, second_result in zip(values, first, second, strict=True):
        np.testing.assert_array_equal(
            first_result.get_expectation(), second_result.get_expectation()
        )
        _, steps = _program_and_reference_steps(value)
        with mp.workdps(60):
            initial = np.zeros(8, dtype=np.complex128)
            initial[0] = 1
            state = _oracle_evolve((2, 2, 2), initial, steps, mp)
            expected = [float(result) for result in _expectations_mp(state, specs, mp)]
        np.testing.assert_allclose(
            first_result.get_expectation(),
            expected,
            atol=0.06 if shots else _ATOL,
            rtol=0,
        )
        assert first_result.metadata["runtime"] == "cuda"
        if not shots:
            np.testing.assert_array_equal(first_result.get_standard_error(), [0, 0, 0])


@pytest.mark.parametrize("shots", [0, 4096], ids=["exact", "sampled"])
def test_estimator_transfers_scalars_and_counts_without_roundtripping_state(
    gpu_available, monkeypatch, record_property, shots
):
    cp = gpu_available
    num_qubits = 14
    state_size = 2**num_qubits
    program = fq.Program(num_qubits)
    program.add(ops.H, 0)
    program.add(ops.RY(0.4), 3)
    program.add(ops.CX, (0, num_qubits - 1))
    program.add(ops.RZ(0.37), 3)
    observables = [
        fq.Observable.from_sparse([(letter, (qubit,), 1.0)], num_qubits=num_qubits)
        for letter, qubit in (("Z", 0), ("X", num_qubits - 1), ("Y", 3))
    ]
    backend = Simulator(runtime="cuda")
    original_asnumpy, original_array, original_asarray = (
        cp.asnumpy,
        cp.array,
        cp.asarray,
    )
    host_transfer_sizes = []

    def check_host_input(value):
        if isinstance(value, np.ndarray):
            assert not (
                value.ndim == 1 and value.size == state_size and value.dtype.kind == "c"
            ), "Estimator imported a full host state"

    def guarded_array(value, *args, **kwargs):
        check_host_input(value)
        return original_array(value, *args, **kwargs)

    def guarded_asarray(value, *args, **kwargs):
        check_host_input(value)
        return original_asarray(value, *args, **kwargs)

    def guarded_asnumpy(value, *args, **kwargs):
        if isinstance(value, cp.ndarray):
            assert not (
                value.size == state_size and value.dtype.kind == "c"
            ), "Estimator exported a full device state"
            host_transfer_sizes.append(int(value.size))
            if not shots:
                assert value.size <= len(observables) * 4096
        return original_asnumpy(value, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(cp, "array", guarded_array)
        guarded.setattr(cp, "asarray", guarded_asarray)
        guarded.setattr(cp, "asnumpy", guarded_asnumpy)
        result = (
            fq.Estimator(backend)
            .run(program, observables, shots=shots, simulation_config={"seed": 51})
            .result()
        )
    np.testing.assert_allclose(
        result.get_expectation(),
        [0, 0, np.sin(0.4) * np.sin(0.37)],
        atol=0.06 if shots else _ATOL,
        rtol=0,
    )
    assert host_transfer_sizes
    record_property("resident_host_transfer_elements", host_transfer_sizes)
    # A public state request still owns an ordinary host array after the
    # private estimator transfer guards have been removed.
    exported = backend.run(program).result().get_statevector()
    assert isinstance(exported, np.ndarray)
    assert exported.shape == (state_size,)
    assert exported.dtype == np.complex128


def test_no_output_done_job_has_completed_queued_device_work(gpu_available):
    cp = gpu_available
    backend = Simulator(runtime="cuda")
    program = fq.Program(10)
    options = {"shots": 0, "result_config": {"counts": False, "final_state": False}}
    delay = cp.RawKernel(
        r"""extern "C" __global__ void delayed_marker(
            unsigned long long cycles, unsigned long long* marker) {
            unsigned long long start = clock64();
            while (clock64() - start < cycles) {}
            marker[0] = 1;
        }""",
        "delayed_marker",
    )
    with cp.cuda.Device(0):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            marker = cp.zeros(1, dtype=cp.uint64)
            delay((1,), (1,), (np.uint64(1), marker))
            for _ in range(3):
                assert backend.run(program, **options).status == "DONE"
            stream.synchronize()
            event = cp.cuda.Event()
            delay((1,), (1,), (np.uint64(50_000_000), marker))
            event.record(stream)
            try:
                assert not event.done, "Delayed event completed before the test request"
                job = backend.run(program, **options)
                assert job.status == "DONE"
                assert (
                    event.done
                ), "DONE was returned while the CUDA stream still had pending work"
                job.result()
            finally:
                stream.synchronize()


@pytest.mark.parametrize("num_qubits,seed", [(3, 92), (4, 61)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_density_resident_reduction_sixty_digit_oracle(
    gpu_available, record_property, num_qubits, seed, order
):
    mpmath = pytest.importorskip("mpmath")
    from fatqat.simulator._engine.cupy import CupyDMEngine
    from fatqat.simulator._engine.nb import NumbaDMEngine
    from fatqat.simulator._engine.np import NumpyDMEngine

    rng = np.random.default_rng(seed)
    size = 2**num_qubits
    matrix = rng.normal(size=(size, 3)) + 1j * rng.normal(size=(size, 3))
    rho = matrix @ matrix.conj().T
    rho = np.array(rho / np.trace(rho), dtype=np.complex128, order=order)
    specs = _observable_specs(num_qubits)
    basis = [tuple(map(int, f"{i:0{num_qubits}b}")) for i in range(size)]
    mp = mpmath.mp
    with mp.workdps(60):
        reference = []
        for terms in specs:
            weighted = []
            for coefficient, factors in terms:
                entries = [
                    _exact_complex(rho[row, column], mp)
                    * _operator_element(basis[column], basis[row], dict(factors), mp)
                    for row in range(size)
                    for column in range(size)
                ]
                weighted.append(
                    _exact_complex(complex(coefficient), mp).real * mp.fsum(entries)
                )
            reference.append(mp.fsum(weighted).real)
        errors = {}
        for name, engine in (
            ("cupy", CupyDMEngine()),
            ("numpy", NumpyDMEngine()),
            ("numba", NumbaDMEngine()),
        ):
            engine.initialize((2,) * num_qubits, initial_state=rho)
            if name == "cupy":
                engine._state = gpu_available.array(rho, order=order)
                assert (
                    engine.state.flags.c_contiguous
                    if order == "C"
                    else engine.state.flags.f_contiguous
                )
            actual = engine.expectation_values(engine.state, specs)
            errors[name] = [
                float(abs(_exact_complex(complex(value), mp).real - expected))
                for value, expected in zip(actual, reference, strict=True)
            ]
    record_property("density_reduction_errors", json.dumps(errors, sort_keys=True))
    for values in errors.values():
        assert max(values) <= _ATOL
    for cpu in ("numpy", "numba"):
        assert np.all(
            np.array(errors["cupy"]) <= np.array(errors[cpu]) + _EQUIVALENCE_FLOOR
        )


@pytest.mark.parametrize("shots", [0, 1024])
@pytest.mark.parametrize("device_id", [0, 1])
def test_density_estimator_retains_device_base(
    gpu_available, monkeypatch, shots, device_id
):
    cp = gpu_available
    if cp.cuda.runtime.getDeviceCount() <= device_id:
        pytest.skip("Requested CUDA device unavailable")
    program = fq.Program(4)
    program.add(ops.RY(0.7), 0)
    program.add(ops.CX, (0, 3))
    noise = fq.NoiseModel()
    noise.add(fq.noise.AmplitudeDamping(p=0.09), operation=ops.RY)
    observables = [
        fq.Observable([("ZIII", 1.0)]),
        fq.Observable([("YIIY", 1.0)]),
        fq.Observable([("XIIX", 1.0)]),
    ]
    expected = (
        fq.Estimator(Simulator("DM", runtime="numba", noise=noise))
        .run(program, observables, shots=0)
        .result()
        .get_expectation()
    )
    original_asnumpy = cp.asnumpy
    original_array = cp.array

    def guarded_export(value, *args, **kwargs):
        assert value.shape != (
            16,
            16,
        ), "Full density matrix downloaded inside Estimator"
        return original_asnumpy(value, *args, **kwargs)

    def guarded_array(value, *args, **kwargs):
        assert not (
            isinstance(value, np.ndarray) and value.shape == (16, 16)
        ), "Full density matrix uploaded inside Estimator"
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", guarded_export)
    monkeypatch.setattr(cp, "array", guarded_array)
    estimator = fq.Estimator(
        Simulator("DM", runtime="cuda", noise=noise, device_id=device_id)
    )
    options = {"shots": shots, "simulation_config": {"seed": 42}}
    result = estimator.run(program, observables, **options).result().get_expectation()
    repeated = estimator.run(program, observables, **options).result().get_expectation()
    np.testing.assert_array_equal(result, repeated)
    np.testing.assert_allclose(result, expected, atol=0.09 if shots else _ATOL, rtol=0)
