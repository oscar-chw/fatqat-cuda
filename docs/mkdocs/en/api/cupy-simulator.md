# CUDA runtime

`Simulator(runtime="cuda")` runs statevector, density-matrix, unitary and
superoperator calculations on one NVIDIA GPU. It retains `complex128` values
and returns ordinary NumPy arrays and FatQat results. It performs no state
truncation, reduced-precision calculation, or change to the circuit model.
Floating-point results can differ from CPU results; bitwise reproducibility
across runtimes is not guaranteed.

CUDA is built into FatQat's numerical runtime selection. The CPU prepares and
validates the circuit and handles classical control; FatQat's CUDA kernels
evolve the state or operator on the GPU. CuPy supplies device arrays, tensor
operations and kernel compilation. NumPy and Numba remain available for CPU
execution, including all simulation methods.

From this source checkout, install exactly one extra matching your CUDA Toolkit:

```sh
python -m pip install '.[cuda13]'  # CUDA 13.x
# Or, for CUDA 12.x: python -m pip install '.[cuda12]'
```

The tested environment uses CuPy 14.2.0 and CUDA 13.2 on Linux. CUDA 12 packaging
is provided but has not been device-tested here. See
[CuPy installation](https://docs.cupy.dev/en/stable/install.html) for driver and
Toolkit requirements. CPU installations need neither extra. CuPy and a
compatible device are required only when device execution starts. This runtime
does not run on Apple's Metal GPU.

## Example

```python
import fatqat as fq
import fatqat.operations as ops
from fatqat.simulator import Simulator

program = fq.Program(2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
result = Simulator(method="SV", runtime="cuda", device_id=0).run(
    program,
    shots=0,
    result_config={"counts": False, "final_state": True},
).result()
state = result.get_statevector()  # NumPy complex128, public subsystem order
```

## Supported requests

The method names and aliases are the same as for CPU execution:

| Method | CUDA coverage | Restrictions |
| --- | --- | --- |
| `statevector` / `SV` | Ideal single-pass evolution, terminal measurement and seeded counts | Rejects quantum channels, reset, intermediate measurement and feedforward |
| `density_matrix` / `DM` | Exact finite channels and reset; terminal or intermediate measurement, seeded counts and feedforward | Dynamic circuits execute shots serially |
| `unitary` | The complete unitary map | Rejects channels, reset, measurement, conditions, counts and `initial_state` |
| `superop` | The complete channel map, including finite channels and reset | Rejects measurement, conditions, counts and `initial_state` |

All four methods support mixed subsystem dimensions, local matrix
implementations and parameter sweeps. State methods accept host initial states
under the usual shape rules and copy them into owned device memory. Requested
final states or operators are exported as owned NumPy arrays in public
subsystem order. The existing single-shot restriction applies when a requested
final state depends on sampled outcomes. Readout confusion remains a classical
reporting operation.

`fq.Estimator` supports exact and sampled statevector and density-matrix
requests within these method limits. Both representations keep the evolved state on the GPU and transfer bounded
reduction data or samples instead of the full state. Each sampled measurement
tail receives an owned device copy of the evolved base. Density expectations
read only the entries needed by the observable, without constructing a full
observable matrix.
Measured programs are rejected by the shared Estimator validation.

The superconducting matrix simulators inherit this runtime coverage while
retaining their device constraints. `AtomArraySimulator` rejects CUDA at
construction because its occupancy and loss lifecycle is unsupported. Pulse
emulation has no CUDA runtime.

Unsupported dynamics or execution controls raise `BackendValidationError`.
For CUDA, use `auto` or `serial` for shot and kernel parallelism,
`max_workers=None`, and `fusion=False`; CPU shot workers and fusion are
unsupported. Missing CuPy, unavailable devices, and execution failures surface
through an `ERROR` Job; `job.result()` raises the captured error.

## Devices and memory

`device_id` selects an ordinal among the process's visible CUDA devices. A
backend instance is not safe for concurrent calls. To run independent circuits
on several GPUs, use one instance in each process, with a distinct device ID or
`CUDA_VISIBLE_DEVICES` selection. This does not distribute one state across GPUs.

For parameter sweeps, `device_id` also accepts a tuple of distinct ordinals.
`run_sweep` then runs the rows on every listed GPU, one worker thread and engine
per GPU, and returns results in input order; each row is computed exactly as it
would be on one GPU. `run` uses the first listed device. If a row fails, the
Job reports the earliest failing row, although rows on other GPUs may already
have run. The speed-up is less than the number of GPUs, because part of each
row's work runs in Python one thread at a time.

```python
import numpy as np
import fatqat as fq
import fatqat.operations as ops
from fatqat.parameters import Parameter
from fatqat.simulator import Simulator

theta = Parameter("theta")
program = fq.Program(2, 2)
program.add(ops.RY(theta), 0)
program.add(ops.CX, (0, 1))
program.measure_all()
results = Simulator(method="SV", runtime="cuda", device_id=(0, 1)).run_sweep(
    program, {theta: np.linspace(0, np.pi, 8)}, shots=1000
).result()  # eight Results, in the order of the theta values
```

The engine retains its most recent device state. Matrix uploads are
released after each public execution; CuPy's allocator may retain freed blocks.
Separate live allocation from reserved pool memory when sizing a workload.
For total Hilbert-space dimension `D = prod(local_dimensions)`, the primary
complex128 array requires:

| Method | Array storage |
| --- | --- |
| `statevector` | `16 * D` bytes |
| `density_matrix` or `unitary` | `16 * D**2` bytes |
| `superop` | `16 * D**4` bytes |

Contractions, channel application, sampling, collapse, host exports and
retained sweep results require additional memory. A historical statevector
pilot exported a 32-qubit GHZ state using only in-place one- and two-qubit
kernels. That capacity check does not establish capacity for the current
matrix methods, arbitrary circuits or their temporary buffers.

Small circuits may run faster on the CPU because GPU setup and launch costs
dominate. Compare end-to-end wall time, including requested host results, at the
same precision and circuit settings.

## Compatibility

The earlier `fatqat.simulator.experimental.CupySimulator` name delegates to
`Simulator(method="SV", runtime="cuda")`. Its results now report runtime `cuda`.
New code should select CUDA through `Simulator`.

See the [Simulator API](simulator.md) for shared methods and result contracts.
