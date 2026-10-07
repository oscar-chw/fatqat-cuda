# Performance and scaling

Before tuning runtimes, estimate how much state FatQat must carry. Subsystem
dimensions and the chosen representation usually matter sooner than a runtime
switch.

First choose the simplest execution level that contains the effect you need;
[Choose how much physics to model](execution-models.md) draws that boundary.
Then use the estimates and benchmark pattern below on your own
[`Program`][fatqat.Program].

## Count the state space before running

If the local dimensions are `d0, d1, ...`, the state-space dimension is their
product. This makes FatQat's mixed-dimensional Program support useful and also
makes its scaling explicit:

```pycon
>>> import math
>>> local_dimensions = (2, 2, 2, 2, 3, 3)  # four qubits and two qutrits
>>> dimension = math.prod(local_dimensions)
>>> dimension
144
>>> dimension**2
20736
```

A statevector stores one complex entry per basis state. A density matrix or
unitary stores a square array, and a super-operator is square in the already
squared density-matrix space. In terms of total dimension `D`, their entry
counts scale as `D`, `D**2`, `D**2`, and `D**4`, respectively. Temporary work
space and backend overhead add to these lower-level counts. With complex128
values, multiply each entry count by 16 to obtain the primary array size in
bytes. GPU execution has the same representation costs, plus device
temporaries and any requested host copies.

The growth is exponential in both subsystem count and local dimension:

![Logarithmic curves show statevector and density-matrix entry counts growing faster for qutrits than for qubits as subsystem count increases.](../assets/generated/guide/performance-1.png)

??? example "Reproduce this figure"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    subsystems = np.arange(1, 9)
    qubit_state = 2 ** subsystems
    qutrit_state = 3 ** subsystems
    qubit_density = qubit_state ** 2
    qutrit_density = qutrit_state ** 2

    assert np.all(np.diff(qubit_state) > 0)
    assert np.all(np.diff(qutrit_density) > 0)

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.semilogy(
        subsystems,
        qubit_state,
        marker="o",
        label="qubit statevector",
    )
    ax.semilogy(
        subsystems,
        qubit_density,
        marker="o",
        label="qubit density matrix",
    )
    ax.semilogy(
        subsystems,
        qutrit_state,
        marker="s",
        label="qutrit statevector",
    )
    ax.semilogy(
        subsystems,
        qutrit_density,
        marker="s",
        label="qutrit density matrix",
    )
    ax.set(
        xlabel="number of equal-dimension subsystems",
        ylabel="complex array entries",
        xticks=subsystems,
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, ncols=2, fontsize="small")
    fig.tight_layout()
    ```

This plot counts entries rather than bytes so it does not assume a dtype or
allocator. Estimate the Program you actually intend to run, including physical
levels that an emulator models even when the logical Program addresses only
qubits.

## Request only the answer you need

Result choice can dominate scaling. A complete unitary or super-operator asks
for the action on every input, while a state run asks for one input state.
Likewise, a full state is unnecessary when you only need counts or a few
expectation values.

Shot cost also depends on the Program. A circuit that evolves deterministically
and measures only at the end can reuse more work than one with mid-circuit
measurement, reset, feedforward, or stochastic trajectories. Avoid estimating
shot cost from `shots` alone; benchmark the same Program and result request you
will use in practice.

See [Estimate observables](interpret-results.md) for expectation values and
uncertainty, and the [Simulator API](../api/simulator.md) for method and result
constraints.

## Compare NumPy and Numba on your workload

FatQat's general simulator offers two CPU runtimes:

- NumPy executes directly and avoids JIT compilation startup.
- Numba compiles numerical kernels on first use and can reuse compatible
  compiled work on later calls.

Neither choice changes the Program or the modeled mathematics. Compilation,
array-library behavior, CPU, operating system, Program shape, and repetition
count all affect the result, so benchmark rather than assuming one runtime is
always preferable.

The built-in [CUDA runtime](../api/cupy-simulator.md) executes statevector,
density-matrix, unitary and superoperator calculations with complex128
precision. The CPU retains circuit preparation, validation, classical control
and result construction; the evolving state or operator remains on the GPU.
Use `Simulator("SV", runtime="cuda", device_id=0)` on an NVIDIA host, selecting
another method when needed. Statevectors and density matrices support
channels, reset and dynamic measurements, run per shot with shot branching;
`device_id="all"` spreads a run's shots over every visible GPU when they can
branch apart (at least as many outcomes of the random steps before the last
as shots; for a density matrix, of its measurements). Operator methods retain their usual restrictions. CUDA does not
accelerate atom occupancy or pulse emulation.

Compare the same requested output and include host transfers in timing.
Statevector and density-matrix Estimator requests retain the base state on the
GPU and transfer only reduction data or samples. Requesting a full
state or operator requires its transfer to the host. Include that cost when
comparing runtimes, and measure both first-call and repeated-call behavior.
Small circuits can favor CPU runtimes because GPU setup and launch costs may
dominate.

The following harness separates one untimed warm-up from repeated measurements
and compares like-for-like final states:

```python
import statistics
import time

import numpy as np
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(8)
for _ in range(6):
    for target in range(8):
        program.add(ops.RY(0.17), target)
    for control in range(7):
        program.add(ops.CX, (control, control + 1))

result_config = {"counts": False, "final_state": True}
numpy_backend = fq.simulator.Simulator("SV", runtime="numpy")
numba_backend = fq.simulator.Simulator("SV", runtime="numba")

def warm_and_measure(backend, repeats=7):
    # Keep compilation and other first-use setup outside steady-state samples.
    backend.run(program, result_config=result_config).result()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        backend.run(program, result_config=result_config).result()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)

numpy_state = numpy_backend.run(program, result_config=result_config).result()
numba_state = numba_backend.run(program, result_config=result_config).result()
assert np.allclose(
    numpy_state.get_statevector(),
    numba_state.get_statevector(),
)

numpy_seconds = warm_and_measure(numpy_backend)
numba_seconds = warm_and_measure(numba_backend)
print({"numpy": numpy_seconds, "numba": numba_seconds})
```

Treat the printed values as local evidence, not package guarantees. If startup
latency matters, measure the first run separately instead of discarding it.
If sustained throughput matters, increase the repetition count and Program
size to match the intended workload.

## Tune parallelism and fusion only after measuring

Automatic execution settings are the appropriate baseline. Manual shot
parallelism, kernel parallelism, worker limits, and operation fusion apply only
to compatible workloads, and overhead can outweigh saved numerical work.

When tuning:

1. Keep the Program, method, runtime, seed, and result request fixed.
2. Warm any compiled path before measuring steady state.
3. Measure several repetitions and report a robust statistic such as the
   median.
4. Change one execution choice at a time.
5. Verify the resulting state, counts distribution, or observable before
   accepting the timing.
6. Repeat at the problem sizes that matter; a small example may rank choices
   differently from the target workload.

For eligible combinations and error behavior, see
[Simulator runtime and execution](../api/simulator.md). Fusion is opt-in,
and explicit parallel modes can be rejected when the Program cannot use them.

## Simplify circuits exactly with `simplify`

Compiled or hand-written circuits often contain gates that cancel or
collapse: `X` then `X`, `H X H` (which is `Z`), `T` then `T` (which is `S`),
three `CX` gates that form a `SWAP`, or `CX RZ CX`, which is one diagonal
`ZZ` rotation. Such runs are multiplied out before execution, so the state
is updated fewer times.

By default (`simplify="auto"`) only rewrites that change no value are made:
products of unit gates (entries `0`, `±1`, `±i`) and gates on known basis
inputs, on Numba and CUDA, when the state (times the shots, for a run evolved shot by shot) is
large enough that the pass costs a few per cent of a plain run at most.
`result.metadata["simplification"]` says whether it ran, and why not if it
did not. `simulation_config={"simplify": True}` does more, as below, and
`False` turns it off.

Products are computed exactly, not in floating point, for unit gates
(entries `0`, `±1`, `±i`: Paulis, `S`, `CX`, `SWAP`), the built-in `H`, `T`,
`Tdg` and `SX`, and rotations conjugated by `±1` permutations. With `True`,
phase gates (`T`, `Tdg`, `S`, `Z`) on the same parity of qubits are summed
across `CX`, `X` and `SWAP` gates into one (phase folding), which removes
most of the work from phase gadgets. A run is
replaced only by a product that rounds no more, so the result is closer to
the ideal circuit on average (no per-circuit bound: a circuit can end up slightly worse): rewrites
of unit gates leave Numba and CUDA values unchanged, and cancelled `H` or `T`
gates no longer add their rounding.
Two rotations are never merged with each other. From the all-zero start,
gates that act as the identity on qubits still in a known basis state (a `CX`
whose control is still `|0⟩`) are dropped. Measurements, resets and channels
are never crossed. Measure the gain on your own circuit, as above.

## What speed to expect on your hardware

`runtime="auto"` picks for you, run by run: the CPU for small states, CUDA
for large ones on an NVIDIA GPU, and Metal for large statevectors on an Apple
GPU. Measured, it was never slower than the better of the CPU and one GPU at
any size ([auto-check.json](https://github.com/oscar-chw/fatqat-cuda/blob/main/results/auto-check.json)).
What it gains depends on the machine. Measured figures are marked; the
others are estimates scaled from them, not measurements.

| Machine | Small circuits (under about 16 qubits) | Large circuits (24–28 qubits) |
| --- | --- | --- |
| NVIDIA workstation GPU, 32 CPU threads (measured) | CPU as fast or faster | 17–21× (default products), 23–26× with `gpu_products="plain"`, on observables; 6–8× with the full state copied back |
| Gaming PC, high-end GPU (24–32 GB) with a desktop CPU (estimate) | CPU as fast or faster | about 20–45×, 25–60× with plain products |
| Gaming PC, mid-range GPU (8–12 GB) with a desktop CPU (estimate) | CPU as fast or faster | about 7–18×, 10–25× with plain products |
| Apple-silicon Mac, Max-class chip, `runtime="metal"` (measured) | CPU (the GPU takes no work below 16 MiB of state) | 1.18–1.52× over the CPU alone |
| Apple-silicon Mac, base or Pro chip (estimate) | CPU | about 1.05–1.2× (base) to 1.1–1.4× (Pro) over the CPU alone |

Why these shapes:

- Simulating a statevector is limited by memory bandwidth, not cores: 96 CPU
  threads were no faster than 32. A desktop CPU has much less bandwidth than
  a server, which is why a GPU gains more over it.
- Consumer and workstation NVIDIA GPUs compute double precision (binary64)
  at 1/64 of their single-precision rate, so the GPU's arithmetic, not only its
  memory, sets its speed; `gpu_products="plain"` does less of it.
- Apple GPUs have no hardware binary64. FatQat computes it in software, bit
  for bit like the CPU, so the GPU adds to the CPU instead of replacing it,
  and a chip with fewer GPU cores gains less.
- On an NVIDIA machine one state is never split between CPU and GPU: the GPU
  does about 20 times the CPU's work, so the CPU could add a few per cent at
  most, at the cost of bit-identical results.
- GPU memory bounds the size: a statevector needs 16 bytes per amplitude
  (4 GiB at 28 qubits), so a 12 GB card holds up to 29 qubits and a 32 GB
  card 31.

Measure your own circuit before relying on any of these, as above.

## Account for physical emulation separately

Hamiltonian emulators add costs that a circuit-level array-size estimate does
not capture: the model's physical levels, unaddressed modeled subsystems,
time-dependent controls, scheduling, integration intervals, and open-system
evolution. A logical qubit may therefore contribute three physical levels in a
transmon model.

Benchmark the actual model, arrangement, controls, duration, solver settings,
and requested result. Do not extrapolate an emulator run from a gate-level
Simulator timing. The [emulator API](../api/emulators/index.md) lists the
solver and schedule controls available to each physical model.
