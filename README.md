# FatQat CUDA backend

[![CUDA fork CI](https://github.com/oscar-chw/fatqat-cuda/actions/workflows/cuda-fork-ci.yml/badge.svg)](https://github.com/oscar-chw/fatqat-cuda/actions/workflows/cuda-fork-ci.yml)

A CUDA backend for the open-source FatQat simulator ([spaceqat/fatqat](https://github.com/spaceqat/fatqat)), which is written by the FatQat authors and released under Apache-2.0; this fork adds a GPU engine to it.
Developed by CHOI Hei Wang (Oscar), a student at The Chinese University of Hong Kong (CUHK), as part of coursework for CENG5280, 2026-27 Term 1.
On the earlier r7 engine, one GPU ran a 24-qubit statevector observable about 35× faster than compiled Numba on 4 CPU threads and 6× faster than on all cores; the committed r8 engine has not been timed against the CPU ([Results](#results)).

The GPU is selected at the engine boundary: FatQat's validation, lowering and execution policy stay on the CPU, and only the numerical engine changes. The key path (heavy arrows) keeps the state on the device and sends back only what the call asked for.

```mermaid
flowchart TB
  P[("Program + Simulator:<br/>method, runtime,<br/>device_id")]:::data
  subgraph HOST["Host CPU: FatQat's existing contracts"]
    PE["pulse emulation:<br/>PulseEngine, QuTiP"]:::ext
    L["validate, bind<br/>parameters, lower once"]:::step
    E[("immutable plan:<br/>matrices, channels,<br/>measurements")]:::data
    C{{"execution policy<br/>+ engine capabilities"}}:::gate
    X["BackendValidationError"]:::gate
    N["NumPy / Numba<br/>CPU engine"]:::step
  end
  subgraph DEV["One CUDA device: cupy.py engines"]
    K["FatQat gate and<br/>Kraus kernels"]:::key
    S[("resident complex128<br/>state or operator")]:::key
    O["compensated<br/>partial reductions"]:::key
    M["probability<br/>sampling"]:::key
  end
  subgraph BACK["Host CPU: results"]
    F["math.fsum of<br/>partial sums"]:::step
    D["decode indices<br/>to counts"]:::step
    J[("Job / Result: NumPy<br/>arrays, counts, values")]:::out
  end

  P -->|"Program"| L
  P -.->|"Hamiltonian<br/>emulation: CPU only"| PE
  L -->|"plan + facts"| E
  E -->|"execution shape"| C
  C -->|"shape or controls<br/>unsupported"| X
  C -->|"numpy / numba"| N
  N -->|"NumPy state,<br/>counts"| J
  C ==>|"runtime='cuda'"| S
  E ====>|"matrices uploaded<br/>once per execution"| K
  K ==>|"updates in place"| S
  S ==>|"state stays<br/>on device"| O
  O ==>|"≤ 4096 doubles<br/>per term"| F
  F ==>|"expectation values"| J
  S -->|"probabilities stay"| M
  M -->|"sampled indices"| D
  D -->|"counts"| J
  S -.->|"full copy only if<br/>state requested"| J

  classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
  classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
  classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
  classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
  classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
  classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
```

Where in the code: `src/fatqat/simulator/simulator.py` (`Simulator.__init__`, `_validate_runtime_config`, `_execute_expectation_base`), `src/fatqat/simulator/planning.py`, `src/fatqat/simulator/_execution_policy.py`, `src/fatqat/simulator/_execution_contract.py` (`_EngineCapabilities`), `src/fatqat/simulator/_engine/` (`np.py`, `nb.py`, `cupy.py`), `src/fatqat/emulator/_core/engine.py` (`PulseEngine`).

## Why this exists

FatQat's general simulator runs on the CPU, through NumPy or compiled Numba.
Upstream did not have a GPU runtime. Dense simulation memory grows as `16·D`
bytes for a statevector, `16·D²` for a density matrix or unitary and `16·D⁴`
for a superoperator, where `D = 2^n` for `n` qubits. Each gate pass reads
and writes the whole array. The goal was a GPU path that keeps FatQat's
`Program`, `Job` and `Result` interfaces and its complex128 precision
unchanged.

## Approach

- `Simulator(method, runtime="cuda", device_id=...)` runs the **statevector,
  density-matrix, unitary and superoperator** methods on **one NVIDIA GPU**
  through CuPy. FatQat owns the kernels for one- and two-qubit gates. Wider
  gates and mixed subsystem dimensions fall back to CuPy tensor contraction.
- **complex128 throughout.** No reduced precision, fast-math, truncation or
  approximate channels.
- **Resident state.** Exact and sampled statevector and density-matrix
  Estimator calls keep the state on the GPU. Only the requested output, bounded
  partial sums or samples are copied back to the host.
- **GPU reductions with a compensated host sum.** Each GPU thread accumulates
  with compensated binary64 arithmetic. The host combines the partial sums with
  `math.fsum`.
- **GPU sampling.** Samples are drawn on the device from the seeded host
  random stream. The probability vector stays on the GPU.
- **Optional.** The `cuda13` and `cuda12` extras install CuPy. Without them
  nothing changes, and CuPy is imported only when a CUDA run starts.
- **CPU engines unchanged in behaviour.** The NumPy engine gained an
  array-namespace hook that defaults to NumPy, so the CUDA engine can reuse
  its code. Eight CPU fixtures, RNG state included, are bit-identical to
  upstream (`verification.json`).

What crosses between host and device on each kind of call. Everything not drawn as an arrow stays where it is.

```mermaid
sequenceDiagram
  participant U as Caller
  participant H as Simulator (host)
  participant G as GPU memory
  Note over H,G: every call
  H->>G: gate/Kraus matrices,<br/>uploaded once per execution
  opt initial_state given
    H->>G: host array copied<br/>into owned device state
  end
  H->>G: kernels queued, state updated in place
  H->>G: synchronize stream before the Job is DONE
  alt run() with final state requested
    G-->>H: full state or operator (export_state)
    H-->>U: NumPy array
  else run() with counts
    H->>G: uniforms from the seeded host RNG
    G-->>H: sampled indices only
    H-->>U: counts decoded on host
  else Estimator, exact
    Note over G: evolved state kept resident
    G-->>H: ≤ 4096 partial sums per term
    H-->>U: math.fsum, then expectation values
  else Estimator, sampled
    Note over G: base state kept resident
    G->>G: device-to-device copy<br/>for each measurement tail
    G-->>H: sampled indices only
    H-->>U: estimates from counts
  end
  Note over H,G: matrix uploads released after each execution
```

Where in the code: `src/fatqat/simulator/_engine/cupy.py` (`_execution_scope`, `_matrix_array`, `_allocate`, `export_state`, `sample_indices`, `expectation_values`), `src/fatqat/simulator/simulator.py` (`_execute_expectation_base`, `_execute_sampled_expectation`); tests: `tests/simulator/test_cuda_resident.py`.

## Results

**r8 = the committed code.** The GPU-vs-CPU ratios were measured on earlier engine revisions (r7 and v5); r8 changed
only the CUDA engine and has not been timed against the CPU. Each figure is how many times faster the GPU (or r8) was,
from a median of 5 or 6 warm calls on 2026-09-15, complex128, including synchronisation and host output; below 1 means slower.

| Workload | GPU vs CPU 4 threads (r7) | GPU vs CPU all cores (r7) | GPU vs CPU 16 workers (v5) | r8 vs r7, GPU only | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| 24-qubit statevector, full state | 21.83× | 3.25× | 10.14× | 1.09× | [benchmarks.json](results/benchmarks.json) |
| 24-qubit statevector, observable | 34.75× | 6.00× | not run | 1.15× | [benchmarks.json](results/benchmarks.json) |
| 11-qubit noisy density matrix, full | 8.04× | 1.75× | 2.38× | 1.60× | [benchmarks.json](results/benchmarks.json) |
| 11-qubit noisy density matrix, observable | 10.17× | 1.63× | not run | 1.90× | [benchmarks.json](results/benchmarks.json) |
| 11-qubit unitary | 12.42× | 4.80× | 10.04× | 1.10× | [benchmarks.json](results/benchmarks.json) |
| 4-qubit noisy superoperator | not run | not run | 0.98× (GPU 1.02× slower) | 1.27× | [benchmarks.json](results/benchmarks.json) |

- **The baselines differ in strength.** "All cores" was the fastest CPU setting recorded and varied widely; 16 workers is a weaker baseline on an older engine.
- **The GPU does not always win:** the 4-qubit superoperator was a tie or a small loss, and r8 is not faster than r7 everywhere.
- **Precision:** within 8 machine epsilons of the CPU engine against an independent 60-digit reference; equal-or-better in every case is not established.

Workloads, ranges, the r8-vs-r7 detail and unmerged experiments: [docs/benchmarks.md](docs/benchmarks.md).

## Quick start

```sh
# CPU path, no GPU needed (Python 3.12+, pip 25.1+ for dependency groups)
python -m pip install --upgrade pip
python -m pip install --editable . --group dev --group qiskit
python -m pytest -q        # expect: 2906 passed, 120 skipped (CUDA tests, CuPy not installed)

# NVIDIA GPU: install exactly one CUDA extra matching your toolkit ('.[cuda12]' is the other)
python -m pip install --editable '.[cuda13]' --group dev --group qiskit mpmath
python -m pytest -q        # now also runs the CUDA tests (r8: 3026 passed, no skips)
```

Then use `Simulator("SV", runtime="cuda", device_id=0)` or `Simulator("DM", runtime="cuda", device_id=0)` from `fatqat.simulator`.
The r8 GPU run used CUDA 13.2 and CuPy 14.2 (`verification.json`); CI runs the CPU suite on Python 3.12 and 3.13.

## Project structure

```text
src/fatqat/                FatQat package; the fork's GPU engine is simulator/_engine/cupy.py
tests/                     upstream suite plus the CUDA tests (tests/simulator/test_cuda_*.py, test_cupy_*.py)
results/                   scrubbed benchmark record (benchmarks.json) and how to read it
docs/                      fork pages, the upstream README and design notes, the MkDocs site (docs/mkdocs/)
.github/workflows/         upstream tests and lint, plus the fork's CPU-path CI
LICENSE, NOTICE            Apache-2.0; the fork notice
pyproject.toml, mkdocs.yml, conftest.py, AGENTS.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md   upstream project files
```

Docs: see [docs/README.md](docs/README.md).

## Limits

- **Not covered on the GPU:** pulse emulation; neutral-atom occupancy and loss (`AtomArraySimulator` rejects CUDA); stochastic statevector trajectories (CUDA statevectors reject channels, reset, mid-circuit measurement and feedforward); Apple GPUs; splitting one state across several GPUs ([coverage diagram](docs/cuda-coverage.md)).
- Small circuits can run faster on the CPU, because GPU launch costs can dominate.
- r8, the committed engine, has not been timed against the CPU; every GPU-vs-CPU ratio is from r7 or v5.
- The CUDA 12 extra is packaged but has not been tested on a device.
- The benchmark source documents belong to a private research record and are not published; `results/benchmarks.json` is a scrubbed transcription.
- **Not merged upstream.** This is an independent fork and has not been submitted upstream.

## Lessons

- The baseline decides the headline: the same 24-qubit full-state run was about 22× faster than 4 CPU threads but about 3× faster than all cores.
- A GPU is not a free win: at 4 qubits the superoperator was a tie or a small loss, and small circuits can be faster on the CPU.
- Precision has to be tested, not assumed: against a 60-digit reference the GPU stays within 8 machine epsilons of the CPU, but is not better in every case.
- Swapping only the numerical engine, behind FatQat's own validation and lowering, kept `Program`, `Job` and `Result` unchanged and the CPU engines bit-identical on eight fixtures.

## Credits and licence

- FatQat, its source, documentation and tests are the work of the FatQat authors ([spaceqat/fatqat](https://github.com/spaceqat/fatqat)), Apache-2.0. Their README is kept in [docs/upstream-README.md](docs/upstream-README.md).
- The fork's CUDA engine, its tests and documentation are Copyright 2026 Oscar Choi, Apache-2.0; see [NOTICE](NOTICE) and [LICENSE](LICENSE).

Implemented with AI coding agents under Oscar's design and review.
