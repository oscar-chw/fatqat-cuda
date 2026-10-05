# Benchmarks: how much faster is the CUDA engine, against which baselines, and how precise?

**r8 = the committed code.** The GPU-vs-CPU ratios below were measured on
earlier engine revisions (r7 and v5). r8 changed only the CUDA engine, and its
gain over r7 is shown separately. r8 has not been timed against the CPU.

Each figure is a median of 5 or 6 warm public calls, measured on 2026-09-15.
The workloads are two layers of RY/RZ on every qubit plus nearest-neighbour CX,
run in complex128. The noisy cases add amplitude damping (p=0.07) and phase
damping (p=0.02). An observable is the exact expectation of three Pauli terms.
Timing includes GPU synchronisation and the requested host output. Each entry
says how many times faster the GPU (or r8) was. Below 1 means slower.

| Workload | GPU vs CPU 4 threads (r7) | GPU vs CPU all cores (r7) | GPU vs CPU 16 workers (v5) | r8 vs r7, GPU only | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| 24-qubit statevector, full state | 21.83× | 3.25× | 10.14× | 1.09× | CROSS_ARCHITECTURE_RESULTS.md; CUDA_IMPLEMENTATION_RESULTS.md |
| 24-qubit statevector, observable | 34.75× | 6.00× | not run | 1.15× | CROSS_ARCHITECTURE_RESULTS.md; CUDA_IMPLEMENTATION_RESULTS.md |
| 11-qubit noisy density matrix, full | 8.04× | 1.75× | 2.38× | 1.60× | CROSS_ARCHITECTURE_RESULTS.md; CUDA_IMPLEMENTATION_RESULTS.md |
| 11-qubit noisy density matrix, observable | 10.17× | 1.63× | not run | 1.90× | CROSS_ARCHITECTURE_RESULTS.md; CUDA_IMPLEMENTATION_RESULTS.md |
| 11-qubit unitary | 12.42× | 4.80× | 10.04× | 1.10× | CROSS_ARCHITECTURE_RESULTS.md; CUDA_IMPLEMENTATION_RESULTS.md |
| 4-qubit noisy superoperator | not run | not run | 0.98× (GPU 1.02× slower) | 1.27× | CROSS_ARCHITECTURE_RESULTS.md; CUDA_IMPLEMENTATION_RESULTS.md |

- **The baselines differ in strength.** "All cores" was the fastest CPU setting
  recorded, and its runs varied widely: for example, the 24-qubit observable
  ranged from 276.8 to 618.1 ms. The 16-worker setting is a weaker baseline,
  and v5 is an older engine. GPU runs gave the host process 16 CPU threads.
- **The GPU does not always win.** The 4-qubit superoperator was a tie or a
  small loss.
- **r8 vs r7 on the same GPU workloads.** The noisy 11-qubit observable went
  from 26.539 to 14.004 ms. Tracked live GPU memory for the 11-qubit noisy
  density matrix went from 192.130 to 128.065 MiB. r8 is not faster
  everywhere: the 9-qubit unitary was 1.9% slower, and dense two-qubit
  fixtures stayed between 0.98× and 1.10×.
- **Precision:** within 8 machine epsilons of the CPU engine; equal-or-better
  in every case is not established. The tests compare against an independent
  60-digit reference with `atol=1e-12` and `rtol=0`, and require each GPU
  error to be at most each CPU engine's error plus `8·eps(float64)`. Evidence:
  `verification.json` (r8), `verification-r7.json`, `FQ020_VERIFICATION.json`.

Every row, with its median milliseconds and source document, is in
[`results/benchmarks.json`](../results/benchmarks.json). See
[`results/README.md`](../results/README.md) for how to read it. The source
documents named there belong to a private research record and are not
published.

**Experiments (not in this code).** Unmerged variants measured against r8
reached 382.7 → 232.2 ms on a 13-qubit noisy density-matrix observable and
66.1 → 45.3 ms on the 24-qubit full statevector. They are not part of this
repository and no claim is made for them. Source: `CUDA_EXPERIMENT_RESULTS.md`.
