# Benchmarks: how much faster is the CUDA engine, against which baselines, and how precise?

## r9 (the committed code), 2026-10-06

Each figure is a median of 5 warm public calls of `perf/scaling.py` (two layers
of RY/RZ on every qubit plus nearest-neighbour CX; the noisy cases add
amplitude damping p=0.07 and phase damping p=0.02; an observable is three
Pauli terms), complex128, including synchronisation and the requested host
output. "CPU" is compiled Numba with `NUMBA_NUM_THREADS=32`; separate CPU runs
varied from run to run. Every GPU output
was checked against a CPU result (an inner product with a fixed random
vector, or the expectation value) at every size.

| Workload | Qubits | CPU r9 (ms) | GPU r9 (ms) | GPU vs CPU | GPU r8 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| statevector, observable | 24 | 774.8 | 19.67 | 39.4× | 42.51 |
| statevector, observable | 26 | 3,296.8 | 77.86 | 42.3× | 180.46 |
| statevector, observable | 28 | 12,456.4 | 351.24 | 35.5× | 757.35 |
| statevector, full state | 24 | 1,144.9 | 49.02 | 23.4× | 102.44 |
| statevector, full state | 26 | 4,653.1 | 187.32 | 24.8× | 348.61 |
| statevector, full state | 28 | 12,654.8 | 799.26 | 15.8× | 1,549.95 |

Sources: [scaling-r9.json](../results/scaling-r9.json) (GPU r9),
[scaling-r9-cpu.json](../results/scaling-r9-cpu.json) (CPU r9),
[scaling.json](../results/scaling.json) (r8, both runtimes, and the
density-matrix and unitary rows, whose code r9 did not change). Separate runs
vary; the controlled comparison of r9 against r8 is the
same-run A/B in [ab-r8-r9.json](../results/ab-r8-r9.json): the r8 and r9
engines alternate child process by child process, with equal GPU memory pools
and host memory in every pair.

- **Where r9 is faster and why:** [optimisations.md](optimisations.md)
  (tiles, simplification, several GPUs), with the CPU tile and
  simplification A/Bs ([cpu-tiles.json](../results/cpu-tiles.json),
  [simplify.json](../results/simplify.json)) and the r10 CPU records
  ([tile-check.json](../results/tile-check.json),
  [simplify-check.json](../results/simplify-check.json)).
- **Precision** ([precision.json](../results/precision.json)): 110 seeded
  circuits over the statevector, density-matrix, unitary and superoperator
  methods, qubit and mixed-radix, against a 60-digit evolution of the stored
  complex128 coefficients. Every runtime stays within 3.1 float64 epsilons.
  Paired over circuits, GPU minus Numba is −0.045 eps (standard error 0.031):
  indistinguishable. NumPy is about 0.1 eps more accurate than both. Turning
  off fused multiply-add on the GPU made it slightly worse (+0.049 eps, SE
  0.024; an unpublished measurement), so it stays on. "Better than the CPU" is not claimed.
- **Rerun:** `python perf/precision.py --require-gpu --out p.json` and
  `python perf/scaling.py --threads 32 --out s.json`; run
  `python perf/scrub_check.py` on any output before publishing it.

## Earlier revisions (r7, v5, r8), 2026-09-15

These ratios were measured on earlier engine revisions (r7 and v5), with
different CPU settings; r8's gain over r7 is shown separately. They are kept
as history and are not comparable with the r9 table above.

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
- **Precision, as tested then:** the r7/r8 tests compared against an
  independent 60-digit reference with `atol=1e-12` and `rtol=0`, and allowed
  each GPU error up to each CPU engine's error plus `8·eps(float64)` (a test
  tolerance, not a measurement; unpublished verification records). The
  measured bound is in the precision entry above: every runtime within 3.1
  eps.

Every row, with its median milliseconds and source document, is in
[`results/benchmarks.json`](../results/benchmarks.json). See
[`results/README.md`](../results/README.md) for how to read it. The source
documents named there belong to a private research record and are not
published.

**Experiments (not in this code).** Unmerged variants measured against r8
reached 382.7 → 232.2 ms on a 13-qubit noisy density-matrix observable and
66.1 → 45.3 ms on the 24-qubit full statevector. They are not part of this
repository and no claim is made for them. Source: `CUDA_EXPERIMENT_RESULTS.md`.
