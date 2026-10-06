# Results

Every file here is produced by a script in [`perf/`](../perf) (or assembled
from its output) and passed through `perf/scrub_check.py`, which refuses
hardware models, host names, paths and device IDs. Machines are named only by
their run configuration ("one GPU", "CPU A, 8 threads", "CPU B, 32 threads").

## r9 records, 2026-10-06

| File | What it holds | Produced by |
| --- | --- | --- |
| `precision.json` | Round-off of NumPy, Numba and CUDA against a 60-digit oracle, 110 circuits, four methods, in float64 epsilons, with paired comparisons | `perf/precision.py --require-gpu` |
| `scaling.json` | r8 engine: GPU vs CPU (compiled Numba, 32 threads), five workloads, 6-28 qubits | `perf/scaling.py --threads 32 --require-gpu` |
| `scaling-r9.json` | r9 engine, GPU rows, each checked against the CPU rows of `scaling.json` | `perf/scaling.py --runtimes cuda --reference scaling.json` |
| `scaling-r9-cpu.json` | r9 engine, CPU statevector rows (Numba cache tiles) | `perf/scaling.py --runtimes numba` |
| `ab-r8-r9.json` | Same-run A/B of the r8 and r9 GPU engines, with GPU pool and host memory | `perf/scaling.py --child`, arms alternating |
| `ab-unitary-tiles.json` | Same-run A/B of the CUDA unitary method without and with gate tiles | `perf/scaling.py --child`, arms alternating |
| `cpu-tiles.json` | Numba gate core, per-gate passes vs cache tiles, two CPUs | same-process A/B of `NumbaSVEngine` variants |
| `simplify.json` | `simplify=True`: speed on a Clifford-rich circuit, and accuracy with and without it | public Estimator calls; extended-precision and 60-digit references |

Ratios are `baseline median / candidate median`; above 1 means the candidate
is faster. Medians come from 5 warm calls (A/B files: the median over rounds
of each round's median).

### Which public commit each `code_revision` is

The `code_revision` fields name the working commits that were measured. Those
commits were squashed before publication, so they are not in the public
history; the engine code they ran is, file for file:

| `code_revision` | Measured in | Engine code identical to public commit |
| --- | --- | --- |
| `6558eab` (r8) | `precision.json`, `scaling.json` | all of `src/`: `c92ff77` |
| `078172a`, `ff32619` (r9, GPU) | `scaling-r9.json`, `ab-r8-r9.json` | `cupy.py`, `nb.py`, `np.py`, `base.py`: `0875056` |
| `597c1a4` (r9, CPU) | `scaling-r9-cpu.json`, `simplify.json` | `cupy.py`, `np.py`, `base.py`: `0875056`; `nb.py`: `5c7c46d` |

`precision.json` was measured on r8 code. Rerunning the same harness on r9
code (`597c1a4`, recorded in `simplify.json`) reproduced all 330 of its
runtime-circuit errors exactly.

## r10 records

r10 changes the simplification pass (exact gate algebra) and the tiles
(controls and diagonal gates take no tile bit). These files record it at the
commit they name, which is in the public history:

| File | What it holds | Produced by |
| --- | --- | --- |
| `simplify-check.json` | `simplify=True` against the ideal circuit (exact gate definitions, 60 digits), paired per circuit on every CPU runtime, and wall time on four circuits | `perf/simplify_check.py` |
| `tile-check.json` | Tiles where controls and diagonal gates take no tile bit, vs tiles where every target does and vs per-gate passes, with pass counts; same engine, arms alternating | `perf/tile_check.py --runtimes numba` |
| `tile-check-cuda.json` | The same on one GPU, 26 qubits | `perf/tile_check.py --runtimes cuda --qubits 26` |
| `simplify-check-cuda.json` | `simplify=True` on one GPU: accuracy against the ideal circuit, and timing at 26 qubits | `perf/simplify_check.py --runtimes cuda --qubits 26` |
| `precision-r10.json` | The 110 circuits of `precision.json` on r10 code, every runtime | `perf/precision.py --require-gpu` |
| `metal-prototype.json` | Apple-GPU prototype sharing each tile batch with Numba, against Numba alone; equality of every final state | `prototypes/metal/metal_engine.py` |
| `metal-fp64-check.json` | Software binary64 on the Apple GPU against the CPU: 8 million multiplies and adds, and a one-qubit gate | `prototypes/metal/fp64check.swift` |

## Earlier record

`benchmarks.json` is a scrubbed transcription of the benchmark record behind
the fork README. It keeps only workload descriptions, the engine revision, run
configuration labels, median milliseconds, the computed ratios, the
measurement date (2026-09-15) and the name of the private source document for
each row. Machine details are left out.

Every number in `benchmarks.json` is tagged with the engine revision it was
measured on:

| Section | What it compares | Revision |
| --- | --- | --- |
| `gpu_vs_cpu_r7` | One GPU vs compiled Numba on 4 threads and on all cores | r7 |
| `gpu_vs_cpu_v5` | One GPU vs compiled Numba on 16 workers (not the fastest CPU setting recorded) | v5 |
| `r8_vs_r7_gpu_only` | The same GPU workloads on r7 and r8 | r7 and r8 |
| `r8_vs_r7_dense_fixtures` | Dense two-qubit gates and channels, r7 and r8 | r7 and r8 |
| `experiments_not_in_this_code` | Experimental variants measured against r8, not merged | not in this code |

No r7 or v5 GPU-vs-CPU ratio should be read as an r8 or r9 figure; r8 and r9
were measured against the CPU in `scaling.json` and `scaling-r9*.json` above,
with a different CPU setting.

Ratios are `baseline median / candidate median`, rounded to two decimals.
Above 1 means the candidate is faster. To recompute every ratio:

```sh
python -c "import json; d=json.load(open('results/benchmarks.json')); [print(k, r['workload'], round(r['baseline']['median_ms']/r['candidate']['median_ms'], 2), r['ratio_baseline_over_candidate']) for k in ('gpu_vs_cpu_r7','gpu_vs_cpu_v5','r8_vs_r7_gpu_only','r8_vs_r7_dense_fixtures') for r in d[k]]"
```

Medians come from 5 or 6 warm calls per cell. No confidence intervals were
collected, and the all-cores CPU runs varied widely (ranges are in the JSON).
Treat small differences as noise.

The source documents named in `source_document` belong to a private research
record and are not published here.
