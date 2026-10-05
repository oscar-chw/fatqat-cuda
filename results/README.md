# Benchmark results

`benchmarks.json` is a scrubbed transcription of the benchmark record behind
the fork README. It keeps only workload descriptions, the engine revision, run
configuration labels, median milliseconds, the computed ratios, the
measurement date (2026-09-15) and the name of the private source document for
each row. Machine details are left out.

**r8 = the committed code.** Every number is tagged with the engine revision
it was measured on:

| Section | What it compares | Revision |
| --- | --- | --- |
| `gpu_vs_cpu_r7` | One GPU vs compiled Numba on 4 threads and on all cores | r7 |
| `gpu_vs_cpu_v5` | One GPU vs compiled Numba on 16 workers (not the fastest CPU setting recorded) | v5 |
| `r8_vs_r7_gpu_only` | The same GPU workloads on r7 and r8 | r7 and r8 |
| `r8_vs_r7_dense_fixtures` | Dense two-qubit gates and channels, r7 and r8 | r7 and r8 |
| `experiments_not_in_this_code` | Experimental variants measured against r8, not merged | not in this code |

r8 changed only the CUDA engine. Its GPU-vs-CPU ratio has not been measured,
so no r7 or v5 GPU-vs-CPU ratio should be read as an r8 figure.

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
