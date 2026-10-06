# Apple GPU (Metal) prototype

A prototype, not a FatQat runtime: it shows that an Apple GPU can take part in
FatQat's statevector simulation without losing any accuracy, and measures what
it gains.

## The problem

Apple GPUs have no FP64 arithmetic, and Metal Shading Language has no `double`
type. FatQat computes in complex128 throughout, and its rule for any speed-up is
accuracy equal to or better than the CPU's. Double-float (two float32 per value,
about 48 bits) would break that rule.

## What the prototype does

- **Binary64 in software** ([fp64.metal](fp64.metal)). IEEE multiply and add are
  built from 64-bit integer arithmetic. They round to nearest even and handle
  subnormals, signed zeros, infinities and NaN.
  - Checked against the CPU on 8 million random and targeted operations,
    every non-NaN result is bit-identical: 0 mismatches
    ([metal-fp64-check.json](../../results/metal-fp64-check.json)).
  - The targeted operations cover the subnormal boundary, the overflow edge,
    ties after alignment shifts, and cancellation into subnormals.
  - NaN results are NaN, but which payload wins when both operands are NaN
    is not matched.
- **The gate-tile kernel on that arithmetic** ([tiles.metal](tiles.metal)). It
  is FatQat's CUDA tile kernel, ported to Metal with the same descriptors
  (`_TileQueue`, `_TileGate`). It works in the operation order of Numba's
  cache tiles, so its results equal theirs bit for bit, signs of zero
  included. Numba's own per-gate kernel can differ from its tiles in the sign
  of a zero.
- **The CPU and GPU sharing one state** ([metal_engine.py](metal_engine.py)).
  - The state is a single shared `MTLBuffer` that NumPy and Numba use with no
    copy.
  - Each tile batch is split: the GPU takes 40% of the tiles while Numba runs
    the rest of the same buffer at the same time.
  - Gates that cannot tile run on Numba.
  - The one buffer is the state, freed when the last array viewing it is gone.
    Peak memory while running is within 3 MiB of Numba's
    ([memory-check.json](../../results/memory-check.json)); loading the Metal
    runtime adds about 35 MiB once.
- **The bridge** ([engine.swift](engine.swift)) is a small Swift library loaded
  with `ctypes`.

## Measured

Same process, arms alternating, medians of 3. The baseline is Numba alone
with its cache tiles, on the workloads of `perf/tile_check.py`
([metal-prototype.json](../../results/metal-prototype.json)):

| Workload | 24 qubits | 26 qubits |
| --- | ---: | ---: |
| QFT | 1.27× | 1.24× |
| Clifford+T adder | 1.39× | 1.48× |
| QAOA | 1.29× | 1.30× |
| random Clifford+T | 1.41× | 1.44× |

Every final state had the same bit patterns as Numba's. So did a check with
conjugated gates (entries `1 - 0j`) on a state full of `-0.0`.

## What did not work

These were measured while developing the split design and dropped. They are
not kept in a results file:

- **The GPU alone.** With software binary64, the GPU alone was slower than
  Numba's tiles at 24-26 qubits (0.55-0.95×). It won only at 20 qubits, where
  Numba does not tile (1.38-1.92×).
- **Exact gates only on the GPU.** Sending only the gates that need no
  arithmetic (X, CX, Toffoli) to the GPU, one pass per gate, was 1.14-1.46× on
  a reversible adder. It was slower on mixed circuits, because every switch
  between devices breaks the CPU's tile batches.

## Build and run (macOS, Xcode command-line tools)

```sh
swiftc -O -emit-library prototypes/metal/engine.swift -o prototypes/metal/libfqmetal.dylib
python prototypes/metal/metal_engine.py --qubits 24 26 --out results/metal-prototype.json
swiftc -O prototypes/metal/fp64check.swift -o prototypes/metal/fp64check
prototypes/metal/fp64check prototypes/metal/fp64.metal results/metal-fp64-check.json
```

## Before it could be a runtime

- **A packaged bridge.** It would need PyObjC's Metal bindings as an optional
  extra in place of a library you build yourself.
- **A tuned share.** The 40% split was tuned by hand on one machine; the share
  should adapt as it runs.
- **Tests.** The tests would need to run in CI on an Apple-silicon runner.
