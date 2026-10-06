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
  Checked against the CPU on 8 million random and edge-case operations, the
  output is bit-identical in every case: 0 mismatches
  ([metal-fp64-check.json](../../results/metal-fp64-check.json)).
- **The gate-tile kernel on that arithmetic** ([tiles.metal](tiles.metal)). It
  is FatQat's CUDA tile kernel, ported to Metal with the same descriptors
  (`_TileQueue`, `_TileGate`). It works in the Numba engine's own operation
  order, so the values equal Numba's bit for bit.
- **The CPU and GPU sharing one state** ([metal_engine.py](metal_engine.py)).
  - The state is a single shared `MTLBuffer` that NumPy and Numba use with no
    copy.
  - Each tile batch is split: the GPU takes 40% of the tiles while Numba runs
    the rest of the same buffer at the same time.
  - Gates that cannot tile run on Numba.
  - Memory use is unchanged.
- **The bridge** ([engine.swift](engine.swift)) is a small Swift library loaded
  with `ctypes`.

## Measured

Same process, arms alternating, medians of 3. The baseline is Numba alone
with its cache tiles, on the workloads of `perf/tile_check.py`
([metal-prototype.json](../../results/metal-prototype.json)):

| Workload | 24 qubits | 26 qubits |
| --- | ---: | ---: |
| QFT | 1.28× | 1.29× |
| Clifford+T adder | 1.42× | 1.45× |
| QAOA | 1.34× | 1.29× |
| random Clifford+T | 1.38× | 1.37× |

Every final state was equal to Numba's, bit for bit.

## What did not work

These were measured before the split design and dropped:

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
