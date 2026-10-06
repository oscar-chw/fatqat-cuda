# Experiments: what was measured, with which parameters, and how to rerun it

Every number in the README comes from one of the experiments below. Each row
gives:

- the question the experiment answers;
- its parameters;
- the command that produced it;
- its result;
- the file holding the data.

Machines are described only by their run configuration ("one GPU", "Numba at
its default thread count").

Shared conventions:

- **Times.** Medians of warm calls, with arms alternating in one process, so
  drift and background load hit every arm alike.
- **Speed-ups.** Baseline median divided by candidate median.
- **Accuracy.** Maximum absolute error over amplitudes, in float64 machine
  epsilons (2⁻⁵²).

## r11 (this branch)

| Experiment | Question | Parameters | Result | File |
| --- | --- | --- | --- | --- |
| Shot branching, CPU | Do shots that share a state, sharing the work on it, give the same counts faster? | NumPy and Numba, 16 and 18 qubits, 200 shots, 3 repeats, arms alternating in one process; workloads: an ideal circuit with three mid-circuit measurements and conditioned gates (`feedforward`), depolarizing noise after every gate at p = 0.002 (`low_noise`) and p = 0.05 (`high_noise`); baseline: every shot evolved alone (the loop before r11). Numba's per-shot path is forced (`kernel_parallelism="threads"`); its default compiled multi-shot kernel is not affected | NumPy 11.3–12.0× (feedforward), 5.6–6.1× (low noise), 1.21–1.31× (high noise); Numba 1.26–71×; never slower; counts equal on every repeat | [branching-check.json](../results/branching-check.json) |
| Shot branching and two GPUs | Does branching pay on a GPU, and do two GPUs beat one for a run's shots? | 20–26 qubits, 256 shots, the three CPU workloads; arms: one GPU with every shot alone (20–24 qubits only), one GPU with branching, two GPUs with branching (one worker process for the second); 5 repeats after a warm-up, arm order alternating each repeat; CPU work pinned to a fixed set of cores | branching on one GPU: 19–77× (feedforward), 1.9–4.9× (low noise), 1.16–1.17× (high noise); two GPUs vs one: 1.64–1.90× on the noisy runs, 0.98–1.07× on feedforward; never slower than the 0.95 noise floor; counts equal in every arm; peak GPU memory 17.1 GiB | [shots-cuda.json](../results/shots-cuda.json) |
| Differential check | Do the "same result" claims hold on circuits the tests never saw? | The 9,500 circuits of r10's check, plus 10,000 random engine-level plans (2,500 each on NumPy and Numba, statevector and density matrix) with every step kind: both channel routes, multi-term conditions, readout confusion, remapped digits, reset, loss and reload, qubits and qutrits, given initial states and small memory budgets; branching against the one-shot loop, every shot's classical bits | 0 failures | [differential-check.json](../results/differential-check.json) |
| Precision, r11 | Did any r11 change add round-off? | The 110 circuits of r10's precision check, every runtime | identical to r10 on every circuit and runtime | [precision-r11.json](../results/precision-r11.json) |

## r10

| Experiment | Question | Parameters | Result | File |
| --- | --- | --- | --- | --- |
| Simplification, CPU | Is `simplify=True` closer to the *ideal* circuit, and is it faster? | **Accuracy:** 48 circuits (4 families × 12 seeds) on 6 qubits: redundant Clifford+T (120 gates), plain Clifford+T (120), a 2-bit Clifford+T adder, QAOA (2 layers); the reference is the exact gate definitions evolved at 60 digits. **Timing:** 20 qubits, NumPy and Numba, 5 repeats; adder, redundant Clifford+T (1,200 gates), QAOA, rotated QFT as control. | 0.78 vs 2.92 eps (NumPy), 0.77 vs 2.89 (Numba); 1.8–4.3× faster; control unchanged | [simplify-check.json](../results/simplify-check.json) |
| Simplification, GPU | The same on one GPU | Accuracy 24 circuits (4 × 6 seeds); timing at 26 qubits, 3 repeats, exact expectation values (state left on the device) | 0.81 vs 3.08 eps; up to 2.75× at 26 qubits; pays only on large states | [simplify-check-cuda.json](../results/simplify-check-cuda.json) |
| Tiles, CPU | Do tiles where controls and diagonal gates take no tile bit beat tiles where every target does? | Numba, 24 qubits, 3 repeats; QFT (after a rotation layer), Cuccaro adder in Clifford+T, QAOA (ring plus n random edges, 2 layers), random Clifford+T (600 gates); arms: per-gate, every-target tiles, insular tiles; pass counts recorded | 1.48–1.68× over every-target tiles, 2.2–2.5× over per-gate; QFT 27 → 5 passes; all states equal | [tile-check.json](../results/tile-check.json) |
| Tiles, GPU | The same on one GPU | 26 qubits, 3 repeats, timed to a device sync | 1.06–2.48× over every-target tiles, 1.7–4.1× over per-gate; QFT 47 → 7 passes; all states equal | [tile-check-cuda.json](../results/tile-check-cuda.json) |
| Precision, r10 | Is every runtime still accurate after the CUDA arithmetic change (Kahan's 2×2-determinant products)? | 110 circuits: 11 families × 10 seeds, statevector, density matrix, unitary and superoperator, qubit and mixed radix, up to 5 subsystems; 60-digit oracle of the stored coefficients | every runtime ≤ 2.9 eps; Numba minus GPU +0.047 eps (SE 0.033) | [precision-r10.json](../results/precision-r10.json) |
| Differential check | Do the "same result" claims hold on circuits the tests never saw? | 9,500 random circuits from seed 1,000,000. Covers:<br>- tiles vs per-gate (3,000, 7–12 qubits);<br>- exact simplifications (3,000, qubit and mixed radix);<br>- known-input specialisation (2,000);<br>- Clifford+T simplification to 1e-12 (1,200);<br>- Metal prototype vs Numba tiles, by bit pattern (300). | 0 failures | [differential-check.json](../results/differential-check.json) |
| Memory | Does any r10 change use more host memory? | 24 qubits (a 256 MiB state), each tile-check workload, one fresh process per arm: per-gate, every-target tiles, insular tiles, insular tiles with `simplify`, the Metal prototype; peak resident memory added while evolving the plan | 287–290 MiB for every arm; no change beyond 3 MiB. Loading the Metal runtime adds about 35 MiB once, measured separately | [memory-check.json](../results/memory-check.json) |
| Software binary64 (Apple GPU) | Can an Apple GPU do exact IEEE double arithmetic in software? | 4M multiplies and 4M adds: random operands of every class, plus pairs at the subnormal boundary, the overflow edge, ties after alignment shifts, and subnormal cancellation; a one-qubit gate on 2²⁰ amplitudes against the CPU loop | 0 mismatches on non-NaN results; gate bit-identical | [metal-fp64-check.json](../results/metal-fp64-check.json) |
| Metal prototype | Does sharing each tile batch between the Apple GPU and the CPU beat the CPU alone? | 24 and 26 qubits; GPU share 0.4 of each batch's tiles (11-bit tiles); 3 repeats; the tile-check workloads; baseline Numba with its 12-bit tiles | 1.24–1.48×; every state bit-identical to Numba's tiles | [metal-prototype.json](../results/metal-prototype.json) |

GPU memory was recorded for r9 (pool and host memory equal between r8 and r9,
[ab-r8-r9.json](../results/ab-r8-r9.json)); r10's CUDA tiles add no global
allocations beyond the small per-batch descriptors, but their GPU pool was not
re-measured.

The CPU tile and simplification files were measured at their own revisions.
At those revisions, timed calls also exported the state; the scripts now time
the kernels alone (a device sync, or an expectation value). The GPU files were
measured that way.

## r9 (public)

| Experiment | Parameters | Result | File |
| --- | --- | --- | --- |
| Precision | the 110 circuits above, r8 code | ≤ 3.1 eps every runtime; GPU vs Numba −0.045 ± 0.031 eps | [precision.json](../results/precision.json) |
| GPU vs CPU scaling | statevector, density matrix, unitary, observables; 6–28 qubits; 5 warm calls; CPU = Numba at `NUMBA_NUM_THREADS=32` (a fixed setting, not every core) | 35–42× (24–28-qubit observables) | [scaling-r9.json](../results/scaling-r9.json), [scaling-r9-cpu.json](../results/scaling-r9-cpu.json), [scaling.json](../results/scaling.json) |
| r9 vs r8, one GPU | same-run A/B, arms alternating, GPU pool and host memory recorded | 2.09–2.22× at 24–28 qubits, same memory | [ab-r8-r9.json](../results/ab-r8-r9.json) |
| Unitary tiles | same-run A/B, 11–14 qubits | 1.28–1.42× at 12–14 | [ab-unitary-tiles.json](../results/ab-unitary-tiles.json) |
| CPU cache tiles | engine apply loop, 22–26 qubits, two CPU settings | 1.23–1.55× | [cpu-tiles.json](../results/cpu-tiles.json) |
| Unit-gate simplification | synthetic circuit with appended cancelling pairs | CPU 1.7–1.9×, GPU 1.0–1.7× | [simplify.json](../results/simplify.json) |

## Tried and dropped

Measured during development; these numbers are not kept in results files.

- **Merging rounded gates in floating point.** Merging `H·H` or `RY·RZ` as
  float matrices was less accurate: about 5× the mean error for `H·H`, and
  +0.015 eps for `RY·RZ` on CUDA. This is why r10 merges in exact algebra
  instead.
- **Plain arithmetic on the GPU.** Turning fused multiply-add off was slightly
  worse (+0.049 eps, SE 0.024). Unfused, or one product fused, each lost a deep
  4,096-gate case to a CPU engine. Kahan's algorithm replaced both.
- **Finer column blocks for the CPU unitary engine:** 0.38–0.91×.
- **12-bit CUDA tiles:** slower than 11-bit ones.
- **Apple GPU alone, with software binary64:** 0.55–0.95× at 24–26 qubits. It
  won only at 20 qubits, where Numba does not tile.
- **Only exact gates on the Apple GPU:** 1.14–1.46× on a reversible adder, and
  slower on mixed circuits.
- **Several GPUs for one sweep:** works, and every test passes on two GPUs. It
  scales less than linearly, because each row's Python-side work runs one
  thread at a time, so no speed figure is published.
- **Threads for a run's shots on several GPUs:** 0.67–0.80× of one GPU at 20
  qubits (1.35–1.74× at 22), because each shot's loop is mostly Python and
  threads take turns on the interpreter lock. One worker process per further
  GPU replaced them.
- **Building every measurement outcome's state at once** in shot branching:
  about 250 copies of a 24-qubit state on a GPU, where the outcomes now share one state.
- **Letting the first shot's group go on in place** in shot branching: when
  that shot drew a rare noise branch, the large no-error group waited,
  overflowed the memory budget and ran shot by shot (396 of 400 shots in a
  test). Two GPUs measured 0.46× of one at 24 qubits because of it, and a
  fixed 1 GiB budget made 26-qubit runs 30× slower than they are now.

## Not tried

- **Clustering gates into larger fused matrices** (three or more qubits, as
  qsim and Qiskit Aer do). Each fused block rounds once per entry it sums over,
  so wider blocks round more, and on these engines tiles were already
  compute-bound. Tiles group gates into one pass over memory *without* changing any gate's
  arithmetic instead.
- **Splitting one state across machines** (a compute cluster, MPI). Several
  GPUs are used only for independent sweep rows and shots on one machine; the
  largest state is bounded by one device's memory.
- **More than two GPUs for a run's shots.** Tests run on two; scaling to more
  was not measured.
- **Optimised libraries** (cuStateVec, Qiskit Aer, qsim). Each was checked on
  paper against the rule that no change may add round-off: qsim computes in
  single precision, Aer fuses gates by default, and cuStateVec documents
  neither its rounding nor its determinism, so none was adopted untested.
- **An all-cores CPU baseline for r9.** It is planned, and until then the
  35–42× headline names its 32-thread setting.

## Rerun

```sh
python perf/simplify_check.py --out s.json          # add --runtimes cuda --qubits 26 on a GPU
python perf/tile_check.py --runtimes numba --out t.json
python perf/precision.py --require-gpu --out p.json
python perf/differential_check.py --out d.json
python perf/branching_check.py --runtimes numpy numba --qubits 16 18 --out b.json
python perf/shots_check.py g.json 20 22 24             # on a machine with 2+ GPUs; then --no-per-shot 26
python perf/scrub_check.py s.json t.json p.json d.json b.json   # before publishing any output
```

The Metal prototype has its own instructions in
[prototypes/metal/README.md](../prototypes/metal/README.md).
