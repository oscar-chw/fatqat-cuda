"""Prototype Apple-GPU (Metal) statevector engine: Numba plus Metal gate tiles.

Apple GPUs have no FP64 arithmetic, so the Metal tile kernel (tiles.metal)
does binary64 in software (fp64.metal): round to nearest even, subnormals,
zeros, infinities and NaN, in the Numba engine's own operation order. The
state is one shared MTLBuffer that NumPy and Numba use with no copy. Each tile
batch is split: the GPU takes a share of the tiles while Numba runs the rest
of the same buffer at the same time; gates that cannot tile run on Numba.
Values equal Numba's bit for bit.

Build the bridge first (macOS, Xcode command-line tools):
    swiftc -O -emit-library prototypes/metal/engine.swift -o prototypes/metal/libfqmetal.dylib
Then, from the repository root:
    python prototypes/metal/metal_engine.py --qubits 24 --out results/metal-prototype.json
"""

import ctypes
import statistics
import sys
import time
import weakref
from pathlib import Path

import numpy as np
from numba import njit, prange

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "perf"))

# pylint: disable=wrong-import-position  # perf/ must be on the path first
from fatqat.simulator import Simulator  # noqa: E402
from fatqat.simulator._engine import nb  # noqa: E402
from fatqat.simulator._engine.nb import NumbaSVEngine  # noqa: E402

lib = ctypes.CDLL(str(HERE / "libfqmetal.dylib"))
lib.fq_metal_alloc.restype = ctypes.c_void_p
lib.fq_metal_alloc.argtypes = [ctypes.c_size_t]
lib.fq_metal_free.argtypes = [ctypes.c_void_p]
lib.fq_metal_tiles.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
    ctypes.c_uint32, ctypes.c_uint32,
]  # fmt: skip
if lib.fq_metal_init(str(HERE / "tiles.metal").encode()) != 0:
    raise RuntimeError("the Metal tile kernel did not compile; see stderr")


@njit(cache=True, parallel=True)
def _cpu_tiles(
    state, tile_bits, rest_bits, first, last, codes, widths, targets, matrices,
    columns, values, fixed_places, fixed_values, fixed_counts, rest,
):  # fmt: skip
    """nb._apply_tiles over tiles [first, last) only; the GPU runs the others.

    A copy, so that the prototype leaves the package unchanged.
    """
    size = 1 << tile_bits.shape[0]
    offsets = np.empty(size, dtype=np.int64)
    for j in range(size):
        offsets[j] = nb._deposit(j, tile_bits)
    for t in prange(first, last):  # pylint: disable=not-an-iterable
        base = nb._deposit(t, rest_bits)
        tile = np.empty(size, dtype=np.complex128)
        for j in range(size):
            tile[j] = state[base + offsets[j]]
        nb._tile_gates(
            tile, base, codes, widths, targets, matrices, columns, values,
            fixed_places, fixed_values, fixed_counts, rest,
        )  # fmt: skip
        for j in range(size):
            state[base + offsets[j]] = tile[j]


def bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal bit patterns: unlike ``np.array_equal``, tells -0.0 from +0.0."""
    return np.array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))


def residual_kind(matrix):
    """(kind, permutation code, fixed rows) as the CUDA tile kernel reads them."""
    nonzero = matrix != 0
    if not np.any(matrix[~np.eye(len(matrix), dtype=bool)]):
        return 0, -1, 0
    if (
        np.all(nonzero.sum(axis=1) == 1)
        and np.all(nonzero.sum(axis=0) == 1)
        and len(matrix) <= 4
    ):
        columns = nonzero.argmax(axis=1)
        permutation = sum(int(c) << (2 * r) for r, c in enumerate(columns))
        fixed = sum(
            1 << r for r, c in enumerate(columns) if r == c and matrix[r, c] == 1
        )
        return 1, permutation, fixed
    return 2, -1, 0


class MetalSVEngine(NumbaSVEngine):
    _TILE_BITS = 11
    _COALESCED_BITS = 5
    _TILE_MIN_BYTES = 0
    gpu_share = 0.4  # fraction of each batch's tiles run on the GPU

    def _allocate(self, size, initial_state):
        pointer = lib.fq_metal_alloc(size * 16)
        if not pointer:
            raise MemoryError("the Metal buffer could not be allocated")
        doubles = np.ctypeslib.as_array(
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_double)), shape=(2 * size,)
        )
        # Free the buffer once nothing views it: views share this base.
        weakref.finalize(doubles, lib.fq_metal_free, pointer)
        array = doubles.view(np.complex128)
        if initial_state is None:
            array[:] = 0
            array[0] = 1
        else:
            array[:] = initial_state
        self._pointer = pointer
        return array

    def _apply_tile_batch(self, steps):
        state = self._raw_state
        if state.ctypes.data != self._pointer:
            raise RuntimeError("a kernel replaced the Metal-backed state")
        tile, rest = self._tile_bits(steps)
        position = {q: i for i, q in enumerate(tile)}
        descriptors, masks, matrices, offset = [], [], [], 0
        for step in steps:
            form = self._tile_form_of(step)
            gate = self._tile_gate(step, position)
            masks += [gate.rest_mask, gate.rest_value]
            if form.diagonal:
                kind, permutation, fixed = 3, -1, 0
            else:
                kind, permutation, fixed = residual_kind(form.matrix)
            targets = list(gate.targets) + [0] * (3 - len(gate.targets))
            places = [p for p, _ in gate.fixed] + [0] * (3 - len(gate.fixed))
            values = [v for _, v in gate.fixed] + [0] * (3 - len(gate.fixed))
            descriptors += [
                kind,
                len(gate.targets),
                *targets,
                offset,
                permutation,
                fixed,
                len(gate.fixed),
                *places,
                *values,
            ]
            matrices.append(np.asarray(form.matrix, dtype=np.complex128).ravel())
            offset += matrices[-1].size
        matrices = np.ascontiguousarray(np.concatenate(matrices))
        gates = np.array(descriptors, dtype=np.int32)
        masks = np.array(masks, dtype=np.uint64)
        tile = np.array(tile, dtype=np.int32)
        rest = np.array(rest, dtype=np.int32)
        total = 1 << len(rest)
        on_gpu = int(round(total * self.gpu_share))
        if on_gpu:
            lib.fq_metal_tiles(
                self._pointer,
                matrices.ctypes.data,
                matrices.nbytes,
                gates.ctypes.data,
                len(steps),
                masks.ctypes.data,
                tile.ctypes.data,
                rest.ctypes.data,
                len(rest),
                0,
                on_gpu,
            )
        try:
            if on_gpu < total:
                self._cpu_share(steps, tile, rest, on_gpu, total)
        finally:
            lib.fq_metal_wait()  # never leave the GPU writing past this batch

    def _cpu_share(self, steps, tile, rest, first, last):
        """The remaining tiles on the CPU, with the Numba engine's descriptors."""
        position = {int(q): i for i, q in enumerate(tile)}
        count = len(steps)
        codes = np.empty(count, dtype=np.int64)
        widths = np.empty(count, dtype=np.int64)
        targets = np.zeros((count, 3), dtype=np.int64)
        matrices = np.zeros((count, 4, 4), dtype=np.complex128)
        columns = np.zeros((count, 4), dtype=np.int64)
        values = np.zeros((count, 8), dtype=np.complex128)
        fixed_places = np.zeros((count, 3), dtype=np.int64)
        fixed_values = np.zeros((count, 3), dtype=np.int64)
        fixed_counts = np.zeros(count, dtype=np.int64)
        rest_controls = np.zeros((count, 2), dtype=np.int64)
        for g, step in enumerate(steps):
            form = self._tile_form_of(step)
            gate = self._tile_gate(step, position)
            widths[g] = len(gate.targets)
            targets[g, : widths[g]] = gate.targets
            fixed_counts[g] = len(gate.fixed)
            for i, (place, value) in enumerate(gate.fixed):
                fixed_places[g, i], fixed_values[g, i] = place, value
            rest_controls[g] = gate.rest_mask, gate.rest_value
            if form.diagonal:
                codes[g] = nb._GLOBAL_DIAGONAL
                values[g, : len(form.matrix)] = form.matrix
                continue
            code, step_columns, step_values = self._resolve_residual(step, form)
            dim = 1 << widths[g]
            codes[g] = code
            matrices[g, :dim, :dim] = form.matrix
            columns[g, :dim] = step_columns
            values[g, :dim] = step_values
        _cpu_tiles(
            self._raw_state,
            tile.astype(np.int64),
            rest.astype(np.int64),
            first,
            last,
            codes,
            widths,
            targets,
            matrices,
            columns,
            values,
            fixed_places,
            fixed_values,
            fixed_counts,
            rest_controls,
        )


def evolve(cls, n, plan):
    engine = cls()
    engine.initialize((2,) * n)
    for step in plan:
        engine.apply(step)
    return engine.state


def check_signed_zeros(n: int = 20) -> bool:
    """Conjugated gates (entries 1 - 0j) on a state with -0.0 parts."""
    from fatqat._backends.steps import (
        ApplyMatrixStep,
    )  # pylint: disable=import-outside-toplevel

    x = np.array([[0, 1], [1, 0]], dtype=np.complex128).conj()
    cx = np.eye(4, dtype=np.complex128)[[0, 1, 3, 2]].conj()
    steps = [
        ApplyMatrixStep(x, (3,)),
        ApplyMatrixStep(cx, (6, 2)),
        ApplyMatrixStep(x, (6,)),
    ]
    rng = np.random.default_rng(0)
    initial = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    initial[rng.random(1 << n) < 0.3] = complex(-0.0, -0.0)
    states = []
    for cls in (NumbaSVEngine, MetalSVEngine):
        engine = cls()
        engine.initialize((2,) * n, initial_state=initial)
        for step in steps:
            engine.apply(step)
        states.append(np.array(engine.state))
    return bit_equal(*states)


def main():
    import argparse  # pylint: disable=import-outside-toplevel
    import json  # pylint: disable=import-outside-toplevel
    from datetime import date  # pylint: disable=import-outside-toplevel

    from tile_check import WORKLOADS, revision  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qubits", type=int, nargs="+", default=[24, 26])
    parser.add_argument("--shares", type=float, nargs="+", default=[0.4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    signed_zeros = check_signed_zeros()
    print(
        f"signed zeros and conjugated gates: bit-identical={signed_zeros}", flush=True
    )
    rows = []
    for n in args.qubits:
        for name, build in WORKLOADS.items():
            plan, _ = Simulator("statevector", runtime="numpy")._lower_program(build(n))
            cpu = evolve(NumbaSVEngine, n, plan).copy()
            arms = {"numba": NumbaSVEngine}
            for share in args.shares:
                arms[f"metal_{share:.2f}"] = type(
                    "Metal", (MetalSVEngine,), {"gpu_share": share}
                )
            identical = all(
                bit_equal(cpu, evolve(cls, n, plan)) for cls in arms.values()
            )
            times = {arm: [] for arm in arms}
            for _ in range(args.repeats):
                for arm, cls in arms.items():
                    start = time.perf_counter()
                    evolve(cls, n, plan)
                    times[arm].append(time.perf_counter() - start)
            medians = {arm: statistics.median(t) for arm, t in times.items()}
            row = {
                "workload": name,
                "qubits": n,
                "steps": len(plan),
                "identical_to_numba": identical,
                "median_s": medians,
                "speedup_over_numba": {
                    arm: medians["numba"] / m
                    for arm, m in medians.items()
                    if arm != "numba"
                },
            }
            rows.append(row)
            print(
                f"{name:11s} {n} qubits identical={identical} "
                + " ".join(
                    f"{a} {r:.2f}x" for a, r in row["speedup_over_numba"].items()
                ),
                flush=True,
            )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "description": "Prototype Metal engine (software binary64 tiles on the Apple GPU, sharing each tile batch with Numba on the CPU) against Numba alone; same process, arms alternating.",
                    "measurement_date": date.today().isoformat(),
                    "code_revision": revision(),
                    "equality": "bit patterns (tells -0.0 from +0.0), against Numba with its cache tiles",
                    "signed_zeros_and_conjugated_gates_bit_identical": signed_zeros,
                    "rows": rows,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
