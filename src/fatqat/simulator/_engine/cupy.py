"""Complex128 CUDA kernels for state-vector, density and operator execution.

CuPy is loaded only when numerical work starts. Circuit lowering, terminal
measurement reporting, jobs, and public results retain the NumPy semantics.
Public calls own a selected-device execution scope; private numerical primitives
require that scope (or an explicit caller-owned device context).
"""

from contextlib import contextmanager
from functools import cached_property
from importlib import import_module
from math import fsum

import numpy as np

from ...errors import BackendValidationError
from .base import _TileQueue
from ...implementation.matrices import shift_matrix
from .np import (
    NumpySVEngine,
    NumpyDMEngine,
    NumpyUnitaryEngine,
    NumpySuperopEngine,
    _contract_local,
    _digit,
    _strides,
)

# Complex multiply and multiply-add with every rounding spelled out. Left to
# the compiler, which product of ``m.x*a.x - m.y*a.y`` is fused into an FMA
# differs between kernels, so equal source in two kernels can differ in the
# last bit; every kernel that must agree with another (the per-gate qubit
# kernels and the gate tiles) uses these. Each part of the product is a 2x2
# determinant, computed with Kahan's algorithm: the rounding error of one
# product is recovered exactly with an FMA, so the part is within about 2 ulp
# even under cancellation (Jeannerod, Louvet and Muller, Math. Comp. 82, 2013).
# Plain arithmetic was measured worse than one CPU engine or the other on deep
# circuits: unfused like Numba on 4096 alternating gates and their adjoints,
# one product fused (the compiler's choice) on 4096 repeated rotations.
_COMPLEX_OPS = r"""
__device__ __forceinline__ double determinant(double a, double b, double c, double d) {
    // a*b - c*d
    const double w = __dmul_rn(c, d);
    const double error = __fma_rn(-c, d, w);  // w - c*d, exactly
    return __dadd_rn(__fma_rn(a, b, -w), error);
}
__device__ __forceinline__ double2 cmul(double2 m, double2 a) {
    return make_double2(determinant(m.x, a.x, m.y, a.y),
                        determinant(m.x, a.y, -m.y, a.x));
}
__device__ __forceinline__ double2 cmul_add(double2 acc, double2 m, double2 a) {
    const double2 t = cmul(m, a);
    return make_double2(__dadd_rn(acc.x, t.x), __dadd_rn(acc.y, t.y));
}
"""
# simulation_config gpu_products="plain": each part of a product is one FMA
# on one rounded product, as the CPU engines' compiled arithmetic can fuse it,
# still spelled out so the per-gate kernels and the tiles agree bit for bit.
# About 1.4-1.5x faster on observables at 24-28 qubits (results/scaling-r15-
# plain.json against scaling-r14.json); on the 110 circuits of
# perf/precision.py no less accurate on average (+0.03 +- 0.03 eps) than the
# compensated products, but without their per-product bound under
# cancellation (one circuit 1.5 eps worse; results/precision-r15-*.json).
_PLAIN_COMPLEX_OPS = _COMPLEX_OPS.replace(
    """    const double w = __dmul_rn(c, d);
    const double error = __fma_rn(-c, d, w);  // w - c*d, exactly
    return __dadd_rn(__fma_rn(a, b, -w), error);""",
    """    return __fma_rn(a, b, -__dmul_rn(c, d));""",
)
assert _PLAIN_COMPLEX_OPS != _COMPLEX_OPS


class _CupyRuntime:
    """Device ownership shared by CUDA matrix-family engines."""

    _supports_shot_workers = False
    # Whether this engine's gate kernels use _complex_ops (gpu_products):
    # the statevector and unitary engines; the density-matrix and
    # superoperator ones apply gates through their own sandwich kernels.
    _uses_gpu_products = False

    def __init__(self, name, *, device_id: int = 0):
        super().__init__(name)
        if type(device_id) is not int or device_id < 0:
            raise BackendValidationError("device_id must be a nonnegative int")
        self.device_id = device_id
        self._matrix_cache = {}
        self._kernels = {}
        # "compensated" or "plain": simulation_config gpu_products, set by
        # the Simulator before each run.
        self.gpu_products = "compensated"

    def _complex_ops(self) -> str:
        if self.gpu_products == "plain":
            return _PLAIN_COMPLEX_OPS
        return _COMPLEX_OPS

    @cached_property
    def _cp(self):
        try:
            return import_module("cupy")
        except ImportError as exc:
            raise BackendValidationError(
                "runtime='cuda' requires CuPy and a compatible NVIDIA CUDA device; install fatqat[cuda13] or fatqat[cuda12] for your CUDA Toolkit"
            ) from exc

    @contextmanager
    def _execution_scope(self, policy):
        del policy
        with self._cp.cuda.Device(self.device_id):
            try:
                yield
            finally:
                try:
                    # Eager Jobs finish device work even when no host output
                    # was requested, and capture asynchronous kernel failures.
                    self._cp.cuda.get_current_stream().synchronize()
                finally:
                    self._matrix_cache.clear()

    def export_state(self):
        with self._cp.cuda.Device(self.device_id):
            return self._cp.asnumpy(self.state)

    def _free_memory_bytes(self):
        """Free device memory (CUDA's, plus this pool's cached free blocks)."""
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            free, _total = cp.cuda.runtime.memGetInfo()
            return free + cp.get_default_memory_pool().free_bytes()

    def _release_device_memory(self):
        """Return this device's cached free blocks to CUDA (a GPU worker,
        between runs, so an idle worker holds no state-sized block)."""
        with self._cp.cuda.Device(self.device_id):
            self._cp.get_default_memory_pool().free_all_blocks()

    def expectation_values(self, state, observables):
        """Reduce on device; transfer bounded partial sums for CPU fsum.

        Each GPU thread uses compensated binary64 accumulation, then each
        block uses a fixed pairwise tree. No atomic floating-point updates or
        fast-math are used. CPU mask preparation overlaps queued device work;
        the host receives at most 4096 doubles per term, never the full state.
        Density expectations read only the shifted diagonal needed by each
        term, using the actual row/column strides without a matrix copy.
        """
        from ..._expectation import _term_masks

        cp = self._cp
        density = self.state_semantics == "dm"
        size = state.shape[0]
        if not density and not state.flags.c_contiguous:
            raise ValueError("resident expectation requires a contiguous state")
        with cp.cuda.Device(self.device_id):
            key = "expectation"
            if key not in self._kernels:
                source = r"""
                extern "C" __global__ void expectation_partial(
                    const double2* state, double* partial,
                    unsigned long long size, unsigned long long xmask,
                    unsigned long long zmask, unsigned long long zeros,
                    unsigned long long ones, int phase, int density,
                    long long row_stride, long long column_stride) {
                    __shared__ double sums[256];
                    unsigned int lane = threadIdx.x;
                    unsigned long long start = (unsigned long long)blockIdx.x * blockDim.x + lane;
                    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
                    double sum = 0.0, correction = 0.0;
                    for (unsigned long long i = start; i < size; i += stride) {
                        unsigned long long j = i ^ xmask;
                        unsigned long long weight_index = density ? i : j;
                        if ((weight_index & ones) != ones || (weight_index & zeros) != 0) continue;
                        double value;
                        if (density) {
                            double2 entry = state[(long long)i * row_stride + (long long)j * column_stride];
                            value = (phase & 1) ? entry.y : entry.x;
                        } else {
                            double2 a = state[i], b = state[j];
                            value = (phase & 1) ? a.x*b.y - a.y*b.x : a.x*b.x + a.y*b.y;
                        }
                        if (phase == 1 || phase == 2) value = -value;
                        if (__popcll(weight_index & zmask) & 1) value = -value;
                        double adjusted = value - correction;
                        double next = sum + adjusted;
                        correction = (next - sum) - adjusted;
                        sum = next;
                    }
                    sums[lane] = sum;
                    __syncthreads();
                    for (unsigned int width = 128; width > 0; width >>= 1) {
                        if (lane < width) sums[lane] += sums[lane + width];
                        __syncthreads();
                    }
                    if (lane == 0) partial[blockIdx.x] = sums[0];
                }
                """
                self._kernels[key] = cp.RawKernel(source, "expectation_partial")
            terms = [
                (index, coefficient, factors)
                for index, observable in enumerate(observables)
                for coefficient, factors in observable
                if coefficient != 0.0
            ]
            blocks = min(4096, (size + 255) // 256)
            partials = cp.empty((len(terms), blocks), dtype=cp.float64)
            n_qubits = int(size).bit_length() - 1
            for row, (_index, _coefficient, factors) in enumerate(terms):
                xmask, zmask, zeros, ones, n_y = _term_masks(factors, n_qubits)
                self._kernels[key](
                    (blocks,),
                    (256,),
                    (
                        state,
                        partials[row],
                        np.uint64(size),
                        np.uint64(xmask),
                        np.uint64(zmask),
                        np.uint64(zeros),
                        np.uint64(ones),
                        np.int32(n_y % 4),
                        np.int32(density),
                        np.int64(state.strides[0] // state.itemsize),
                        np.int64(state.strides[1] // state.itemsize if density else 0),
                    ),
                )
            host_partials = cp.asnumpy(partials)
        values = [[] for _observable in observables]
        for row, (index, coefficient, _factors) in enumerate(terms):
            values[index].append(coefficient * fsum(host_partials[row]))
        return tuple(fsum(parts) for parts in values)

    @property
    def _xp(self):
        return self._cp

    def _matrix_array(self, matrix):
        cached = self._matrix_cache.get(id(matrix))
        if cached is None or cached[0] is not matrix:
            cached = (
                matrix,
                self._cp.array(matrix, dtype=self._cp.complex128, copy=True, order="C"),
            )
            self._matrix_cache[id(matrix)] = cached
        return cached[1]

    def _uses_qubit_kernel(self, targets):
        return len(targets) in (1, 2) and all(d == 2 for d in self._dims)

    def _qubit_matrix(self, matrix):
        device_matrix = self._matrix_array(matrix)
        cached = self._matrix_cache[id(matrix)]
        if len(cached) == 2:
            diagonal = np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0
            permutation, fixed = -1, 0
            if (
                not diagonal
                and matrix.shape[0] <= 4
                and np.count_nonzero(matrix) == matrix.shape[0]
            ):
                nonzero = matrix != 0
                columns = nonzero.argmax(axis=1)
                if not (
                    np.all(nonzero.sum(axis=1) == 1)
                    and len(set(columns)) == len(columns)
                ):
                    cached = (*cached, (diagonal, permutation, fixed))
                    self._matrix_cache[id(matrix)] = cached
                    return device_matrix, cached[2]
                permutation = sum(
                    int(column) << (2 * row) for row, column in enumerate(columns)
                )
                fixed = sum(
                    1 << row
                    for row, column in enumerate(columns)
                    if row == column and matrix[row, column] == 1
                )
            cached = (*cached, (diagonal, permutation, fixed))
            self._matrix_cache[id(matrix)] = cached
        return device_matrix, cached[2]

    def _small_qubit_apply(self, state, matrix, targets, structure):
        """Update disjoint pairs/quartets without transposing the full state."""
        cp = self._cp
        width = len(targets)
        dim = 1 << width
        diagonal, permutation, fixed = structure
        key = (width, diagonal, permutation >= 0, self.gpu_products)
        if key not in self._kernels:
            if diagonal:
                source = r"""
                extern "C" __global__ void local_gate(
                    double2* state, const double2* matrix,
                    unsigned long long size, int first, int second, int permutation, int fixed) {
                    unsigned long long i = blockDim.x * (unsigned long long)blockIdx.x + threadIdx.x;
                    if (i >= size) return;
                    int row = (i >> first) & 1;
                    if (WIDTH == 2) row = 2 * row + ((i >> second) & 1);
                    state[i] = cmul(matrix[row * DIM + row], state[i]);
                }
                """
            else:
                source = r"""
                extern "C" __global__ void local_gate(
                    double2* state, const double2* matrix,
                    unsigned long long size, int first, int second, int permutation, int fixed) {
                    unsigned long long g = blockDim.x * (unsigned long long)blockIdx.x + threadIdx.x;
                    if (g >= size / DIM) return;
                    int lo = first, hi = second;
                    if (WIDTH == 2 && lo > hi) { int tmp=lo; lo=hi; hi=tmp; }
                    unsigned long long mask = (1ULL << lo) - 1;
                    unsigned long long base = (g & mask) | ((g >> lo) << (lo + 1));
                    if (WIDTH == 2) {
                        mask = (1ULL << hi) - 1;
                        base = (base & mask) | ((base >> hi) << (hi + 1));
                    }
                    double2 values[DIM];
                    unsigned long long indices[DIM];
                    for (int r=0; r<DIM; ++r) {
                        unsigned long long offset;
                        if (WIDTH == 1) offset = (unsigned long long)r << first;
                        else offset = ((unsigned long long)(r >> 1) << first) | ((unsigned long long)(r & 1) << second);
                        indices[r] = base | offset;
                        if (!MONOMIAL || !(fixed & (1 << r))) values[r] = state[indices[r]];
                    }
                    for (int r=0; r<DIM; ++r) {
                        if (MONOMIAL) {
                            if (fixed & (1 << r)) continue;
                            int c = (permutation >> (2*r)) & 3;
                            state[indices[r]] = cmul(matrix[r*DIM+c], values[c]);
                            continue;
                        }
                        double2 sum = make_double2(0.0, 0.0);
                        for (int c=0; c<DIM; ++c)
                            sum = cmul_add(sum, matrix[r*DIM+c], values[c]);
                        state[indices[r]] = sum;
                    }
                }
                """
            source = source.replace("WIDTH", str(width)).replace("DIM", str(dim))
            source = self._complex_ops() + source.replace(
                "MONOMIAL", str(int(permutation >= 0))
            )
            # No fast-math or reduced-precision mode; ordinary binary64 ops.
            self._kernels[key] = cp.RawKernel(source, "local_gate")
        work = state.size if diagonal else state.size // dim
        self._kernels[key](
            ((work + 255) // 256,),
            (256,),
            (
                state,
                matrix,
                np.uint64(state.size),
                np.int32(targets[0]),
                np.int32(targets[-1]),
                np.int32(permutation),
                np.int32(fixed),
            ),
        )
        return state

    def _sandwich_offsets(self):
        return len(self._dims), 0

    def _qubit_sandwich(self, rho, matrix, targets, *, copy, out=None):
        """Keep each local ket/bra tile in registers, preserving Kraus branches."""
        cp = self._cp
        state = cp.ascontiguousarray(rho)
        output = (cp.empty_like(state) if copy else state) if out is None else out
        device_matrix, structure = self._qubit_matrix(matrix)
        diagonal, permutation, _fixed = structure
        width = len(targets)
        dim = 1 << width
        key = ("sandwich", width, diagonal, permutation >= 0)
        if key not in self._kernels:
            source = r"""
            __device__ void store_sandwich(double2* output, unsigned long long index,
                double2 value, int accumulate) {
                if (accumulate) {
                    double2 previous = output[index];
                    value = make_double2(previous.x+value.x, previous.y+value.y);
                }
                output[index] = value;
            }
            extern "C" __global__ void local_sandwich(
                const double2* input, double2* output, const double2* matrix,
                unsigned long long size, int row_first, int row_second,
                int col_first, int col_second, int h0, int h1, int h2, int h3,
                int permutation, int accumulate) {
                unsigned long long g = blockDim.x * (unsigned long long)blockIdx.x + threadIdx.x;
                if (DIAGONAL) {
                    if (g >= size) return;
                    int row = (g >> row_first) & 1, col = (g >> col_first) & 1;
                    if (WIDTH == 2) {
                        row = 2*row + ((g >> row_second) & 1);
                        col = 2*col + ((g >> col_second) & 1);
                    }
                    double2 a = input[g], m = matrix[row*DIM+row], b = matrix[col*DIM+col];
                    double real = m.x*a.x-m.y*a.y, imag = m.x*a.y+m.y*a.x;
                    store_sandwich(output, g, make_double2(b.x*real+b.y*imag, b.x*imag-b.y*real), accumulate);
                    return;
                }
                if (g >= size/(DIM*DIM)) return;
                int holes[2*WIDTH] = {HOLES};
                unsigned long long base = g;
                for (int bit=0; bit<2*WIDTH; ++bit) {
                    unsigned long long mask = (1ULL << holes[bit])-1;
                    base = (base & mask) | ((base >> holes[bit]) << (holes[bit]+1));
                }
                double2 values[DIM*DIM], intermediate[DIM*DIM];
                unsigned long long indices[DIM*DIM];
                for (int r=0; r<DIM; ++r) {
                    unsigned long long row = (unsigned long long)r << row_first;
                    if (WIDTH == 2) row = ((unsigned long long)(r>>1) << row_first) | ((unsigned long long)(r&1) << row_second);
                    for (int c=0; c<DIM; ++c) {
                        unsigned long long col = (unsigned long long)c << col_first;
                        if (WIDTH == 2) col = ((unsigned long long)(c>>1) << col_first) | ((unsigned long long)(c&1) << col_second);
                        indices[r*DIM+c] = base | row | col;
                        values[r*DIM+c] = input[indices[r*DIM+c]];
                    }
                }
                for (int r=0; r<DIM; ++r) {
                    for (int c=0; c<DIM; ++c) {
                        double real=0, imag=0;
                        for (int k=0; k<DIM; ++k) {
                            if (MONOMIAL && k != ((permutation >> (2*r)) & 3)) continue;
                            double2 m=matrix[r*DIM+k], a=values[k*DIM+c];
                            real += m.x*a.x-m.y*a.y;
                            imag += m.x*a.y+m.y*a.x;
                        }
                        intermediate[r*DIM+c] = make_double2(real,imag);
                    }
                }
                for (int r=0; r<DIM; ++r) {
                    for (int c=0; c<DIM; ++c) {
                        double real=0, imag=0;
                        for (int k=0; k<DIM; ++k) {
                            if (MONOMIAL && k != ((permutation >> (2*c)) & 3)) continue;
                            double2 m=matrix[c*DIM+k], a=intermediate[r*DIM+k];
                            real += m.x*a.x+m.y*a.y;
                            imag += m.x*a.y-m.y*a.x;
                        }
                        store_sandwich(output, indices[r*DIM+c], make_double2(real,imag), accumulate);
                    }
                }
            }
            """
            source = source.replace(
                "HOLES", ", ".join(f"h{i}" for i in range(2 * width))
            )
            for name, value in (
                ("WIDTH", width),
                ("DIM", dim),
                ("DIAGONAL", int(diagonal)),
                ("MONOMIAL", int(permutation >= 0)),
            ):
                source = source.replace(name, str(value))
            self._kernels[key] = cp.RawKernel(source, "local_sandwich")
        ket_offset, bra_offset = self._sandwich_offsets()
        rows = tuple(ket_offset + t for t in targets)
        columns = tuple(bra_offset + t for t in targets)
        holes = sorted((*rows, *columns))
        holes += [0] * (4 - len(holes))
        work = state.size if diagonal else state.size // (dim * dim)
        self._kernels[key](
            ((work + 255) // 256,),
            (256,),
            (
                state,
                output,
                device_matrix,
                np.uint64(state.size),
                *(
                    np.int32(bit)
                    for bit in (rows[0], rows[-1], columns[0], columns[-1], *holes)
                ),
                np.int32(permutation),
                np.int32(out is not None),
            ),
        )
        return output

    def _channel_output(self, step):
        """Accumulate ordered Kraus branches directly into one owned output."""
        source = self._cp.ascontiguousarray(self.state)
        output = self._cp.zeros_like(source)
        for kraus in step.kraus_ops:
            self._qubit_sandwich(
                source, kraus, step.target_indices, copy=False, out=output
            )
        return output

    def _apply_sandwich(self, rho, matrix, targets, *, copy):
        if self._uses_qubit_kernel(targets):
            return self._qubit_sandwich(rho, matrix, targets, copy=copy)
        return super()._apply_local_sandwich(rho, matrix, targets)


class _CupyStateRuntime(_CupyRuntime):
    """GPU probability sampling with the existing CPU random stream."""

    def probabilities(self):
        with self._cp.cuda.Device(self.device_id):
            return self._cp.asnumpy(self._device_probabilities())

    def sample_indices(self, shots, rng):
        return self._draw_indices(self._weigh_collapse(), rng.random(shots))

    def _weigh_collapse(self):
        """The normalized device cdf collapse and sampling invert."""
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            cdf = cp.cumsum(self._device_probabilities())
            total = float(cdf[-1])
            if not np.isfinite(total) or total <= 0:
                raise ValueError("probabilities must have a finite, positive sum")
            cdf /= total
            return cdf

    def _pick_index(self, weighed, rng):
        return int(self._draw_indices(weighed, rng.random(1))[0])

    def _draw_indices(self, cdf, uniforms):
        # Keep the seeded CPU random stream, transfer only uniforms and
        # sampled indices; the full probability vector stays on device.
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            return cp.asnumpy(cp.searchsorted(cdf, cp.asarray(uniforms), side="right"))


# Ints per gate in the tile kernel's descriptor (see `_apply_tile_batch`).
_TILE_GATE_INTS = 15


class _GateTiles(_TileQueue):
    """Queue runs of qubit gates and apply them tile by tile in shared memory."""

    # Gate tiles, shared by the statevector and unitary engines. Each one- or
    # two-qubit gate is otherwise one full pass over the state in global
    # memory, and those passes dominate large runs. ``_tile_layout`` places
    # the gate targets among the flat state's index bits (a unitary's gates
    # act on its row bits, above the column bits).
    # Consecutive qubit gates whose targets fit in one tile of _TILE_BITS
    # qubits are queued and applied together: each CUDA block loads one tile
    # into shared memory, applies the queued gates in order with the same
    # per-gate arithmetic as _small_qubit_apply, and writes the tile back once.
    # The tile always includes the _COALESCED_BITS lowest qubits so global
    # loads stay contiguous. 2**11 complex128 values is 32 KiB of shared
    # memory, within the default per-block limit.
    #
    # Tiling engages only for states larger than the device's L2 cache
    # (_TILE_MIN_BYTES=None). Below that, per-gate passes are served from L2
    # and were measured as fast as tiles or faster, while states several
    # times larger than L2 gained 2.2-2.4x.
    _TILE_BITS = 11
    _COALESCED_BITS = 5
    _TILE_MIN_BYTES = None

    @contextmanager
    def _execution_scope(self, policy):
        with super()._execution_scope(policy):
            yield
            # Finish queued gates before the scope synchronizes and clears the
            # matrix cache, even when the run requested no output.
            self._flush_pending()

    def _can_tile(self, targets):
        _offset, total = self._tile_layout()
        return (
            total >= self._TILE_BITS
            and (1 << total) * 16 > self._tile_min_bytes
            and len(targets) <= 3
            and all(d == 2 for d in self._dims)
        )

    def _tile_form_of(self, step):
        form = super()._tile_form_of(step)
        if form is None or len(step.target_indices) < 3:
            return form
        # Gates on three qubits otherwise take the cuBLAS contraction path,
        # whose fused rounding differs from the tile kernel's (measured: last
        # bits of a three-qubit diagonal or controlled two-qubit gate). Gates
        # whose entries are 0, +-1 and +-i round on neither path, so only
        # those (Toffoli, Fredkin, CCZ) join a tile.
        matrix = np.asarray(step.matrix)
        parts = np.concatenate((matrix.real.ravel(), matrix.imag.ravel()))
        exact = np.all(np.isin(parts, (-1.0, 0.0, 1.0))) and np.all(
            np.abs(matrix)[matrix != 0] == 1
        )
        return form if exact else None

    @cached_property
    def _tile_min_bytes(self):
        if self._TILE_MIN_BYTES is not None:
            return self._TILE_MIN_BYTES
        return self._cp.cuda.Device(self.device_id).attributes["L2CacheSize"]

    def _apply_tile_batch(self, steps):
        """Apply queued qubit gates tile by tile in shared memory."""
        cp = self._cp
        tile, rest = self._tile_bits(steps)
        position = {q: i for i, q in enumerate(tile)}
        descriptors, masks, matrices, matrix_offset = [], [], [], 0
        for step in steps:
            form = self._tile_form_of(step)
            gate = self._tile_gate(step, position)
            masks += [gate.rest_mask, gate.rest_value]
            if form.diagonal:
                kind, permutation, fixed = 3, -1, 0
            else:
                _device, (diagonal, permutation, fixed) = self._qubit_matrix(
                    form.matrix
                )
                kind = 0 if diagonal else 1 if permutation >= 0 else 2
            targets = list(gate.targets) + [0] * (3 - len(gate.targets))
            places = [p for p, _ in gate.fixed] + [0] * (3 - len(gate.fixed))
            values = [v for _, v in gate.fixed] + [0] * (3 - len(gate.fixed))
            descriptors += [
                kind, len(gate.targets), *targets, matrix_offset, permutation,
                fixed, len(gate.fixed), *places, *values,
            ]  # fmt: skip
            matrices.append(np.asarray(form.matrix, dtype=np.complex128).ravel())
            matrix_offset += matrices[-1].size
        key = ("tile", self._TILE_BITS, self.gpu_products)
        if key not in self._kernels:
            source = r"""
            __device__ unsigned long long tile_index(
                unsigned long long base, int j, const int* tile_bits) {
                for (int b = 0; b < TILE_BITS; ++b)
                    if ((j >> b) & 1) base |= 1ULL << tile_bits[b];
                return base;
            }
            // Insert bit values[i] at tile position places[i] (ascending).
            __device__ __forceinline__ int spread(
                int k, int count, const int* places, const int* values) {
                for (int i = 0; i < count; ++i) {
                    const int p = places[i];
                    k = (k & ((1 << p) - 1)) | ((k >> p) << (p + 1)) | (values[i] << p);
                }
                return k;
            }
            extern "C" __global__ void gate_tile(
                double2* state, const double2* matrices, const int* gates,
                const unsigned long long* masks, int n_gates, const int* tile_bits,
                const int* rest_bits, int n_rest) {
                extern __shared__ double2 tile[];
                const int size = 1 << TILE_BITS;
                unsigned long long base = 0;
                for (int r = 0; r < n_rest; ++r)
                    if ((blockIdx.x >> r) & 1ULL) base |= 1ULL << rest_bits[r];
                for (int j = threadIdx.x; j < size; j += blockDim.x)
                    tile[j] = state[tile_index(base, j, tile_bits)];
                __syncthreads();
                for (int g = 0; g < n_gates; ++g) {
                    // Controls outside the tile are the same for the whole
                    // block, so every thread skips (and syncs) together.
                    if ((base & masks[2 * g]) != masks[2 * g + 1]) continue;
                    const int* d = gates + GATE_INTS * g;
                    const int kind = d[0], width = d[1];
                    const int permutation = d[6], fixed = d[7], count = d[8];
                    const int* places = d + 9;
                    const int* values = d + 12;
                    const double2* matrix = matrices + d[5];
                    // Only amplitudes where the tile-local controls hold are
                    // visited: `count` positions are fixed around each k.
                    const int visits = size >> count;
                    if (kind == 3) {
                        // A diagonal anywhere: its targets outside the tile
                        // fold into the row; those inside are enumerated.
                        int fixed_row = 0, inside = 0, b0 = 0, b1 = 0, b2 = 0;
                        int r0 = 0, r1 = 0, r2 = 0;
                        for (int p = 0; p < width; ++p) {
                            const int target = d[2 + p], row_bit = width - 1 - p;
                            if (target < 0) {
                                if ((base >> (-1 - target)) & 1ULL)
                                    fixed_row |= 1 << row_bit;
                            } else {
                                if (inside == 0) { b0 = target; r0 = row_bit; }
                                else if (inside == 1) { b1 = target; r1 = row_bit; }
                                else { b2 = target; r2 = row_bit; }
                                ++inside;
                            }
                        }
                        const int dim = 1 << inside;
                        bool trivial = true;
                        for (int r = 0; r < dim; ++r) {
                            const int row = fixed_row | ((r & 1) << r0)
                                | (((r >> 1) & 1) << r1) | (((r >> 2) & 1) << r2);
                            trivial = trivial && matrix[row].x == 1.0 && matrix[row].y == 0.0;
                        }
                        if (trivial) continue;  // uniform over the block
                        for (int k = threadIdx.x; k < visits; k += blockDim.x) {
                            const int start = spread(k, count, places, values);
                            for (int r = 0; r < dim; ++r) {
                                const int row = fixed_row | ((r & 1) << r0)
                                    | (((r >> 1) & 1) << r1) | (((r >> 2) & 1) << r2);
                                const double2 m = matrix[row];
                                // A factor of exactly 1 changes nothing.
                                if (m.x == 1.0 && m.y == 0.0) continue;
                                const int i = start | ((r & 1) << b0)
                                    | (((r >> 1) & 1) << b1) | (((r >> 2) & 1) << b2);
                                tile[i] = cmul(m, tile[i]);
                            }
                        }
                    } else {
                        const int first = d[2], second = d[3], dim = 1 << width;
                        for (int k = threadIdx.x; k < visits; k += blockDim.x) {
                            const int local = spread(k, count, places, values);
                            double2 amplitudes[4];
                            int indices[4];
                            for (int r = 0; r < dim; ++r) {
                                int offset = width == 1 ? (r << first)
                                    : (((r >> 1) << first) | ((r & 1) << second));
                                indices[r] = local | offset;
                                if (kind != 1 || !(fixed & (1 << r)))
                                    amplitudes[r] = tile[indices[r]];
                            }
                            for (int r = 0; r < dim; ++r) {
                                if (kind == 0) {
                                    // A diagonal gate given tile bits: not
                                    // produced by _tile_form, but by the
                                    // every-target arm of perf/tile_check.py.
                                    tile[indices[r]] = cmul(matrix[r * dim + r], amplitudes[r]);
                                    continue;
                                }
                                if (kind == 1) {
                                    if (fixed & (1 << r)) continue;
                                    int c = (permutation >> (2 * r)) & 3;
                                    tile[indices[r]] = cmul(matrix[r * dim + c], amplitudes[c]);
                                    continue;
                                }
                                double2 sum = make_double2(0.0, 0.0);
                                for (int c = 0; c < dim; ++c)
                                    sum = cmul_add(sum, matrix[r * dim + c], amplitudes[c]);
                                tile[indices[r]] = sum;
                            }
                        }
                    }
                    __syncthreads();
                }
                for (int j = threadIdx.x; j < size; j += blockDim.x)
                    state[tile_index(base, j, tile_bits)] = tile[j];
            }
            """
            source = source.replace("TILE_BITS", str(self._TILE_BITS))
            source = self._complex_ops() + source.replace(
                "GATE_INTS", str(_TILE_GATE_INTS)
            )
            # No fast-math or reduced-precision mode; ordinary binary64 ops.
            kernel = cp.RawKernel(source, "gate_tile")
            shared = (1 << self._TILE_BITS) * 16
            if shared > 48 * 1024:
                # Above 48 KiB a block must opt in to more shared memory.
                kernel.max_dynamic_shared_size_bytes = shared
            self._kernels[key] = kernel
        state = cp.ascontiguousarray(self._state)
        self._kernels[key](
            (1 << len(rest),),
            (256,),
            (
                state,
                cp.asarray(np.concatenate(matrices)),
                cp.asarray(np.array(descriptors, dtype=np.int32)),
                cp.asarray(np.array(masks, dtype=np.uint64)),
                np.int32(len(steps)),
                cp.asarray(np.array(tile, dtype=np.int32)),
                cp.asarray(np.array(rest, dtype=np.int32)),
                np.int32(len(rest)),
            ),
            shared_mem=(1 << self._TILE_BITS) * 16,
        )
        self._state = state


class CupySVEngine(  # pylint: disable=too-many-ancestors
    _GateTiles, _CupyStateRuntime, NumpySVEngine
):
    """One device, complex128 state vectors, ideal or per-shot trajectories.

    Per-shot trajectories (channels, reset, mid-circuit measurement,
    feedforward, loss) run on the device; every random draw comes from the
    shot's host seed stream, the same draws the CPU engines make, so a seed
    selects the same branches wherever the probabilities agree.
    """

    _uses_gpu_products = True

    _supported_execution_shapes = frozenset({"single_pass", "per_shot"})
    _supports_shot_workers = False
    _supports_resident_expectation = True
    # Device copies of gate matrices are kept across the shots of a run; past
    # this many the cache is dropped, which bounds it when an engine is used
    # directly, outside an execution scope (which clears it at the end).
    _MATRIX_CACHE_LIMIT = 4096

    def __init__(self, device_id: int = 0):
        super().__init__("cupy-sv", device_id=device_id)
        self._shift_matrices: dict[tuple[int, int], np.ndarray] = {}

    def execute_local(self, context, payload, policy):
        if policy.shot_strategy in ("threads", "processes"):
            raise BackendValidationError("runtime='cuda' does not use CPU shot workers")
        return super().execute_local(context, payload, policy)

    def _allocate(self, size, initial_state):
        cp = self._cp
        if len(self._matrix_cache) > self._MATRIX_CACHE_LIMIT:
            self._matrix_cache.clear()
        with cp.cuda.Device(self.device_id):
            if initial_state is not None:
                return cp.array(initial_state, dtype=cp.complex128, copy=True).reshape(
                    size
                )
            state = cp.zeros(size, dtype=cp.complex128)
            state[0] = 1.0
            return state

    def _apply_local(self, state, matrix, targets):
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            n = len(self._dims)
            k = len(targets)
            local_dims = tuple(self._dims[t] for t in targets)
            device_matrix, structure = self._qubit_matrix(matrix)
            if self._uses_qubit_kernel(targets):
                state = cp.ascontiguousarray(state)
                return self._small_qubit_apply(state, device_matrix, targets, structure)
            m = device_matrix.reshape(local_dims + local_dims)
            axes = [n - 1 - q for q in targets]
            tensor = state.reshape(self._reversed_dims)
            return _contract_local(m, tensor, axes, n, k, xp=cp).reshape(-1)

    def _device_probabilities(self):
        cp = self._cp
        probabilities = cp.abs(self.state) ** 2
        total = probabilities.sum()
        return probabilities / cp.where(total > 0, total, 1.0)

    def _project(self, measured_subsystems, idx):
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            basis = cp.arange(self.state.size, dtype=cp.int64)
            keep = cp.ones(self.state.size, dtype=cp.bool_)
            strides = _strides(self._dims)
            for q in measured_subsystems:
                keep &= (basis // strides[q]) % self._dims[q] == (
                    idx // strides[q]
                ) % self._dims[q]
            new = cp.where(keep, self.state, 0.0)
            norm = cp.linalg.norm(new)
            self._state = new / cp.where(norm > 0, norm, 1.0)

    def _weigh_kraus_jump(self, step):
        """Every quantum-jump branch, with its squared norm reduced on the device.

        Only the norms cross to the host, where the shot's seed stream draws
        the branch exactly as the CPU engines draw it (`_pick_kraus_jump`).
        """
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            source = self.state
            branches = [
                self._apply_local(source.copy(), kraus, step.target_indices)
                for kraus in step.kraus_ops
            ]
            return branches, np.array([float(cp.vdot(b, b).real) for b in branches])

    def _take_kraus_jump(self, weighed, chosen):
        branches, norms = weighed
        with self._cp.cuda.Device(self.device_id):
            return branches[chosen] / np.sqrt(norms[chosen])

    def _shift_to_zero(self, indices, idx):
        """As the CPU engines do, with each shift matrix built once.

        The device matrix cache is keyed by the matrix object, so a fresh
        array per reset would upload a new copy for every shot.
        """
        for index in indices:
            outcome = _digit(idx, index, self._dims)
            if outcome != 0:
                key = (self._dims[index], -outcome)
                if key not in self._shift_matrices:
                    matrix = shift_matrix(*key)
                    matrix.flags.writeable = False
                    self._shift_matrices[key] = matrix
                self._state = self._apply_local(
                    self.state, self._shift_matrices[key], (index,)
                )


class CupyDMEngine(_CupyStateRuntime, NumpyDMEngine):
    """Complex128 density matrices with shared exact Kraus/reset semantics."""

    _supported_execution_shapes = frozenset({"single_pass", "per_shot"})
    _supports_resident_expectation = True

    def apply(self, step):
        self._state = self._apply_sandwich(
            self.state, step.matrix, step.target_indices, copy=False
        )

    def apply_channel(self, step, rng):
        if self._uses_qubit_kernel(step.target_indices):
            self._state = self._channel_output(step)
        else:
            super().apply_channel(step, rng)

    def _apply_local_sandwich(self, rho, matrix, targets):
        # Every Kraus term receives the same input; only gates may mutate it.
        return self._apply_sandwich(rho, matrix, targets, copy=True)

    def __init__(self, device_id: int = 0):
        super().__init__("cupy-dm", device_id=device_id)

    def _device_probabilities(self):
        return NumpyDMEngine.probabilities(self)


# Like its NumPy parent, an operator engine deliberately does not sample.
class CupyUnitaryEngine(  # pylint: disable=abstract-method,too-many-ancestors
    _GateTiles, _CupyRuntime, NumpyUnitaryEngine
):
    """CUDA local-operator application to all unitary columns together."""

    _uses_gpu_products = True

    _supported_execution_shapes = frozenset({"operator"})

    def __init__(self, device_id: int = 0):
        super().__init__("cupy-unitary", device_id=device_id)

    def _tile_layout(self):
        # Row-major (size, size): a gate acts on the row bits, which sit
        # above the n column bits that every gate leaves alone.
        n = len(self._dims)
        return n, 2 * n

    def _apply_local(self, state, matrix, targets):
        if self._uses_qubit_kernel(targets):
            state = self._cp.ascontiguousarray(state)
            device_matrix, structure = self._qubit_matrix(matrix)
            bits = tuple(len(self._dims) + t for t in targets)
            return self._small_qubit_apply(state, device_matrix, bits, structure)
        return super()._apply_local(state, matrix, targets)


# Like its NumPy parent, an operator engine deliberately does not sample.
class CupySuperopEngine(  # pylint: disable=abstract-method
    _CupyRuntime, NumpySuperopEngine
):
    """CUDA channel maps with the existing public vectorization convention."""

    _supported_execution_shapes = frozenset({"operator"})

    def apply(self, step):
        self._state = self._apply_sandwich(
            self.state, step.matrix, step.target_indices, copy=False
        )

    def apply_channel(self, step, rng):
        if self._uses_qubit_kernel(step.target_indices):
            self._state = self._channel_output(step)
        else:
            super().apply_channel(step, rng)

    def _apply_local_sandwich(self, rho, matrix, targets):
        # Every Kraus term receives the same input; only gates may mutate it.
        return self._apply_sandwich(rho, matrix, targets, copy=True)

    def __init__(self, device_id: int = 0):
        super().__init__("cupy-superop", device_id=device_id)

    def _sandwich_offsets(self):
        n = len(self._dims)
        return 3 * n, 2 * n

    def export_state(self):
        with self._cp.cuda.Device(self.device_id):
            return self._cp.asnumpy(NumpySuperopEngine.export_state(self))
