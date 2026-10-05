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
from ..._backends.steps import ApplyMatrixStep, MeasurementStep
from .np import (
    NumpySVEngine,
    NumpyDMEngine,
    NumpyUnitaryEngine,
    NumpySuperopEngine,
    _strides,
)


class _CupyRuntime:
    """Device ownership shared by CUDA matrix-family engines."""

    _supports_shot_workers = False

    def __init__(self, name, *, device_id: int = 0):
        super().__init__(name)
        if type(device_id) is not int or device_id < 0:
            raise BackendValidationError("device_id must be a nonnegative int")
        self.device_id = device_id
        self._matrix_cache = {}
        self._kernels = {}

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

    def _conjugate_matrix(self, matrix):
        self._qubit_matrix(matrix)
        cached = self._matrix_cache[id(matrix)]
        if len(cached) == 3:
            cached = (*cached, cached[1].conj())
            self._matrix_cache[id(matrix)] = cached
        return cached[3]

    def _small_qubit_apply(self, state, matrix, targets, structure):
        """Update disjoint pairs/quartets without transposing the full state."""
        cp = self._cp
        width = len(targets)
        dim = 1 << width
        diagonal, permutation, fixed = structure
        key = (width, diagonal, permutation >= 0)
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
                    double2 a = state[i], m = matrix[row * DIM + row];
                    state[i] = make_double2(m.x*a.x - m.y*a.y, m.x*a.y + m.y*a.x);
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
                            double2 m = matrix[r*DIM+c], a = values[c];
                            state[indices[r]] = make_double2(m.x*a.x-m.y*a.y, m.x*a.y+m.y*a.x);
                            continue;
                        }
                        double real=0, imag=0;
                        for (int c=0; c<DIM; ++c) {
                            double2 m = matrix[r*DIM+c], a=values[c];
                            real += m.x*a.x - m.y*a.y;
                            imag += m.x*a.y + m.y*a.x;
                        }
                        state[indices[r]] = make_double2(real, imag);
                    }
                }
                """
            source = source.replace("WIDTH", str(width)).replace("DIM", str(dim))
            source = source.replace("MONOMIAL", str(int(permutation >= 0)))
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

    def execute_shot_batch(self, context, payload, seed_batch, policy):
        raise BackendValidationError("CUDA engines do not use CPU shot workers")


class _CupyStateRuntime(_CupyRuntime):
    """GPU probability sampling with the existing CPU random stream."""

    def probabilities(self):
        with self._cp.cuda.Device(self.device_id):
            return self._cp.asnumpy(self._device_probabilities())

    def sample_indices(self, shots, rng):
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            cdf = cp.cumsum(self._device_probabilities())
            total = float(cdf[-1])
            if not np.isfinite(total) or total <= 0:
                raise ValueError("probabilities must have a finite, positive sum")
            cdf /= total
            # Keep the seeded CPU random stream, transfer only uniforms and
            # sampled indices; the full probability vector stays on device.
            uniforms = cp.asarray(rng.random(shots))
            return cp.asnumpy(cp.searchsorted(cdf, uniforms, side="right"))


class CupySVEngine(_CupyStateRuntime, NumpySVEngine):
    """One device, complex128, ideal single-pass state vectors only."""

    _supported_execution_shapes = frozenset({"single_pass"})
    _supports_shot_workers = False
    _supports_resident_expectation = True

    def __init__(self, device_id: int = 0):
        super().__init__("cupy-sv", device_id=device_id)

    def materialize_execution(self, plan, **kwargs):
        if any(
            not isinstance(step, (ApplyMatrixStep, MeasurementStep))
            or getattr(step, "condition", None) is not None
            for step in plan
        ):
            raise BackendValidationError(
                "runtime='cuda' supports ideal gates and terminal measurement only"
            )
        return super().materialize_execution(plan, **kwargs)

    def execute_local(self, context, payload, policy):
        if context.execution_shape != "single_pass":
            raise BackendValidationError(
                "runtime='cuda' does not support dynamic shots"
            )
        if policy.shot_strategy in ("threads", "processes"):
            raise BackendValidationError("runtime='cuda' does not use CPU shot workers")
        return super().execute_local(context, payload, policy)

    def execute_shot_batch(self, context, payload, seed_batch, policy):
        raise BackendValidationError("runtime='cuda' does not support dynamic shots")

    def _allocate(self, size, initial_state):
        cp = self._cp
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
            out = cp.tensordot(
                m,
                state.reshape(self._reversed_dims),
                axes=(list(range(k, 2 * k)), axes),
            )
            remaining = [ax for ax in range(n) if ax not in axes]
            perm = [0] * n
            for j, ax in enumerate(axes):
                perm[ax] = j
            for j, ax in enumerate(remaining):
                perm[ax] = k + j
            return cp.transpose(out, perm).reshape(-1)

    def _device_probabilities(self):
        cp = self._cp
        probabilities = cp.abs(self.state) ** 2
        total = probabilities.sum()
        return probabilities / cp.where(total > 0, total, 1.0)

    def collapse(self, measured_subsystems, rng):
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            idx = int(self.sample_indices(1, rng)[0])
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
            return idx

    def apply_channel(self, step, rng):
        raise BackendValidationError("runtime='cuda' does not support channel noise")

    def reset_subsystems(self, indices, rng):
        raise BackendValidationError("runtime='cuda' does not support reset")


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
class CupyUnitaryEngine(  # pylint: disable=abstract-method
    _CupyRuntime, NumpyUnitaryEngine
):
    """CUDA local-operator application to all unitary columns together."""

    _supported_execution_shapes = frozenset({"operator"})

    def __init__(self, device_id: int = 0):
        super().__init__("cupy-unitary", device_id=device_id)

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
