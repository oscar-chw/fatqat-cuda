"""Apple-GPU (Metal) statevector engine: Numba with Metal gate tiles beside it.

Apple GPUs have no FP64 arithmetic, so the Metal tile kernel
(``metal_tiles.metal``) does binary64 in software: round to nearest even,
subnormals, signed zeros, infinities and NaN, in the operation order of
Numba's cache tiles, so its results equal theirs bit for bit. The state is one
shared ``MTLBuffer`` that NumPy and Numba use with no copy. Each tile batch is
split: the GPU takes the first share of the tiles while Numba runs the rest of
the same buffer at the same time, and the share follows the measured speed of
each side (as StarPU's scheduler does), so neither waits long for the other.
Gates that cannot tile run on Numba.

On Apple GPUs, software binary64 does about a quarter of the CPU's binary64
work, so the GPU adds to the CPU rather than replacing it: measured 1.24-1.48x
over Numba alone at 24-26 qubits (results/metal-prototype.json).

Importing this module does not import PyObjC; the first engine does.
"""

from __future__ import annotations

from importlib import import_module, resources
import threading
import time

import numpy as np
from numba import njit, prange

from ...errors import BackendValidationError
from . import nb

# Threads per threadgroup, and the GPU's first share of a batch's tiles.
_THREADS = 256
_FIRST_SHARE = 0.25
# The share never leaves this range: the CPU always keeps some tiles, so a
# slow first batch on a busy GPU cannot hand it everything.
_MAX_SHARE = 0.9
# Weight of the newest batch's balanced share in the running share.
_SMOOTHING = 0.5


class _MetalContext:
    """The device, queue and compiled tile kernel, shared by every engine."""

    _lock = threading.Lock()
    _shared: _MetalContext | None = None

    def __init__(self):
        try:
            self.metal = import_module("Metal")
            self.objc = import_module("objc")
        except ImportError as exc:
            raise BackendValidationError(
                "runtime='metal' requires PyObjC's Metal bindings on macOS; "
                "install fatqat[metal]"
            ) from exc
        self.device = self.metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise BackendValidationError("runtime='metal' found no Metal device")
        source = (
            resources.files(__package__)
            .joinpath("metal_tiles.metal")
            .read_text(encoding="utf-8")
        )
        library, error = self.device.newLibraryWithSource_options_error_(
            source, None, None
        )
        if library is None:
            raise BackendValidationError(
                f"the Metal tile kernel did not compile: {error}"
            )
        function = library.newFunctionWithName_("gate_tile")
        self.pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if self.pipeline is None:
            raise BackendValidationError(f"the Metal tile kernel did not load: {error}")
        self.queue = self.device.newCommandQueue()

    @classmethod
    def get(cls) -> _MetalContext:
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared


class _SharedState:
    """A shared ``MTLBuffer`` as a Python buffer: arrays viewing it keep it alive."""

    def __init__(self, context: _MetalContext, nbytes: int):
        self.buffer = context.device.newBufferWithLength_options_(
            nbytes, context.metal.MTLResourceStorageModeShared
        )
        if self.buffer is None:
            raise MemoryError(f"the Metal buffer of {nbytes} bytes was not allocated")
        self.view = self.buffer.contents().as_buffer(nbytes)

    def __buffer__(self, flags):
        return memoryview(self.view)


@njit(cache=True, parallel=True)
def _cpu_tiles(
    state, tile_bits, rest_bits, first, last, codes, widths, targets, matrices,
    columns, values, fixed_places, fixed_values, fixed_counts, rest,
):  # fmt: skip  # pragma: no cover - compiled by Numba
    """`nb._apply_tiles` over tiles ``[first, last)`` only; the GPU runs the others."""
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


def _residual_kind(matrix: np.ndarray) -> tuple[int, int, int]:
    """(kind, permutation code, fixed rows) as the tile kernel reads them."""
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


class MetalSVEngine(nb.NumbaSVEngine):
    """Statevectors on Numba, with each tile batch shared with the Apple GPU."""

    # The Metal kernel's threadgroup tile (TILE_BITS in metal_tiles.metal),
    # with the lowest bits contiguous for its loads.
    _TILE_BITS = 11
    _COALESCED_BITS = 5
    # Below this a GPU dispatch costs more than the tiles it would take.
    _TILE_MIN_BYTES = 16 * 2**20
    # The share the last batch in this process settled on: a new engine (a
    # new Simulator) starts from it rather than from scratch.
    learned_share = _FIRST_SHARE

    def __init__(self, name: str = "metal-sv"):
        super().__init__(name)
        self._context: _MetalContext | None = None
        # Starts from what earlier engines in this process learnt.
        self.gpu_share = MetalSVEngine.learned_share

    @property
    def _metal(self) -> _MetalContext:
        if self._context is None:
            self._context = _MetalContext.get()
        return self._context

    def _allocate(self, size: int, initial_state: np.ndarray | None) -> np.ndarray:
        if size * 16 <= self._TILE_MIN_BYTES:
            return super()._allocate(size, initial_state)  # never tiled here
        shared = _SharedState(self._metal, size * 16)
        array = np.frombuffer(shared, dtype=np.complex128)
        if initial_state is None:
            array[:] = 0
            array[0] = 1
        else:
            array[:] = np.asarray(initial_state, dtype=np.complex128).reshape(size)
        return array

    def _apply_tile_batch(self, pending) -> None:
        state = self._raw_state
        if not isinstance(state.base, _SharedState):
            # A kernel replaced the shared buffer (or it was never one):
            # the GPU cannot see this array, so Numba runs the whole batch.
            super()._apply_tile_batch(pending)
            return
        tile, rest = self._tile_bits(pending)
        tile_bits = np.array(tile, dtype=np.int64)
        rest_bits = np.array(rest, dtype=np.int64)
        total = 1 << len(rest)
        on_gpu = int(round(total * self.gpu_share))
        # Each side keeps at least one tile, so both stay timed: a share that
        # once fell to nothing (a GPU busy elsewhere) can still come back.
        on_gpu = min(total - 1, max(1, on_gpu)) if total > 1 else min(total, on_gpu)
        # Both sides are timed from here, so the GPU's encoding and launch
        # count against its share (perf_counter and Metal's GPU timestamps
        # share the host clock on macOS).
        start = time.perf_counter()
        command = self._encode(pending, tile, rest, on_gpu) if on_gpu else None
        try:
            if on_gpu < total:
                _cpu_tiles(
                    state,
                    tile_bits,
                    rest_bits,
                    on_gpu,
                    total,
                    *self._tile_descriptors(pending, tile_bits),
                )
        finally:
            cpu_done = time.perf_counter()
            if command is not None:
                command.waitUntilCompleted()  # never leave the GPU writing
        if command is not None and command.error() is not None:
            raise RuntimeError(f"the Metal tile kernel failed: {command.error()}")
        if command is not None and on_gpu < total:
            self._learn(
                on_gpu, command.GPUEndTime() - start, total - on_gpu, cpu_done - start
            )

    def _encode(self, pending, tile, rest, count):
        """Start the GPU on tiles ``[0, count)`` of this batch, without waiting."""
        context = self._metal
        position = {q: i for i, q in enumerate(tile)}
        descriptors, masks, matrices, offset = [], [], [], 0
        for step in pending:
            form = self._tile_form_of(step)
            gate = self._tile_gate(step, position)
            masks += [gate.rest_mask, gate.rest_value]
            if form.diagonal:
                kind, permutation, fixed = 3, -1, 0
            else:
                kind, permutation, fixed = _residual_kind(form.matrix)
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
        arguments = [
            np.ascontiguousarray(np.concatenate(matrices)),
            np.array(descriptors, dtype=np.int32),
            np.array(masks, dtype=np.uint64),
            np.int32(len(pending)),
            np.array(tile, dtype=np.int32),
            np.array(rest, dtype=np.int32),
            np.int32(len(rest)),
            np.uint32(0),
        ]
        shared = self._raw_state.base
        with context.objc.autorelease_pool():
            command = context.queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(context.pipeline)
            encoder.setBuffer_offset_atIndex_(shared.buffer, 0, 0)
            for index, value in enumerate(arguments, start=1):
                data = np.ascontiguousarray(value).tobytes() or b"\0" * 4
                # setBytes copies at most 4 KiB; larger arguments get buffers.
                if len(data) <= 4096:
                    encoder.setBytes_length_atIndex_(data, len(data), index)
                else:
                    buffer = context.device.newBufferWithBytes_length_options_(
                        data, len(data), context.metal.MTLResourceStorageModeShared
                    )
                    encoder.setBuffer_offset_atIndex_(buffer, 0, index)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                context.metal.MTLSizeMake(count, 1, 1),
                context.metal.MTLSizeMake(_THREADS, 1, 1),
            )
            encoder.endEncoding()
            command.commit()
        return command

    def _learn(self, gpu_tiles, gpu_seconds, cpu_tiles, cpu_seconds) -> None:
        """Move the GPU share towards the one that would have balanced this batch.

        Both sides ran the same gates, so the ratio of their speeds in one
        batch does not depend on how costly its gates were; averaging the
        share itself, not each speed, keeps cheap and costly batches from
        skewing it. A batch run by one side alone teaches nothing.
        """
        if gpu_seconds <= 0 or cpu_seconds <= 0:
            return
        gpu, cpu = gpu_tiles / gpu_seconds, cpu_tiles / cpu_seconds
        balanced = gpu / (gpu + cpu)
        self.gpu_share = min(
            _MAX_SHARE, _SMOOTHING * balanced + (1 - _SMOOTHING) * self.gpu_share
        )
        MetalSVEngine.learned_share = self.gpu_share
