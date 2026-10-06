"""Private, runtime-neutral circuit simplification of a lowered plan.

Gates are local tensors, so two gates on the same subsystems can be replaced by
their product, and a product equal to the identity can be dropped. This pass
does that on a resolved plan, before any engine sees it, so every runtime
(NumPy, Numba, CUDA) and every method sees the same, shorter circuit.

Simplification never changes a computed value. It only touches *unit
monomial* gates: matrices whose entries are all ``0``, ``+-1`` or ``+-i``
with exactly one nonzero per row and column (Paulis, ``S``/``Sdg``, ``CX``,
``CZ``, ``SWAP``, level permutations and phase flips). Applying such a gate
moves and negates amplitudes without rounding on every runtime, and a product
of two is again one, computed exactly. So merging ``S.S`` into ``Z``, three
``CX`` into a ``SWAP``, or removing ``X.X`` and ``CX.CX`` leaves every amplitude
exactly as the original plan computes it on the Numba and CUDA runtimes,
with fewer passes over the state. The NumPy runtime contracts through BLAS,
and some BLAS builds round an element by a fused or unfused path depending
on its position, so when a removed gate would have moved amplitudes, other
gates' last-bit rounding can move with them; the change is unbiased.

Gates that round (rotations, ``H``, any other matrix) are never merged or
moved. Measured against an extended-precision reference, both were worse:
merging rounded products (``H.H``, ``RY.RZ``) biased amplitudes, and even an
exactly representable product such as ``RY.Z``, applied as one dense gate,
adds the same terms in another order, which rounds differently wherever
multiply-adds are fused.

Two unit gates on the same subsystem set merge when no step in between
touches those subsystems, or when the later gate and every step in between
on them are unit *diagonal* gates, which commute exactly. Conditioned gates
are barriers on their subsystems. Channels, loss, measurements, resets and
reloads are barriers on every subsystem, because their renormalization sums
over the whole state in memory order. A global phase is never discarded, so
``-I`` is kept. "Same value" means equal values: a unit gate can turn a
``-0.0`` into ``+0.0``.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np

from .steps import ApplyMatrixStep


def _touched(step) -> tuple[int, ...] | None:
    """Subsystems a gate acts on; ``None`` makes the step a global barrier.

    Measurements, resets, channels, loss and reloads are global barriers,
    not barriers on their own subsystems only: their renormalization (and
    outcome sampling) sums amplitudes over the whole state in memory order,
    so moving a permutation across them on *other* subsystems would change
    that order and the last bit of the result.
    """
    if isinstance(step, ApplyMatrixStep):
        return step.target_indices
    return None


def _unit_kind(step) -> str | None:
    """``"diagonal"``, ``"monomial"`` or ``None`` for a rounding-free gate."""
    if not isinstance(step, ApplyMatrixStep) or step.condition is not None:
        return None
    matrix = step.matrix
    first = complex(matrix.flat[0])
    if abs(first.real) not in (0.0, 1.0) or abs(first.imag) not in (0.0, 1.0):
        return None  # most rounding gates (rotations, H) stop here
    parts = np.abs(np.ascontiguousarray(matrix).view(np.float64))
    # Cheap rejection first: rotations fail here on their first entry.
    if not np.all((parts == 0) | (parts == 1)):
        return None
    pairs = parts.reshape(matrix.shape + (2,)).sum(axis=-1)  # |re| + |im|
    nonzero = pairs != 0
    if not (
        np.all(pairs[nonzero] == 1)  # 1, -1, i or -i, never 1 + i
        and np.all(nonzero.sum(axis=0) == 1)
        and np.all(nonzero.sum(axis=1) == 1)
    ):
        return None
    return "diagonal" if np.all(np.diag(nonzero)) else "monomial"


def _reorder(matrix: np.ndarray, source, target, system_dims) -> np.ndarray:
    """Express a local matrix on ``source`` targets in ``target``'s order.

    The first target is the most significant local digit, so reordering the
    targets permutes the tensor axes; no arithmetic, hence exact.
    """
    if tuple(source) == tuple(target):
        return matrix
    dims = [system_dims[q] for q in source]
    k = len(source)
    axes = [list(source).index(q) for q in target]
    tensor = matrix.reshape(dims + dims).transpose(axes + [k + a for a in axes])
    size = prod(dims)
    return tensor.reshape(size, size)


def simplify_plan(plan: Sequence, system_dims: Sequence[int]) -> tuple:
    """Return an equivalent plan with runs of unit gates multiplied out."""
    out: list = []
    # For each subsystem, indices into ``out`` of live steps touching it, in order.
    history: dict[int, list[int]] = {}
    global_barrier = -1
    # The kind is checked many times per gate while searching; compute it once
    # per step object (kept alive in the cache, so ids are not reused), and
    # classify each distinct matrix content only once per plan.
    kinds: dict[int, tuple[object, str | None]] = {}
    by_content: dict[tuple, str | None] = {}
    # Equal products share one read-only array, so a runtime that caches
    # device copies by array (CUDA) uploads each distinct product once.
    products: dict[bytes, np.ndarray] = {}

    def kind(step) -> str | None:
        cached = kinds.get(id(step))
        if cached is None:
            if isinstance(step, ApplyMatrixStep) and step.condition is None:
                content = (step.matrix.shape, step.matrix.tobytes())
                if content not in by_content:
                    by_content[content] = _unit_kind(step)
                result = by_content[content]
            else:
                result = None
            cached = kinds[id(step)] = (step, result)
        return cached[1]

    def passable(index: int, diagonal: bool) -> bool:
        """Can a later unit gate move back across ``out[index]`` exactly?"""
        return diagonal and kind(out[index]) == "diagonal"

    def candidate(step) -> int | None:
        """The earlier unit gate ``step`` can be multiplied into, if any."""
        targets = step.target_indices
        diagonal = kind(step) == "diagonal"
        for index in reversed(history.get(targets[0], [])):
            if index <= global_barrier:
                return None
            previous = out[index]
            if kind(previous) and set(previous.target_indices) == set(targets):
                # The walk above cleared targets[0]; every other target's steps
                # after ``index`` must be passable as well.
                for q in targets[1:]:
                    stack = history[q]
                    after = stack[stack.index(index) + 1 :]
                    if not all(passable(i, diagonal) for i in after):
                        return None
                return index
            if not passable(index, diagonal):
                return None
        return None

    for step in plan:
        touched = _touched(step)
        if touched is None:
            out.append(step)
            global_barrier = len(out) - 1
            continue
        index = candidate(step) if kind(step) else None
        if index is None:
            out.append(step)
            for q in touched:
                history.setdefault(q, []).append(len(out) - 1)
            continue
        previous = out[index]
        later = _reorder(
            step.matrix, step.target_indices, previous.target_indices, system_dims
        )
        # Exact: every entry of the product is one unit times another unit.
        product = later @ previous.matrix
        key = product.tobytes()
        product = products.setdefault(key, product)
        product.flags.writeable = False
        if np.array_equal(product, np.eye(len(product))):
            out[index] = None
            for q in previous.target_indices:
                history[q].remove(index)
        else:
            out[index] = ApplyMatrixStep(product, previous.target_indices)
    return tuple(step for step in out if step is not None)
