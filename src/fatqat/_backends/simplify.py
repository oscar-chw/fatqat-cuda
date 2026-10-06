"""Private, runtime-neutral circuit simplification of a lowered plan.

Gates are local tensors, so neighbouring gates can be replaced by their
product, and a product equal to the identity can be dropped. This pass does
that on a resolved plan, before any engine sees it, so every runtime (NumPy,
Numba, CUDA) and every method sees the same, shorter circuit.

Products are computed *exactly*, never in floating point. Every matrix entry
of the gates the pass touches lies in the ring ``Z[w] / sqrt(2)**k`` with
``w = exp(i pi / 4)``: unit gates (entries ``0``, ``+-1``, ``+-i``, one per row
and column, such as Paulis, ``S``, ``CX``, ``SWAP`` and level permutations) by
their content, and the built-in ``H``, ``T``, ``Tdg`` and ``SX`` by their
declared identity. Products in that ring are exact, so identities such as
``H X H = Z``, ``H H = I``, ``T T = S``, ``(H x H) CX (H x H) = CX`` reversed, or
three ``CX`` making a ``SWAP`` are recognised as equalities, not up to rounding.

A *scaled permutation* (one nonzero per row and column, such as ``RZ`` or
``CPhase``) is kept as its exact entries and multiplied only by permutations
with entries ``+-1``, which move and negate entries without rounding: ``CX RZ
CX`` (a ``ZZ`` rotation, as QAOA writes it), ``X RZ X`` and ``Z RZ`` become one
diagonal or permutation, applied with exactly the multiplications of the
original.

A run of gates is replaced by its product only when that costs no more
rounding and fewer passes over the state. The rounding cost of a gate is ``0``
for a unit gate, which only moves and negates amplitudes, and otherwise the
largest number of nonzero entries in a row, the number of products summed
into each amplitude. The replacement is the exact product rounded once, so
compared with the *ideal* circuit (exact gate definitions) the result is at
least as accurate: ``H X H`` becomes an exact ``Z`` instead of two rounded
``H`` applications. Rewrites among unit gates alone leave every value
unchanged on the Numba and CUDA runtimes.

Two blocks merge when they are adjacent on their subsystems, or when the later
one commutes exactly with every step in between: disjoint steps, two diagonal
steps, or two exact gates whose products in either order are equal (``CX``
gates sharing a control, ``X`` on a ``CX`` target). Floating point neither
distributes nor reassociates, so moving a gate that rounds, or moving any gate
across one that rounds (on any subsystem, for a moving gate that rounds), is
done only when the merge removes rounding. Conditioned gates are barriers on their subsystems.
Channels, loss, measurements, resets and reloads are barriers on every
subsystem, because their renormalization sums over the whole state in memory
order. A global phase is never discarded.

When the run starts from the all-zero basis state, a second rule from circuit
synthesis applies (specialisation under known inputs): subsystems still in a
known basis state are tracked, a gate that acts as the identity on that known
input is dropped (a ``CX`` whose control is still ``|0>``), and a gate whose
known inputs select a smaller block is replaced by that block (a ``CX`` whose
control is known ``|1>`` becomes an ``X``). The affected amplitudes are exact
zeros, so these rewrites leave every value unchanged as well.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from fractions import Fraction
import heapq
from math import isqrt, prod

import numpy as np

from .steps import ApplyMatrixStep, BuiltinKernelKey, LossStep, PutStep

# --- exact arithmetic in Z[w] / sqrt(2)**k, w = exp(i pi / 4) -------------------
#
# A matrix is ``(num, k)``: an integer array of shape (rows, cols, 4) holding
# each entry's coefficients of 1, w, w**2 = i, w**3, and one shared exponent k.
# ``_normal`` keeps k minimal, which makes the representation unique, so
# equality of matrices is equality of arrays.

# Numerators stay far below int64 overflow; longer non-reducing products
# (which no gate set used here produces in practice) are simply not merged.
_MAX_NUMERATOR = 1 << 40
_MAX_EXACT_SUBSYSTEMS = 4
_SQRT2_BITS = 256
_SQRT2_SCALED = isqrt(2 << (2 * _SQRT2_BITS))  # floor(sqrt(2) * 2**256)
_EPS = float(np.finfo(np.float64).eps)


# Multiplication by b in Z[w] is linear in the coefficients of x:
# coeff(x * b)[j] = sum_i x[i] * sign(i, j) * b[(j - i) % 4], negative where
# w**i * w**l wraps past w**4 = -1.
_SHIFT = np.array([[(j - i) % 4 for j in range(4)] for i in range(4)])
_SIGN = np.array(
    [[-1 if i + (j - i) % 4 >= 4 else 1 for j in range(4)] for i in range(4)]
)


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact product, as one integer matrix product of coefficient blocks."""
    rows, inner, columns = a.shape[0], a.shape[1], b.shape[1]
    blocks = b[..., _SHIFT] * _SIGN  # (inner, columns, i, j)
    right = blocks.transpose(0, 2, 1, 3).reshape(4 * inner, 4 * columns)
    return (a.reshape(rows, 4 * inner) @ right).reshape(rows, columns, 4)


_BASIS = np.array([1, (1 + 1j) / np.sqrt(2), 1j, (-1 + 1j) / np.sqrt(2)])


def _approximate(num: np.ndarray, k: int) -> np.ndarray:
    """A close (not correctly rounded) complex value, for quick rejections."""
    return (num @ _BASIS) / np.sqrt(2) ** k


def _normal(num: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """Divide by sqrt(2) while every entry stays in Z[w]."""
    while k > 0:
        # x * sqrt(2) = x * (w - w**3)
        a0, a1, a2, a3 = (num[..., j] for j in range(4))
        doubled = np.stack((a1 - a3, a0 + a2, a1 + a3, a2 - a0), axis=-1)
        if np.any(doubled % 2):
            break
        num, k = doubled // 2, k - 1
    return num, k


def _to_float(num: np.ndarray, k: int) -> np.ndarray:
    """The correctly rounded complex128 matrix of an exact one."""

    def part(rational: int, irrational: int) -> float:
        # rational / sqrt(2)**k + irrational / sqrt(2)**(k + 1)
        value = Fraction(0)
        for coefficient, exponent in ((rational, k), (irrational, k + 1)):
            if coefficient == 0:
                continue
            if exponent % 2 == 0:
                value += Fraction(coefficient, 1 << (exponent // 2))
            else:
                # sqrt(2) / 2**((e + 1) / 2), with sqrt(2) to 256 bits: far
                # beyond binary64, and never a rounding tie (it is irrational).
                value += Fraction(
                    coefficient * _SQRT2_SCALED,
                    1 << (_SQRT2_BITS + (exponent + 1) // 2),
                )
        return float(value)

    out = np.empty(num.shape[:2], dtype=np.complex128)
    for (r, c), entry in np.ndenumerate(out):
        del entry
        a0, a1, a2, a3 = (int(v) for v in num[r, c])
        out[r, c] = complex(part(a0, a1 - a3), part(a2, a1 + a3))
    return out


def _from_unit(matrix: np.ndarray) -> np.ndarray:
    num = np.zeros(matrix.shape + (4,), dtype=np.int64)
    num[..., 0] = matrix.real
    num[..., 2] = matrix.imag
    return num


def _table() -> dict[BuiltinKernelKey, tuple[np.ndarray, int]]:
    """Exact forms of the built-in gates that round."""
    h = np.zeros((2, 2, 4), dtype=np.int64)
    h[..., 0] = [[1, 1], [1, -1]]
    t = np.zeros((2, 2, 4), dtype=np.int64)
    t[0, 0, 0] = 1
    t[1, 1, 1] = 1  # w
    tdg = np.zeros((2, 2, 4), dtype=np.int64)
    tdg[0, 0, 0] = 1
    tdg[1, 1, 3] = -1  # w**7 = -w**3
    sx = np.zeros((2, 2, 4), dtype=np.int64)
    sx[..., 0] = 1
    sx[..., 2] = [[1, -1], [-1, 1]]  # (1 +- i) / 2
    return {
        BuiltinKernelKey.H: (h, 1),
        BuiltinKernelKey.T: (t, 0),
        BuiltinKernelKey.TDG: (tdg, 0),
        BuiltinKernelKey.SX: (sx, 2),
    }


_TABLE = _table()


def _unit_kind(matrix: np.ndarray) -> str | None:
    """``"diagonal"``, ``"monomial"`` or ``None`` for a rounding-free matrix."""
    first = complex(matrix.flat[0])
    if abs(first.real) not in (0.0, 1.0) or abs(first.imag) not in (0.0, 1.0):
        return None  # most rounding gates (rotations, H) stop here
    parts = np.abs(np.ascontiguousarray(matrix).view(np.float64))
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


def _rounding(num: np.ndarray, k: int) -> int:
    """Products summed into each output amplitude; 0 when nothing rounds."""
    nonzero = np.any(num != 0, axis=-1)
    if (
        k == 0
        and not np.any(num[..., 1])
        and not np.any(num[..., 3])
        and np.all(np.abs(num[..., 0]) + np.abs(num[..., 2]) <= 1)
        and np.all(nonzero.sum(axis=0) == 1)
        and np.all(nonzero.sum(axis=1) == 1)
    ):
        return 0
    return int(nonzero.sum(axis=1).max())


def _reorder(matrix: np.ndarray, source, target, system_dims) -> np.ndarray:
    """Express a local matrix on ``source`` targets in ``target``'s order.

    The first target is the most significant local digit, so reordering the
    targets permutes the tensor axes; no arithmetic, hence exact. Trailing
    axes (an exact matrix's coefficients) ride along.
    """
    if tuple(source) == tuple(target):
        return matrix
    dims = [system_dims[q] for q in source]
    n = len(source)
    tail = matrix.shape[2:]
    axes = [list(source).index(q) for q in target]
    tensor = matrix.reshape(dims + dims + list(tail))
    tensor = tensor.transpose(
        axes + [n + a for a in axes] + list(range(2 * n, 2 * n + len(tail)))
    )
    size = prod(dims)
    return tensor.reshape((size, size) + tail)


def _embed(num: np.ndarray, source, target, system_dims) -> np.ndarray:
    """An exact matrix on ``source`` as one on the superset ``target``."""
    extra = [q for q in target if q not in source]
    if extra:
        size = prod(system_dims[q] for q in extra)
        eye = np.eye(size, dtype=np.int64)
        rows = num.shape[0] * size
        num = (num[:, None, :, None, :] * eye[None, :, None, :, None]).reshape(
            rows, rows, 4
        )
    return _reorder(num, tuple(source) + tuple(extra), target, system_dims)


def _is_diagonal(matrix: np.ndarray) -> bool:
    size = matrix.shape[0]
    off = matrix.reshape(size * size, *matrix.shape[2:])[1:].reshape(
        size - 1, size + 1, *matrix.shape[2:]
    )[:, :-1]
    return not np.any(off)


# --- blocks -----------------------------------------------------------------------


class _Block:
    """Exact gates multiplied into one, remembering how to emit the cheaper form.

    ``matrix`` is the id of the exact product in its merger's table. ``cost``
    is ``(rounding, passes)`` of the cheaper of that product and the
    children's own best forms. A block formed by moving a gate across other
    steps has no children: only its product is valid at its position.
    """

    __slots__ = ("targets", "matrix", "children", "leaf", "cost", "use_product")

    def __init__(self, targets, matrix, product_cost, *, leaf=None, children=None):
        self.targets = tuple(targets)
        self.matrix = matrix
        self.leaf = leaf
        self.children = children
        if children is None:
            self.cost, self.use_product = product_cost, leaf is None
        else:
            split = tuple(map(sum, zip(*(child.cost for child in children))))
            self.use_product = product_cost <= split
            self.cost = min(product_cost, split)


def simplify_plan(
    plan: Sequence, system_dims: Sequence[int], *, zero_start: bool = False
) -> tuple:
    """Return an equivalent plan with exact runs multiplied out.

    ``zero_start`` declares that the plan runs from the all-zero basis state,
    which enables specialisation under known inputs.
    """
    steps = _Merger(system_dims).run(plan)
    if zero_start and not any(isinstance(s, (LossStep, PutStep)) for s in steps):
        # Loss and reload make occupancy, not amplitudes, decide what a gate
        # does per shot, so known inputs are not tracked across them at all.
        specialised = _specialise(steps, system_dims)
        # Steps hold arrays, so compare by identity: any rewrite is a new object.
        if len(specialised) != len(steps) or any(
            a is not b for a, b in zip(specialised, steps)
        ):
            steps = _Merger(system_dims).run(specialised)
    return steps


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


class _Merger:
    """One forward sweep that merges each exact gate into an earlier block.

    Every distinct exact matrix is interned once and named by an integer, and
    products, commutation and costs are memoised on those names and on the
    targets' relative layout, so a circuit that repeats the same few gates
    (any compiled Clifford+T circuit) costs dictionary lookups, not algebra.
    """

    def __init__(self, system_dims: Sequence[int]):
        self.dims = tuple(system_dims)
        self.out: list = []
        # For each subsystem, sorted indices into ``out`` of live entries on it.
        self.history: dict[int, list[int]] = {}
        self.barrier = -1
        self.matrices: list[tuple[np.ndarray, int]] = []
        self.ids: dict[tuple, int] = {}
        # Per id: (rounding, diagonal, identity).
        self.info: list[tuple[int, bool, bool]] = []
        self.exact_by_content: dict[tuple, int | None] = {}
        self.diagonal_by_content: dict[tuple, bool] = {}
        self.product_cache: dict[tuple, int | None] = {}
        self.commute_cache: dict[tuple, bool] = {}
        self.emit_cache: dict[tuple, tuple[np.ndarray, tuple[int, ...]]] = {}
        self.embed_cache: dict[tuple, np.ndarray] = {}
        # Equal products share one read-only array, so a runtime that caches
        # device copies by array (CUDA) uploads each distinct product once.
        self.products: dict[bytes, np.ndarray] = {}
        # A product equal to a built-in gate in the plan keeps its identity.
        self.keyed: dict[tuple, ApplyMatrixStep] = {}

    # -- the matrix table

    def _intern(self, num: np.ndarray, k: int) -> int:
        key = (num.shape, k, num.tobytes())
        index = self.ids.get(key)
        if index is None:
            index = self.ids[key] = len(self.matrices)
            self.matrices.append((num, k))
            identity = k == 0 and np.array_equal(
                num, _from_unit(np.eye(len(num), dtype=np.complex128))
            )
            self.info.append((_rounding(num, k), _is_diagonal(num), identity))
        return index

    def _intern_scaled(self, matrix: np.ndarray) -> int:
        """Intern a *scaled permutation*: one nonzero per row and column, some
        of which round (``RZ``, ``CPhase``). Kept as its exact complex128
        entries; ``k`` is ``None`` to tell it from an exact-ring matrix."""
        key = ("scaled", matrix.shape, matrix.tobytes())
        index = self.ids.get(key)
        if index is None:
            index = self.ids[key] = len(self.matrices)
            self.matrices.append((matrix, None))
            self.info.append((1, _is_diagonal(matrix), False))
        return index

    def _block(self, targets, matrix, *, leaf=None, children=None) -> _Block:
        rounding, _, identity = self.info[matrix]
        product = (0, 0) if identity else (rounding, 1)
        return _Block(targets, matrix, product, leaf=leaf, children=children)

    def _layout(self, inner, outer) -> tuple:
        return (
            tuple(outer.index(q) for q in inner),
            tuple(self.dims[q] for q in outer),
        )

    def _embedded(self, matrix: int, inner, outer) -> np.ndarray:
        key = (matrix, *self._layout(inner, outer))
        if key not in self.embed_cache:
            num, k = self.matrices[matrix]
            if k is None:
                size = prod(self.dims[q] for q in outer if q not in inner)
                num = _reorder(
                    np.kron(num, np.eye(size)),
                    tuple(inner) + tuple(q for q in outer if q not in inner),
                    outer,
                    self.dims,
                )
            else:
                num = _embed(num, inner, outer, self.dims)
            self.embed_cache[key] = num
        return self.embed_cache[key]

    # -- classification

    def _exact(self, step) -> int | None:
        if not isinstance(step, ApplyMatrixStep) or step.condition is not None:
            return None
        matrix = np.asarray(step.matrix, dtype=np.complex128)
        content = (step.kernel_key, matrix.shape, matrix.tobytes())
        if content not in self.exact_by_content:
            exact = None
            if _unit_kind(matrix):
                exact = self._intern(_from_unit(matrix), 0)
            elif step.kernel_key in _TABLE:
                # The declared identity decides; the content check only
                # guards against a mislabelled matrix. Built-in H and T store
                # 1/sqrt(2) as 0.7071067811865475, one ulp below the
                # correctly rounded value, so equality is not required.
                num, k = _TABLE[step.kernel_key]
                if num.shape[:2] == matrix.shape and np.allclose(
                    _to_float(num, k), matrix, rtol=0, atol=4 * _EPS
                ):
                    exact = self._intern(num, k)
            elif _is_scaled_permutation(matrix):
                exact = self._intern_scaled(matrix.copy())
            self.exact_by_content[content] = exact
        return self.exact_by_content[content]

    def _diagonal(self, entry) -> bool:
        if isinstance(entry, _Block):
            return self.info[entry.matrix][1]
        if not isinstance(entry, ApplyMatrixStep) or entry.condition is not None:
            return False
        key = (entry.matrix.shape, entry.matrix.tobytes())
        if key not in self.diagonal_by_content:
            self.diagonal_by_content[key] = _is_diagonal(entry.matrix)
        return self.diagonal_by_content[key]

    def _commutes(self, entry, block: _Block) -> bool:
        if not set(_targets(entry)) & set(block.targets):
            return True
        if self._diagonal(entry) and self._diagonal(block):
            return True
        if not isinstance(entry, _Block):
            return False
        union = tuple(dict.fromkeys(entry.targets + block.targets))
        if len(union) > _MAX_EXACT_SUBSYSTEMS:
            return False
        key = (
            entry.matrix,
            block.matrix,
            *self._layout(entry.targets, union),
            self._layout(block.targets, union)[0],
        )
        if key not in self.commute_cache:
            first = self._embedded(entry.matrix, entry.targets, union)
            second = self._embedded(block.matrix, block.targets, union)
            first_k = self.matrices[entry.matrix][1]
            second_k = self.matrices[block.matrix][1]
            if first_k is None or second_k is None:
                one = _scaled_product(first, first_k, second, second_k)
                other = _scaled_product(second, second_k, first, first_k)
                commutes = (
                    one is not None and other is not None and np.array_equal(one, other)
                )
            else:
                near_first = _approximate(first, first_k)
                near_second = _approximate(second, second_k)
                # Most pairs do not commute, and floating point shows it at
                # once; only a pair that might is decided exactly.
                commutes = np.abs(
                    near_first @ near_second - near_second @ near_first
                ).max() < 1e-9 and np.array_equal(
                    _matmul(first, second), _matmul(second, first)
                )
            self.commute_cache[key] = commutes
        return self.commute_cache[key]

    def _product(self, earlier: _Block, later: _Block, targets) -> int | None:
        key = (
            earlier.matrix,
            later.matrix,
            *self._layout(earlier.targets, targets),
            self._layout(later.targets, targets)[0],
        )
        if key not in self.product_cache:
            before = self._embedded(earlier.matrix, earlier.targets, targets)
            after = self._embedded(later.matrix, later.targets, targets)
            before_k = self.matrices[earlier.matrix][1]
            after_k = self.matrices[later.matrix][1]
            if before_k is None or after_k is None:
                scaled = _scaled_product(after, after_k, before, before_k)
                product = None if scaled is None else self._intern_scaled(scaled)
            else:
                num, k = _normal(_matmul(after, before), before_k + after_k)
                product = (
                    None
                    if np.abs(num).max(initial=0) > _MAX_NUMERATOR
                    else self._intern(num, k)
                )
            self.product_cache[key] = product
        return self.product_cache[key]

    # -- the sweep

    def run(self, plan: Sequence) -> tuple:
        for step in plan:
            if isinstance(step, ApplyMatrixStep) and step.kernel_key is not None:
                matrix = np.asarray(step.matrix, dtype=np.complex128)
                self.keyed.setdefault((step.target_indices, matrix.tobytes()), step)
            touched = _touched(step)
            if touched is None:
                self.out.append(step)
                self.barrier = len(self.out) - 1
                continue
            exact = self._exact(step)
            if exact is None:
                self._append(step)
                continue
            block, position = self._block(touched, exact, leaf=step), None
            while True:
                merged = self._merge_back(block, position)
                if merged is None:
                    break
                block, position = merged
            if position is None:
                self._append(block)
        return tuple(self._emit())

    def _append(self, entry) -> None:
        self.out.append(entry)
        for q in _targets(entry):
            self.history.setdefault(q, []).append(len(self.out) - 1)

    def _merge_back(self, block: _Block, position: int | None):
        """Merge ``block`` into the earlier block it can reach, if any."""
        limit = len(self.out) if position is None else position
        crossed_any = crossed_rounding = False
        lists = [reversed(self.history.get(q, ())) for q in block.targets]
        previous = None
        for index in heapq.merge(*lists, reverse=True):
            if index == previous or index >= limit:
                continue
            previous = index
            if index <= self.barrier:
                return None
            entry = self.out[index]
            if isinstance(entry, _Block):
                own, other = set(block.targets), set(entry.targets)
                if own <= other or other <= own:
                    if block.cost[0] > 0 and self._rounds_between(index, limit):
                        # Floating point does not distribute: a rounding gate
                        # moved past one on other subsystems rounds anew.
                        crossed_any = crossed_rounding = True
                    merged = self._combine(
                        entry, block, crossed_any=crossed_any, crossed_rounding=crossed_rounding
                    )  # fmt: skip
                    if merged is not None:
                        self._move(block, position, merged, index)
                        return merged, index
                    # No exact product here (T into RZ); it may still be
                    # crossed to reach a block further back.
            if not self._commutes(entry, block):
                return None
            crossed_any = True
            # Moving a rounding gate, or moving across one, reorders rounding.
            crossed_rounding = (
                crossed_rounding or block.cost[0] > 0 or self._entry_rounds(entry)
            )
        return None

    @staticmethod
    def _entry_rounds(entry) -> bool:
        """A non-exact step always rounds; a block unless its best form is exact."""
        return entry.cost[0] > 0 if isinstance(entry, _Block) else True

    def _rounds_between(self, start: int, stop: int) -> bool:
        """Whether any live entry strictly between ``start`` and ``stop`` rounds."""
        return any(
            entry is not None and self._entry_rounds(entry)
            for entry in self.out[start + 1 : stop]
        )

    def _combine(self, earlier, later, *, crossed_any, crossed_rounding):
        if set(later.targets) <= set(earlier.targets):
            targets = earlier.targets
        else:
            targets = later.targets
        product = self._product(earlier, later, targets)
        if product is None:
            return None
        if not crossed_any:
            return self._block(targets, product, children=(earlier, later))
        # Pinned: the children are not valid at this position, only the product.
        merged = self._block(targets, product)
        before = tuple(map(sum, zip(earlier.cost, later.cost)))
        if crossed_rounding:
            # A rounding step in between rounds in another order now; that is
            # only worth it when the merge removes rounding of its own.
            accept = merged.cost[0] < before[0]
        else:
            accept = merged.cost <= before
        return merged if accept else None

    def _move(self, block, position, merged, index) -> None:
        if position is not None:
            self.out[position] = None
            for q in block.targets:
                self.history[q].remove(position)
        previous = self.out[index]
        self.out[index] = merged
        for q in set(merged.targets) - set(previous.targets):
            bisect.insort(self.history.setdefault(q, []), index)

    # -- output

    def _emit(self):
        for entry in self.out:
            if entry is None:
                continue
            if not isinstance(entry, _Block):
                yield entry
                continue
            stack = [entry]
            while stack:
                block = stack.pop()
                if block.leaf is not None:
                    yield block.leaf
                elif block.use_product:
                    yield from self._product_steps(block)
                else:
                    stack.extend(reversed(block.children))

    def _product_steps(self, block: _Block):
        if block.cost == (0, 0):
            return
        dims = tuple(self.dims[q] for q in block.targets)
        if (block.matrix, dims) not in self.emit_cache:
            num, k = self.matrices[block.matrix]
            num, kept = _drop_identity_factors(num, dims)
            matrix = num if k is None else _to_float(num, k)
            self.emit_cache[(block.matrix, dims)] = (matrix, kept)
        matrix, kept = self.emit_cache[(block.matrix, dims)]
        targets = tuple(block.targets[p] for p in kept)
        key = matrix.tobytes()
        if (targets, key) in self.keyed:
            yield self.keyed[(targets, key)]
            return
        matrix = self.products.setdefault(key, matrix)
        matrix.flags.writeable = False
        yield ApplyMatrixStep(matrix, targets)


def _targets(entry) -> tuple[int, ...]:
    return entry.targets if isinstance(entry, _Block) else entry.target_indices


def _drop_identity_factors(num, dims) -> tuple[np.ndarray, tuple[int, ...]]:
    """Remove target positions the matrix acts on as the identity (one is kept).

    ``num`` is an exact-ring matrix (a trailing coefficient axis) or a complex
    one. Returns the smaller matrix and the kept positions.
    """
    dims, kept = list(dims), list(range(len(dims)))
    p = 0
    while p < len(kept) and len(kept) > 1:
        n, d = len(dims), dims[p]
        rest = prod(dims) // d
        tail = list(num.shape[2:])
        order = [p] + [i for i in range(n) if i != p]
        tensor = num.reshape(dims + dims + tail).transpose(
            order + [n + i for i in order] + list(range(2 * n, 2 * n + len(tail)))
        )
        blocks = tensor.reshape([d, rest, d, rest] + tail)  # [row_p, rows, col_p, cols]
        inner = blocks[0, :, 0, :]
        if all(
            np.array_equal(
                blocks[r, :, c, :], inner if r == c else np.zeros_like(inner)
            )
            for r in range(d)
            for c in range(d)
        ):
            num = np.ascontiguousarray(inner)
            del dims[p], kept[p]
        else:
            p += 1
    return num, tuple(kept)


def _is_scaled_permutation(matrix: np.ndarray) -> bool:
    nonzero = matrix != 0
    return bool(np.all(nonzero.sum(axis=0) == 1) and np.all(nonzero.sum(axis=1) == 1))


def _scaled_product(after, after_k, before, before_k) -> np.ndarray | None:
    """``after @ before`` when one is a scaled permutation and it is exact.

    Exact means the other factor is a permutation with entries ``+-1``: each
    product entry is then one entry, moved and perhaps negated. A factor of
    ``+-i`` would also be exact, but it swaps real and imaginary parts, so a
    later complex multiply would pair its products differently and, with fused
    multiply-adds, round differently; two rounding factors would round.
    """

    def complex_of(matrix, k):
        if k is None:
            return matrix
        if k != 0 or np.any(matrix[..., 1:]) or np.any(np.abs(matrix[..., 0]) > 1):
            return None  # not a +-1 permutation
        return matrix[..., 0].astype(np.complex128)

    left, right = complex_of(after, after_k), complex_of(before, before_k)
    if left is None or right is None or (after_k is None and before_k is None):
        return None
    columns_left = np.abs(left).argmax(axis=1)
    columns_right = np.abs(right).argmax(axis=1)
    rows = np.arange(len(left))
    out = np.zeros_like(left)
    out[rows, columns_right[columns_left]] = (
        left[rows, columns_left] * right[columns_left, columns_right[columns_left]]
    )
    return out


# --- specialisation under known inputs -------------------------------------------


def _specialise(plan: Sequence, system_dims: Sequence[int]) -> tuple:
    """Drop or shrink gates whose inputs are known basis states.

    Valid only from the all-zero start. A subsystem is *known* while every
    amplitude away from one level of it is exactly zero; gates on other
    subsystems keep those zeros exact, whatever they round.
    """
    known: dict[int, int] = dict.fromkeys(range(len(system_dims)), 0)
    out = []
    for step in plan:
        if not isinstance(step, ApplyMatrixStep) or step.condition is not None:
            touched = _step_subsystems(step)
            if touched is None:
                known.clear()
            for q in touched or ():
                known.pop(q, None)
            out.append(step)
            continue
        targets = step.target_indices
        fixed = [i for i, q in enumerate(targets) if q in known]
        if not fixed:
            out.append(step)
            continue
        n = len(targets)
        dims = [system_dims[q] for q in targets]
        tensor = np.asarray(step.matrix).reshape(dims + dims)
        # Columns where every known input digit has its known value.
        columns = tensor[
            (slice(None),) * n
            + tuple(known[q] if i in fixed else slice(None) for i, q in enumerate(targets))
        ]  # fmt: skip
        nonzero = np.nonzero(columns)
        outputs = {i: set(nonzero[i].tolist()) for i in fixed}
        if not nonzero[0].size or any(len(v) != 1 for v in outputs.values()):
            out.append(step)
            for i in fixed:
                known.pop(targets[i])
            continue
        image = {i: outputs[i].pop() for i in fixed}
        if any(image[i] != known[targets[i]] for i in fixed):
            # The gate moves a known input to another known level: it must
            # still run, and the new level is known afterwards.
            out.append(step)
            for i in fixed:
                known[targets[i]] = image[i]
            continue
        free = [q for i, q in enumerate(targets) if i not in fixed]
        block = columns[
            tuple(image[i] if i in fixed else slice(None) for i in range(n))
        ]
        size = prod(system_dims[q] for q in free)
        block = block.reshape(size, size)
        if np.array_equal(block, np.eye(size)):
            continue  # identity on this input
        if free:
            out.append(
                ApplyMatrixStep(np.array(block, dtype=np.complex128), tuple(free))
            )
        else:
            out.append(step)  # a phase on a known state: never discarded
    return tuple(out)


def _step_subsystems(step) -> tuple[int, ...] | None:
    for name in ("target_indices", "measured_indices", "reset_indices"):
        value = getattr(step, name, None)
        if value is not None:
            return tuple(value)
    return None
