from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..._backends.engine_contract import RawResult
from ..._backends.steps import ApplyMatrixStep, ResolvedStep
from .._execution_contract import (
    _EngineCapabilities,
    _ExecutionContext as ExecutionContext,
    _ExecutionPolicy as ExecutionPolicy,
)


def _shot_seed_sequences(
    seed: int | None, n_iters: int
) -> list[np.random.SeedSequence]:
    """Spawn stable, ordered child streams for sampled shots."""
    return np.random.SeedSequence(seed).spawn(n_iters)


@dataclass(frozen=True)
class _TileForm:
    """How a qubit gate occupies a tile: only its *active* targets need tile bits.

    A *control* (a target the gate never flips, with the identity on its other
    value) needs no tile bit: whether it holds is read from the amplitude's
    index, and only amplitudes where every control holds are visited. A
    diagonal gate needs none either: it multiplies each amplitude by an entry
    chosen by the amplitude's own index bits, wherever they lie. Positions
    count from the first target, the most significant local digit.

    ``matrix`` is the gate where every control holds: for a diagonal gate its
    diagonal over the ``active`` targets (a single phase for ``CZ``, ``T`` or
    ``CPhase``, whose targets are all controls), otherwise the block acting on
    the ``active`` targets, at most two.
    """

    diagonal: bool
    controls: tuple[tuple[int, int], ...]
    active: tuple[int, ...]
    matrix: np.ndarray


def _tile_form(matrix: np.ndarray) -> _TileForm | None:
    """Classify a qubit gate exactly (no tolerance); ``None`` if it cannot tile.

    Applying only the residual block, where the controls hold, does the same
    arithmetic as the full gate: elsewhere the full gate multiplies by exact
    ones and adds exact zeros, so the values are equal.
    """
    size = matrix.shape[0]
    width = size.bit_length() - 1
    diagonal = not np.any(matrix[~np.eye(size, dtype=bool)])
    index = np.arange(size)
    bits = [(index >> (width - 1 - p)) & 1 for p in range(width)]
    controls = []
    for p in range(width):
        if np.any(matrix[bits[p][:, None] != bits[p][None, :]]):
            continue  # the gate flips this target
        for value in (1, 0):
            other = np.flatnonzero(bits[p] != value)
            if np.array_equal(matrix[np.ix_(other, other)], np.eye(len(other))):
                controls.append((p, value))
                break
    # The controls hold together: each is a target the gate never flips, with
    # the identity on its other value, so outside the block where all of them
    # hold the gate is the identity and nothing crosses into the block.
    held = np.ones(size, dtype=bool)
    for p, value in controls:
        held &= bits[p] == value
    inside = np.flatnonzero(held)
    active = tuple(p for p in range(width) if p not in dict(controls))
    residual = np.ascontiguousarray(matrix[np.ix_(inside, inside)])
    if diagonal:
        return _TileForm(True, tuple(controls), active, np.diag(residual).copy())
    if len(active) > 2:
        return None
    return _TileForm(False, tuple(controls), active, residual)


@dataclass(frozen=True)
class _TileGate:
    """Where one queued gate sits in a tile, for the tile kernels.

    ``fixed`` holds the tile positions the kernel enumerates around, in
    ascending order, with the value each is fixed to: a control's value, or 0
    for an active (or, for a diagonal, an inside) target whose values the
    kernel then runs over. ``rest_mask``/``rest_value`` are the controls
    outside the tile, tested once per tile against its base index.
    ``targets`` are the active targets: tile positions, and for a diagonal
    ``-1 - bit`` for a target outside the tile (constant over it).
    """

    fixed: tuple[tuple[int, int], ...]
    rest_mask: int
    rest_value: int
    targets: tuple[int, ...]


class _TileQueue:
    """Queue runs of qubit gates for engines that apply them tile by tile.

    Consecutive gates whose *active* targets (see `_TileForm`) fit in one tile
    of ``_TILE_BITS`` index bits (always including the ``_COALESCED_BITS``
    lowest, so each gather reads contiguous memory) are queued and applied
    together by ``_apply_tile_batch``. Diagonal gates and controls take no
    tile bit, so a run of controlled phases, ``CX`` ladders or ``T`` gates
    shares one pass however widely it spreads. ``_state`` is a property:
    reading it, by any code path, applies the queue first, and assigning it
    replaces the state and drops the queue, whose gates were meant for the
    state being replaced.

    Engines provide ``_can_tile(targets)`` and ``_apply_tile_batch(steps)``,
    and may override ``_tile_layout`` and ``_apply_now``.
    """

    _TILE_BITS: int
    _COALESCED_BITS: int
    # Per step id: (step, form), the step pinned so a recycled id never aliases.
    _tile_forms: dict[int, tuple] | None = None

    @property
    def _state(self):
        if self._pending:
            self._flush_pending()
        return self._raw_state

    @_state.setter
    def _state(self, value):
        self._pending: list[ApplyMatrixStep] = []
        self._pending_bits: frozenset[int] = frozenset()
        self._raw_state = value

    def _tile_layout(self) -> tuple[int, int]:
        """(bit of target 0, total index bits) in the flat state."""
        return 0, len(self._dims)

    def _apply_now(self, step: ApplyMatrixStep) -> None:
        """Apply one step immediately, through the engine's per-gate kernels."""
        super().apply(step)

    def _tile_form_of(self, step: ApplyMatrixStep) -> _TileForm | None:
        """`_tile_form` of a step, once per step (pinned, so ids never alias)."""
        if self._tile_forms is None:
            self._tile_forms = {}
        cached = self._tile_forms.get(id(step))
        if cached is None or cached[0] is not step:
            matrix = np.asarray(step.matrix, dtype=np.complex128)
            cached = self._tile_forms[id(step)] = (step, _tile_form(matrix))
        return cached[1]

    def _active_bits(self, step: ApplyMatrixStep) -> frozenset[int]:
        offset, _total = self._tile_layout()
        form = self._tile_form_of(step)
        if form.diagonal:
            return frozenset()  # read from the index, wherever its bits lie
        return frozenset(offset + step.target_indices[p] for p in form.active)

    def apply(self, step: ApplyMatrixStep) -> None:
        if not self._can_tile(step.target_indices) or self._tile_form_of(step) is None:
            self._flush_pending()
            self._apply_now(step)
            return
        step_bits = self._active_bits(step)
        bits = self._pending_bits | step_bits
        if len(bits | frozenset(range(self._COALESCED_BITS))) > self._TILE_BITS:
            self._flush_pending()
            bits = step_bits
        self._pending.append(step)
        self._pending_bits = bits

    def _flush_pending(self) -> None:
        pending = self._pending
        if not pending:
            return
        self._pending, self._pending_bits = [], frozenset()
        if len(pending) == 1:
            self._apply_now(pending[0])
        else:
            self._apply_tile_batch(pending)

    def _tile_bits(self, steps) -> tuple[list[int], list[int]]:
        """Sorted bits of one batch's tile, and every other index bit."""
        _offset, total = self._tile_layout()
        tile = set(range(self._COALESCED_BITS))
        for step in steps:
            tile.update(self._active_bits(step))
        for bit in range(total):
            if len(tile) == self._TILE_BITS:
                break
            tile.add(bit)
        tile = sorted(tile)
        return tile, [bit for bit in range(total) if bit not in tile]

    def _tile_gate(self, step, position) -> _TileGate:
        """A queued step's `_TileGate` in a tile whose bits map by ``position``."""
        offset, _total = self._tile_layout()
        form = self._tile_form_of(step)
        fixed, rest_mask, rest_value, targets = [], 0, 0, []
        for p, value in form.controls:
            bit = offset + step.target_indices[p]
            if bit in position:
                fixed.append((position[bit], value))
            else:
                rest_mask |= 1 << bit
                rest_value |= value << bit
        for p in form.active:
            bit = offset + step.target_indices[p]
            if bit in position:
                fixed.append((position[bit], 0))
                targets.append(position[bit])
            else:
                targets.append(-1 - bit)  # only diagonal targets can be outside
        return _TileGate(tuple(sorted(fixed)), rest_mask, rest_value, tuple(targets))


class MatrixEngine(ABC):
    """
    Abstract base class and interface contract for all engines.
    """

    _supports_kernel_threads = False
    _thread_capacity = 1
    _supports_fusion = False
    _supported_execution_shapes = frozenset({"operator", "single_pass", "per_shot"})
    _supports_shot_workers = True
    _supports_resident_expectation = False

    def __init__(
        self,
        name: str,
        *,
        state_semantics: Literal["sv", "dm"],
    ):
        self.name = name
        self.state_semantics = state_semantics

        self._state: np.ndarray | None = None
        self._dims: tuple[int, ...] = ()
        self._reversed_dims: tuple[int, ...] = ()
        self._n_clbits = 0

    @property
    def _xp(self):
        """Array namespace for local numerical primitives; orchestration stays on CPU."""
        return np

    def _matrix_array(self, matrix):
        return self._xp.asarray(matrix, dtype=complex)

    @property
    def state(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("MatrixEngine state has not been initialized.")
        return self._state

    @state.setter
    def state(self, value: np.ndarray) -> None:
        self._state = value

    @property
    def n_subsystems(self) -> int:
        return len(self._dims)

    @property
    def capabilities(self) -> _EngineCapabilities:
        """Return static numerical support without initializing the engine."""
        return _EngineCapabilities(
            supports_kernel_threads=self._supports_kernel_threads,
            thread_capacity=self._thread_capacity,
            supports_fusion=self._supports_fusion,
            supported_execution_shapes=self._supported_execution_shapes,
            supports_shot_workers=self._supports_shot_workers,
            supports_resident_expectation=self._supports_resident_expectation,
        )

    def compiled_multi_shot_compatible(self, plan: Sequence[ResolvedStep]) -> bool:
        """Whether this engine can own the complete per-shot outer loop."""
        return False

    def configure_system(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        """Configure dimensions without allocating an evolving state."""
        self._set_dims(system_dims)
        self._n_clbits = int(n_clbits)
        self._state = None

    @abstractmethod
    def initialize(
        self,
        system_dims: Sequence[int],
        n_clbits: int = 0,
        *,
        initial_state: np.ndarray | None = None,
    ) -> None:
        """Configure the system and allocate a fresh owned evolving state."""

    def _set_dims(self, system_dims: Sequence[int]) -> None:
        """Set ``_dims`` and its cached reverse together, so they never drift apart."""
        self._dims = tuple(int(d) for d in system_dims)
        self._reversed_dims = tuple(reversed(self._dims))

    @abstractmethod
    def materialize_execution(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        system_dims: tuple[int, ...],
        n_clbits: int,
        deferred_measurements: tuple[tuple[int, int], ...],
        policy: ExecutionPolicy,
    ) -> Any:
        """Build the engine-owned immutable payload for this run."""

    @abstractmethod
    def execute_local(
        self,
        context: ExecutionContext,
        payload: Any,
        policy: ExecutionPolicy,
    ) -> RawResult:
        """Execute a materialized payload locally without dispatching."""

    def execute_shot_batch(
        self,
        context: ExecutionContext,
        payload: Any,
        seed_batch: list[np.random.SeedSequence],
        policy: ExecutionPolicy,
    ) -> list[tuple[int, ...]]:
        """Execute one ordered shot batch on engines that support it."""
        raise NotImplementedError

    @abstractmethod
    def measure_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> tuple[int, ...]: ...

    def measure_subsystem(self, index: int, rng: np.random.Generator) -> int:
        """
        Measure a single subsystem and return the result.
        """
        return self.measure_subsystems([index], rng)[0]

    @abstractmethod
    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> None: ...

    def reset_subsystem(self, index: int, rng: np.random.Generator) -> None:
        """
        Reset a single subsystem to the |0> state.
        """
        self.reset_subsystems([index], rng)

    @abstractmethod
    def probabilities(self) -> np.ndarray:
        """Return the computational-basis probability distribution of the state."""

    @abstractmethod
    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        """Sample one outcome, project the internal state, return the flat index."""

    @abstractmethod
    def apply(self, step: ApplyMatrixStep) -> None:
        """Apply a single matrix step to the internal state in place."""

    def export_state(self) -> np.ndarray:
        """
        Export the current state of the engine as a numpy array.
        """
        return self.state.copy()

    def expectation_values(self, state, observables):
        """Evaluate exact term lists without constructing full operators.

        The CPU path uses the existing reference/compiled contractions. Device
        engines can keep their intermediate state resident and return only
        scalar values through this same private execution boundary.
        """
        from ..._expectation import (
            expectation_density_matrix,
            expectation_statevector,
        )

        kernel = (
            expectation_statevector
            if self.state_semantics == "sv"
            else expectation_density_matrix
        )
        return tuple(kernel(state, terms) for terms in observables)

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample flat basis-state indices from the current state.
        """
        return rng.choice(self.state.shape[0], size=shots, p=self.probabilities())
