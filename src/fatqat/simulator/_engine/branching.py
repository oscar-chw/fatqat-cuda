"""Exact shot branching: shots that share a state share the work on it.

A per-shot run used to evolve every shot from the start, although most shots
spend most of the circuit in a state some other shot is also in: before the
first random step all of them are, and a low-noise or measured circuit keeps
them in a few states. Here shots travel in groups, one state per group.

- A deterministic step is applied once per group.
- A random step is weighed once on the group's state; each shot then picks
  from its own seed stream, in its own order, exactly as it would alone; the
  group splits by pick, and each picked branch is built once.
- A conditioned step splits the group by the shots' classical bits.

Every shot therefore draws the same numbers and sees the same arithmetic as in
the one-shot-at-a-time loop, so counts and classical bits are bit-identical
(the engines' weigh/pick/take methods are the pieces their one-shot methods
are built from). Loss and reload (the atom lifecycle) and any group past the
memory budget continue shot by shot from their current state, also exactly.

Memory: a random step can split a group into as many parts as it has shots,
so the parts share what the step read (the state, or a channel's weighed
branches) and build their own state only when they run. The largest part goes
on in place; the others wait. Pending groups may hold `_BRANCH_MEMORY_BYTES`
of such shared data, or `_BRANCH_STATES` states where half the free memory
allows it; past that, a waiting part runs shot by shot at once. At the plan's last step no state is built at all: only
the draws and the classical bits it writes can still matter. Shots run in
chunks of `_SHOTS_PER_CHUNK`, so per-shot bookkeeping stays bounded too.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import TYPE_CHECKING

import numpy as np

from ..._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    LossStep,
    MeasurementStep,
    PutStep,
    ResetStep,
    ResolvedStep,
)

# The module object, not its names: np.py imports this module while it is
# still initializing, and its helpers are only read when a run executes.
from . import np as _engine_np

if TYPE_CHECKING:
    from .np import _NumpyMatrixEngine

_LOG = logging.getLogger(__name__)

# Pending groups may hold this many bytes of shared data, or this many
# states where half the engine's free memory allows it; a split past the
# budget runs its extra parts shot by shot instead of keeping them. (A state
# as large as the bytes floor would otherwise leave room for one.)
_BRANCH_MEMORY_BYTES = 1 << 30
_BRANCH_STATES = 8
# Shots run in chunks of this many: each chunk shares work among its shots.
_SHOTS_PER_CHUNK = 4096


class _Shot:
    """One shot's classical bits, occupancy and seed stream."""

    __slots__ = ("index", "rng", "clbits", "occupied")

    def __init__(self, index, rng, clbits, occupied):
        self.index = index
        self.rng = rng
        self.clbits = clbits
        self.occupied = occupied


def _run_branched(
    engine: _NumpyMatrixEngine,
    plan: Sequence[ResolvedStep],
    seed_sequences: Sequence[np.random.SeedSequence],
    initial_occupied: frozenset[int] | None,
    initial_state: np.ndarray | None,
) -> list[tuple[int, ...]]:
    """Run every shot of ``plan``; return their clbit snapshots in shot order."""
    occupied = (
        set(range(len(engine._dims)))
        if initial_occupied is None
        else set(initial_occupied)
    )
    snapshots: list = []
    for start in range(0, len(seed_sequences), _SHOTS_PER_CHUNK):
        chunk = seed_sequences[start : start + _SHOTS_PER_CHUNK]
        engine.initialize(engine._dims, engine._n_clbits, initial_state=initial_state)
        shots = [
            _Shot(i, np.random.default_rng(seed), [0] * engine._n_clbits, set(occupied))
            for i, seed in enumerate(chunk)
        ]
        run = _Branching(engine, plan, len(shots))
        run.budget = _budget(engine)
        run.groups.append((engine.state, None, shots, 0))
        while run.groups:
            group = run.groups.pop()
            run.release(group[0])
            run.advance(*group)
        snapshots.extend(run.snapshots)
        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(
                "shot branching, %s: %d shots ran as %d groups (%d splits); "
                "%d shots replayed alone; peak %d of %d bytes held for pending groups",
                type(engine).__name__,
                len(shots),
                run.groups_run,
                run.splits,
                run.replayed,
                run.peak_bytes,
                run.budget,
            )
    return snapshots


def _budget(engine) -> int:
    """Bytes pending groups may hold: the floor, or up to `_BRANCH_STATES`
    states if half the engine's free memory (when it can tell) allows."""
    free = engine._free_memory_bytes()
    if free is None:
        return _BRANCH_MEMORY_BYTES
    states = _BRANCH_STATES * _nbytes(engine.state)
    return max(_BRANCH_MEMORY_BYTES, min(states, free // 2))


def _nbytes(shared) -> int:
    """Bytes held by a part's shared data: an array, or a tuple/list of them."""
    if isinstance(shared, (tuple, list)):
        return sum(_nbytes(item) for item in shared)
    return int(getattr(shared, "nbytes", 0))


class _Branching:
    """The pending groups of one chunk.

    A part (and a pending group) is ``(shared, derive, shots)``, plus its next
    step when pending. Its state is ``shared`` itself when ``derive`` is None,
    else what ``derive(shared)`` makes the engine's state: the parts a random
    step splits into share what it read.
    """

    def __init__(self, engine, plan, n_shots):
        self.engine = engine
        self.plan = plan
        self.snapshots: list = [None] * n_shots
        self.groups: list = []
        self.stochastic = engine.state_semantics == "sv"
        # id(shared) -> [pending groups using it, its bytes]; and their total.
        self.held: dict = {}
        self.held_bytes = 0
        self.budget = _BRANCH_MEMORY_BYTES
        # For the debug summary only.
        self.groups_run = self.splits = self.replayed = self.peak_bytes = 0

    def _enter(self, shared, derive) -> None:
        if derive is None:
            self.engine._state = shared
        else:
            derive(shared)

    def release(self, shared) -> None:
        """Account for a pending group leaving the queue."""
        entry = self.held.get(id(shared))
        if entry is None:  # the chunk's first group is never counted
            return
        entry[0] -= 1
        if not entry[0]:
            del self.held[id(shared)]
            self.held_bytes -= entry[1]

    def advance(self, shared, derive, shots, position) -> None:
        """Run one group from ``position`` until it ends."""
        self.groups_run += 1
        self._enter(shared, derive)
        last = len(self.plan) - 1
        for index in range(position, len(self.plan)):
            step = self.plan[index]
            if isinstance(step, (LossStep, PutStep)):
                # The atom lifecycle draws per target and resets in between;
                # it is rare enough to replay shot by shot, from here.
                self._replay(self.plan[index:], shots)
                return
            parts = self._step(step, shots, index == last)
            if parts:
                self.splits += len(parts) > 1
                # The largest part goes on in place, so only smaller ones can
                # ever wait, or run shot by shot past the budget. (Shots are
                # independent: the order changes no result.)
                parts = sorted(parts, key=lambda part: -len(part[2]))
                (shared, derive, shots), *rest = parts
                for part in rest:
                    self._keep(*part, index + 1)
                self._enter(shared, derive)
        for shot in shots:
            self.snapshots[shot.index] = tuple(shot.clbits)

    def _keep(self, shared, derive, shots, position) -> None:
        """Queue a split-off group, or run it shot by shot past the budget."""
        entry = self.held.get(id(shared))
        if entry is None:
            size = _nbytes(shared)
            if self.held and self.held_bytes + size > self.budget:
                current = self.engine.state
                self._enter(shared, derive)
                self._replay(self.plan[position:], shots)
                self.engine._state = current
                return
            entry = self.held[id(shared)] = [0, size]
            self.held_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.held_bytes)
        entry[0] += 1
        self.groups.append((shared, derive, shots, position))

    def _replay(self, plan, shots) -> None:
        """Run each shot alone from the current state, as the one-shot loop does."""
        engine = self.engine
        self.replayed += len(shots)
        start = engine.state
        for number, shot in enumerate(shots):
            engine._state = start.copy() if number < len(shots) - 1 else start
            self.snapshots[shot.index] = engine._continue_shot(
                plan, shot.rng, shot.clbits, shot.occupied
            )

    def _split_inactive(self, shots, targets, condition):
        """Shots the step applies to, and a parked part for the rest, if any."""
        active, inactive = [], []
        for shot in shots:
            applies = all(
                t in shot.occupied for t in targets
            ) and _engine_np._condition_matches(condition, shot.clbits)
            (active if applies else inactive).append(shot)
        if not active or not inactive:
            return active, None
        return active, (self.engine.state.copy(), None, inactive)

    def _step(self, step, shots, last):
        """Apply ``step`` to the group: no parts if it stays whole.

        Otherwise the parts it splits into (see the class); the first goes on
        in the engine. At the ``last`` step the draws are made and the bits
        written, but no state is built.
        """
        engine = self.engine
        if isinstance(step, MeasurementStep):
            return self._measure(step, shots, last)
        if isinstance(step, ApplyMatrixStep):
            targets = step.target_indices
        elif isinstance(step, ApplyChannelStep):
            targets = step.target_indices
        elif isinstance(step, ResetStep):
            targets = step.reset_indices
        else:
            raise AssertionError(f"unexpected plan step {type(step).__name__}")
        active, parked = self._split_inactive(shots, targets, step.condition)
        if not active:
            return None
        if isinstance(step, ApplyMatrixStep) or not self.stochastic:
            if isinstance(step, ApplyMatrixStep):
                engine.apply(step)
            elif isinstance(step, ApplyChannelStep):  # density matrix: exact,
                engine.apply_channel(step, None)  # and it draws nothing
            else:
                engine.reset_subsystems(step.reset_indices, None)
            # Not even a one-part list when the group stays whole: entering
            # a part reads the state, which would flush queued gate tiles.
            if parked is None:
                return None
            return [(engine._state, None, active), parked]
        if isinstance(step, ApplyChannelStep):
            parts = self._channel(step, active, last)
        else:
            parts, _ = self._collapse(step.reset_indices, active, last, reset=True)
        # A random step's parts are returned even when there is one: its
        # state (a picked branch, or a projection) is not built yet.
        if parked is None:
            return parts
        if not parts:  # the last step: the active shots need no state
            parts = [(engine._state, None, active)]
        return [*parts, parked]

    def _channel(self, step, shots, last):
        engine = self.engine
        route = engine._channel_route(step)
        if route is None:
            weighed = engine._weigh_kraus_jump(step)
            picks = [engine._pick_kraus_jump(weighed, shot.rng) for shot in shots]
            if last:
                return None

            def jump(chosen):
                def derive(shared):
                    engine._state = engine._take_kraus_jump(shared, chosen)

                return derive

            return [
                (weighed, jump(chosen), members)
                for chosen, members in _by_key(picks, shots)
            ]
        picks = [engine._pick_sampled_unitary(route, shot.rng) for shot in shots]
        if last:
            return None
        groups = _by_key(picks, shots)
        alone = len(groups) == 1
        targets = step.target_indices

        def unitary(chosen):
            def derive(shared):
                # The kernels may update their input in place: a part starts
                # from its own copy unless no other part shares the state.
                engine._state = shared if alone else shared.copy()
                engine._take_sampled_unitary(route, targets, chosen)

            return derive

        base = engine.state
        return [(base, unitary(chosen), members) for chosen, members in groups]

    def _collapse(self, indices, shots, last, *, reset):
        """Pick each shot's outcome on ``indices``; one part per outcome.

        The projection depends on the outcome's digits on ``indices`` only, so
        shots are grouped by those digits. The parts share the current state
        and are projected when they run: every engine's projection writes a
        new array and leaves its input as it was. Returns the parts (None at
        the ``last`` step, which needs no state) and each shot's flat index.
        """
        engine = self.engine
        weighed = engine._weigh_collapse()
        picks = [engine._pick_index(weighed, shot.rng) for shot in shots]
        by_shot = dict(zip((shot.index for shot in shots), picks))
        if last:
            return None, by_shot
        dims = engine._dims
        keys = [
            tuple(_engine_np._digit(idx, i, dims) for i in indices) for idx in picks
        ]
        first: dict = {}
        for key, idx in zip(keys, picks):
            first.setdefault(key, idx)

        def project(idx):
            def derive(shared):
                engine._state = shared
                engine._project(indices, idx)
                if reset:
                    engine._shift_to_zero(indices, idx)

            return derive

        base = engine.state
        parts = [
            (base, project(first[key]), members)
            for key, members in _by_key(keys, shots)
        ]
        return parts, by_shot

    def _measure(self, step, shots, last):
        parts, picks = self._collapse(step.measured_indices, shots, last, reset=False)
        dims = self.engine._dims
        confusions = step.confusions or (None,) * len(step.measured_indices)
        maps = step.reported_digit_maps or (None,) * len(step.measured_indices)
        for shot in shots:
            flat = picks[shot.index]
            bits = [
                _engine_np._digit(flat, index, dims) for index in step.measured_indices
            ]
            # As the one-shot loop: the collapse keeps the physical outcome;
            # only its mapped, optionally confused report is written.
            for m, c, bit, reported_map, confusion in zip(
                step.measured_indices,
                step.classical_indices,
                bits,
                maps,
                confusions,
            ):
                if m not in shot.occupied:
                    shot.clbits[c] = _engine_np.ERASURE_DIGIT
                else:
                    shot.clbits[c] = _engine_np._report_digit(
                        _engine_np._map_physical_digit(bit, reported_map),
                        confusion,
                        shot.rng,
                    )
        return parts


def _by_key(keys, shots):
    """Shots grouped by key, groups in order of first appearance."""
    groups: dict = {}
    for key, shot in zip(keys, shots):
        groups.setdefault(key, []).append(shot)
    return list(groups.items())
