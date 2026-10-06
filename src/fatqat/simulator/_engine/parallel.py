"""Loky process routes for already-resolved matrix executions.

Two routes: CPU shot workers (`_run_shots_in_processes`), and one worker per
further GPU for a run's shots (`_run_shots_on_device_workers`).
"""

from __future__ import annotations

from concurrent.futures import wait
from concurrent.futures.process import BrokenProcessPool
from itertools import repeat
import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from .._execution_contract import (
    _ExecutionContext as ExecutionContext,
    _ExecutionPolicy as ExecutionPolicy,
)
from .._execution_policy import _process_child_policy
from .base import _shot_seed_sequences

_LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import Protocol

    class _ProcessEngine(Protocol):
        def execute_shot_batch(
            self,
            context: ExecutionContext,
            payload: Any,
            seed_batch: list[np.random.SeedSequence],
            policy: ExecutionPolicy,
        ) -> list[tuple[int, ...]]: ...

    class _EngineFactory(Protocol):
        def __call__(self) -> _ProcessEngine: ...


# Installed before scientific modules import in a loky child. The explicit
# Numba mask is also applied by the child's local execution scope.
_WORKER_THREAD_VARS = (
    "BLIS_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _split_into_batches(
    seed_sequences: list[np.random.SeedSequence], n_batches: int
) -> list[list[np.random.SeedSequence]]:
    """Split ordered shots into no more batches than useful work items."""
    n_items = len(seed_sequences)
    n_batches = max(1, min(n_batches, n_items))
    base, extra = divmod(n_items, n_batches)
    batches: list[list[np.random.SeedSequence]] = []
    start = 0
    for index in range(n_batches):
        size = base + (1 if index < extra else 0)
        batches.append(seed_sequences[start : start + size])
        start += size
    return batches


def _run_shot_batch(
    engine_cls: _EngineFactory,
    context: ExecutionContext,
    payload: Any,
    seed_batch: list[np.random.SeedSequence],
    child_policy: ExecutionPolicy,
) -> list[tuple[int, ...]]:
    engine = engine_cls()
    return engine.execute_shot_batch(context, payload, seed_batch, child_policy)


def _loky_executor(max_workers: int):
    from loky import get_reusable_executor

    return get_reusable_executor(
        max_workers=max_workers,
        env={name: "1" for name in _WORKER_THREAD_VARS},
    )


def _run_shots_in_processes(
    engine_cls: _EngineFactory,
    context: ExecutionContext,
    payload: Any,
    policy: ExecutionPolicy,
) -> list[tuple[int, ...]]:
    """Run shot batches in real child processes under a serial child policy."""
    assert policy.shot_strategy == "processes"
    assert policy.worker_limit is not None, "process policy requires a worker ceiling"
    seeds = _shot_seed_sequences(context.seed, context.shots)
    batches = _split_into_batches(seeds, policy.worker_limit)
    child = _process_child_policy(policy)
    # Keep Loky's process-global reusable pool at the stable resolved ceiling;
    # the number of submitted batches still bounds useful work for this run.
    executor = _loky_executor(policy.worker_limit)
    results = executor.map(
        _run_shot_batch,
        repeat(engine_cls),
        repeat(context),
        repeat(payload),
        batches,
        repeat(child),
    )
    return [snapshot for batch in results for snapshot in batch]


# A GPU worker's engines, one per (engine class, device), kept between runs so
# its CUDA context, compiled kernels and matrix uploads are reused.
_DEVICE_ENGINES: dict = {}
# Parent side: one single-worker executor per further device, so each device
# keeps one worker and each worker one device (one CUDA context apiece).
_DEVICE_EXECUTORS: dict = {}
# Idle GPU workers exit after this many seconds, releasing their contexts.
_DEVICE_WORKER_IDLE_SECONDS = 300
_SERIAL = ExecutionPolicy(
    shot_strategy="serial",
    kernel_strategy="serial",
    worker_limit=1,
    fusion=False,
    use_compiled_multi_shot_kernel=False,
)


def _run_device_batch(engine_cls, device, context, payload, seed_batch):
    """In a GPU worker: run one ordered batch of shots on ``device``."""
    engine = _DEVICE_ENGINES.get((engine_cls, device))
    if engine is None:
        _LOG.debug(
            "GPU worker %d: building its engine for device %d", os.getpid(), device
        )
        engine = _DEVICE_ENGINES[(engine_cls, device)] = engine_cls(device_id=device)
    try:
        return engine.execute_shot_batch(context, payload, seed_batch, _SERIAL)
    finally:
        # Nothing reads the state, and an idle worker must not keep a whole
        # state's blocks reserved on its device for the next run to trip on.
        engine.state = None
        release = getattr(engine, "_release_device_memory", None)
        if release is not None:
            release()


def _device_executor(device: int):
    from loky import ProcessPoolExecutor  # pylint: disable=import-outside-toplevel

    executor = _DEVICE_EXECUTORS.get(device)
    if executor is None:
        _LOG.debug("starting the GPU worker for device %d", device)
        executor = _DEVICE_EXECUTORS[device] = ProcessPoolExecutor(
            max_workers=1,
            timeout=_DEVICE_WORKER_IDLE_SECONDS,
            env={name: "1" for name in _WORKER_THREAD_VARS},
        )
    return executor


def _run_shots_on_device_workers(
    engine_cls, devices, context, payload, batches, run_local
) -> list[tuple[int, ...]]:
    """Run ``batches[0]`` here with ``run_local``, the rest in GPU workers.

    One worker process per further device, rather than threads: each shot's
    loop is mostly Python, and threads take turns on the interpreter lock
    (two GPUs on threads measured 0.67-0.80x of one at 20 qubits). Loky
    starts workers as fresh interpreters, so a CUDA context is never forked
    and an unguarded user script is not re-run. Batches come back in order.

    Every batch finishes before this returns or raises, so none outlives the
    run. On failure the first device's error is raised if it had one, else
    the first failing device's in device order, with a note naming the
    device. A worker that died (a crash, or an out-of-memory kill) is
    replaced on the next run.
    """
    futures = []
    for device, batch in zip(devices[1:], batches[1:]):
        try:
            future = _device_executor(device).submit(
                _run_device_batch, engine_cls, device, context, payload, batch
            )
        except BrokenProcessPool:  # died since its last run: start a fresh one
            _LOG.info(
                "the GPU worker for device %d had exited; starting another", device
            )
            _DEVICE_EXECUTORS.pop(device, None)
            future = _device_executor(device).submit(
                _run_device_batch, engine_cls, device, context, payload, batch
            )
        futures.append((device, future))
    local = local_error = None
    try:
        local = run_local(batches[0])
    except Exception as error:  # pylint: disable=broad-except  # re-raised below
        local_error = error
        error.add_note(f"in the shot batch for CUDA device {devices[0]}")
    wait([future for _, future in futures])
    remote, errors = [], []
    for device, future in futures:
        try:
            remote.append(future.result())
        except BrokenProcessPool as error:
            _DEVICE_EXECUTORS.pop(device, None)
            errors.append(
                RuntimeError(
                    f"the GPU worker for CUDA device {device} exited while running "
                    "its shot batch (a crash or an out-of-memory kill); the next "
                    "run starts a new worker"
                )
            )
            errors[-1].__cause__ = error
        except Exception as error:  # pylint: disable=broad-except  # re-raised below
            error.add_note(f"in the GPU worker for CUDA device {device}")
            errors.append(error)
    if local_error is not None:
        raise local_error
    if errors:
        raise errors[0]
    return [snapshot for batch in [local, *remote] for snapshot in batch]
