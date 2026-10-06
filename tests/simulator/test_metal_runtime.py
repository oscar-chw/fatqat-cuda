"""runtime="metal": Numba with Metal gate tiles beside it, bit for bit.

The GPU's software binary64 follows Numba's cache tiles operation by
operation, so states are compared by bit pattern (which also tells -0.0 from
+0.0) against Numba's tiles, at every GPU share. Tests that need a Metal
device skip without one; the error paths run everywhere.
"""

import gc
from pathlib import Path
import sys
import weakref

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.steps import ApplyMatrixStep
from fatqat.errors import BackendValidationError
from fatqat.noise import Depolarizing, NoiseModel
from fatqat.simulator import Simulator

pytest.importorskip("numba")

# pylint: disable=wrong-import-position  # imports require the guard above
from fatqat.simulator._engine import metal as engine_metal  # noqa: E402
from fatqat.simulator._engine.nb import NumbaSVEngine  # noqa: E402

# The tile tests' random circuit generators (importlib mode: no sibling import).
sys.path.insert(0, str(Path(__file__).parent))
# pylint: disable-next=import-error,wrong-import-order
from test_numba_tiles import _insular_steps, _random_steps  # noqa: E402


def _metal_available() -> bool:
    try:
        engine_metal._MetalContext.get()
    except BackendValidationError:
        return False
    return True


needs_metal = pytest.mark.skipif(
    not _metal_available(), reason="needs PyObjC's Metal bindings and an Apple GPU"
)

_N = 15  # above _TILE_MIN_BYTES once the test lowers it


class _NumbaTiles(NumbaSVEngine):
    _TILE_BITS = 11
    _COALESCED_BITS = 5
    _TILE_MIN_BYTES = 0


class _Metal(engine_metal.MetalSVEngine):
    _TILE_MIN_BYTES = 0


def _fixed(share):
    """A Metal engine whose GPU share never moves."""

    class Fixed(_Metal):  # pylint: disable=too-many-ancestors
        def __init__(self):
            super().__init__()
            self.gpu_share = share

        def _learn(self, *args):
            del args

    return Fixed


def _bits(a):
    return np.asarray(a).view(np.uint64)


def _evolve(cls, steps, initial):
    engine = cls()
    engine.initialize((2,) * _N, initial_state=initial)
    for step in steps:
        engine.apply(step)
    return np.array(engine.state), engine


@needs_metal
@pytest.mark.parametrize("share", [0.0, 0.1, 0.25, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("seed", range(4))
def test_every_gpu_share_gives_numbas_tile_bits(share, seed):
    rng = np.random.default_rng(seed)
    initial = rng.normal(size=1 << _N) + 1j * rng.normal(size=1 << _N)
    initial /= np.linalg.norm(initial)
    steps = _random_steps(rng, _N, 40) + _insular_steps(rng, _N, 40)
    expected, _ = _evolve(_NumbaTiles, steps, initial)
    actual, engine = _evolve(_fixed(share), steps, initial)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))
    assert engine.gpu_share == share


@needs_metal
def test_the_adaptive_share_settles_and_keeps_the_bits(monkeypatch):
    monkeypatch.setattr(
        engine_metal.MetalSVEngine, "learned_share", engine_metal._FIRST_SHARE
    )
    rng = np.random.default_rng(11)
    steps = _random_steps(rng, _N, 200)
    expected, _ = _evolve(_NumbaTiles, steps, None)
    actual, engine = _evolve(_Metal, steps, None)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))
    # The share moved off its starting point, and a new engine starts there.
    assert 0 < engine.gpu_share <= engine_metal._MAX_SHARE
    assert engine.gpu_share != engine_metal._FIRST_SHARE
    assert _Metal().gpu_share == engine.gpu_share


@needs_metal
def test_signed_zeros_and_conjugated_gates_keep_their_bits():
    # Entries 1 - 0j on a state full of -0.0: moving an amplitude must not
    # multiply it, or the sign of a zero changes.
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128).conj()
    cx = np.eye(4, dtype=np.complex128)[[0, 1, 3, 2]].conj()
    steps = [
        ApplyMatrixStep(x, (3,)),
        ApplyMatrixStep(cx, (6, 2)),
        ApplyMatrixStep(x, (6,)),
    ]
    rng = np.random.default_rng(0)
    initial = rng.normal(size=1 << _N) + 1j * rng.normal(size=1 << _N)
    initial[rng.random(1 << _N) < 0.3] = complex(-0.0, -0.0)
    expected, _ = _evolve(_NumbaTiles, steps, initial)
    actual, _ = _evolve(_fixed(0.5), steps, initial)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


@needs_metal
def test_the_shared_buffer_is_freed_with_the_last_array_viewing_it():
    engine = _Metal()
    engine.initialize((2,) * _N)
    shared = weakref.ref(engine._raw_state.base)
    view = np.asarray(engine.state)[:4]
    engine.initialize((2,) * _N)  # a new buffer for the new run
    gc.collect()
    assert shared() is not None  # the old one lives while `view` does
    del view
    gc.collect()
    assert shared() is None


def _spy_dispatches(monkeypatch):
    """Count the batches the GPU is given."""
    dispatches = []
    shipped = engine_metal.MetalSVEngine._encode

    def encode(self, *args):
        dispatches.append(args[-1])
        return shipped(self, *args)

    monkeypatch.setattr(engine_metal.MetalSVEngine, "_encode", encode)
    return dispatches


@needs_metal
def test_public_metal_runs_equal_numba_and_use_the_gpu(monkeypatch):
    monkeypatch.setattr(engine_metal.MetalSVEngine, "_TILE_MIN_BYTES", 0)
    dispatches = _spy_dispatches(monkeypatch)
    program = fq.Program(_N)
    for q in range(_N):
        program.add(ops.H, q)
    for q in range(_N - 1):
        program.add(ops.CX, (q, q + 1))
        program.add(ops.T, q)
    request = {"counts": False, "final_state": True}
    metal = Simulator("statevector", runtime="metal").run(
        program, shots=0, result_config=request
    )
    numba = Simulator("statevector", runtime="numba").run(
        program, shots=0, result_config=request
    )
    np.testing.assert_array_equal(
        _bits(metal.result().get_statevector()), _bits(numba.result().get_statevector())
    )
    assert metal.result().metadata["runtime"] == "metal"
    assert dispatches, "the GPU took no tiles"
    # A noisy run with feedforward runs Numba's per-shot paths: equal counts.
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.02), operation=ops.H)
    dynamic = fq.Program(_N, _N)
    for q in range(_N):
        dynamic.add(ops.H, q)
    dynamic.measure(0, 0)
    dynamic.add(ops.H, 1, condition=(0, 1))
    dynamic.measure_all()
    options = {"shots": 30, "simulation_config": {"seed": 5}}
    assert (
        Simulator("statevector", runtime="metal", noise=noise)
        .run(dynamic, **options)
        .result()
        .get_counts()
        == Simulator("statevector", runtime="numba", noise=noise)
        .run(dynamic, **options)
        .result()
        .get_counts()
    )


@needs_metal
def test_a_share_that_fell_to_nothing_comes_back(monkeypatch):
    monkeypatch.setattr(engine_metal.MetalSVEngine, "learned_share", 0.0)
    dispatches = _spy_dispatches(monkeypatch)
    rng = np.random.default_rng(5)
    steps = _random_steps(rng, _N, 120)
    expected, _ = _evolve(_NumbaTiles, steps, None)
    actual, engine = _evolve(_Metal, steps, None)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))
    assert dispatches and min(dispatches) >= 1
    assert engine.gpu_share > 0


def test_metal_runs_statevectors_only():
    for method in ("density_matrix", "unitary", "superop"):
        with pytest.raises(BackendValidationError, match="statevector method only"):
            Simulator(method, runtime="metal")


def test_without_the_bindings_metal_names_its_extra(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(engine_metal, "import_module", missing)
    monkeypatch.setattr(engine_metal._MetalContext, "_shared", None)
    with pytest.raises(BackendValidationError, match=r"fatqat\[metal\]"):
        engine_metal._MetalContext.get()
