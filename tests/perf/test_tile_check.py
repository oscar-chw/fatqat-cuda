"""The tile check must fail on each regression it exists to catch."""

from perf.tile_check import MIN_SPEEDUP, verdict


def _row(workload="qft", runtime="numba", ratio=2.0, passes=(27, 5), identical=True):
    return {
        "workload": workload,
        "runtime": runtime,
        "identical": identical,
        "insular_over_every_target": ratio,
        "tile_passes": {"every_target": passes[0], "insular": passes[1]},
    }


def test_verdict_passes_a_good_run():
    assert verdict([_row(), _row("adder", ratio=1.2, passes=(7, 7))]) == []
    assert verdict([_row(runtime="cuda", passes=(47, 7))]) == []


def test_verdict_fails_on_each_regression():
    assert verdict([_row(identical=False)])
    assert verdict([_row(ratio=MIN_SPEEDUP - 0.01)])  # the QFT is gated
    assert verdict([_row("adder", ratio=0.9, passes=(7, 7))])  # slower anywhere
    assert verdict([_row("adder", ratio=1.2, passes=(7, 8))])  # more passes
    assert verdict([_row(passes=(27, 8))])  # QFT passes not cut fourfold
