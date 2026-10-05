"""Compatibility names for early experimental backends."""

from .simulator import Simulator


class CupySimulator(Simulator):
    """Compatibility spelling of ``Simulator(method="SV", runtime="cuda")``.

    Uses the built-in complex128 CUDA engine and its validation. New code can
    select CUDA directly on ``Simulator``. Requires optional CuPy and an NVIDIA
    GPU at execution time; see ``Simulator`` for supported programs and controls.
    """

    def __init__(self, *, device_id=0, implementation_map=None):
        super().__init__(
            "SV",
            runtime="cuda",
            device_id=device_id,
            implementation_map=implementation_map,
        )
