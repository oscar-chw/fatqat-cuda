"""CUDA runs of many shots: the one-shot loop against branching, and 1 vs all GPUs.

Arms, alternating in one process (warm-up round first, arm order reversed on
every other repeat, median of the repeats): one GPU with every shot evolved
alone (the loop before r11), one GPU with shot branching, and every visible
GPU with shot branching. Every arm's counts must equal the others' (same
seed). The workloads are `branching_check.py`'s. "never_slower" is true when
every row's all-GPU time is within the noise floor of one GPU's. Each row
also records the 1-minute load average after every timed call; drop it
before publishing (results/shots-cuda.json has none).

Usage, from the repository root, on a machine with CUDA GPUs:
    python perf/shots_check.py OUT.json [--no-per-shot] QUBITS...
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "perf")
# pylint: disable=wrong-import-position,import-error  # path set up above
import branching_check as bc  # noqa: E402
from fatqat.simulator import Simulator  # noqa: E402
from fatqat.simulator._engine import np as engine_np  # noqa: E402

out = Path(sys.argv[1])
per_shot = "--no-per-shot" not in sys.argv
qubits = [int(q) for q in sys.argv[2:] if q != "--no-per-shot"]
NOISE_FLOOR = 0.95
SHOTS, REPEATS = 256, 5


def load1():
    with open("/proc/loadavg", encoding="ascii") as f:
        return float(f.read().split()[0])


rows = []
for n in qubits:
    for workload, build in bc.WORKLOADS.items():
        program, noise = build(n)
        backends = {
            "one_per_shot": Simulator("SV", runtime="cuda", device_id=0, noise=noise),
            "one": Simulator("SV", runtime="cuda", device_id=0, noise=noise),
            "all": Simulator("SV", runtime="cuda", device_id="all", noise=noise),
        }
        shipped = engine_np._run_branched
        times = {arm: [] for arm in backends}
        loads = {arm: [] for arm in backends}
        counts = {}
        try:
            for repeat in range(REPEATS + 1):
                # Alternate the order each repeat, so drift hits both arms.
                order = list(backends.items())
                if repeat % 2:
                    order.reverse()
                for arm, backend in order:
                    if arm == "one_per_shot" and not per_shot:
                        continue
                    engine_np._run_branched = (
                        bc._per_shot if arm == "one_per_shot" else shipped
                    )
                    start = time.perf_counter()
                    got = (
                        backend.run(program, shots=SHOTS, simulation_config={"seed": 3})
                        .result()
                        .get_counts()
                    )
                    if repeat:
                        times[arm].append(time.perf_counter() - start)
                        loads[arm].append(load1())
                    counts.setdefault(arm, got)
                    assert got == counts[arm]
        finally:
            engine_np._run_branched = shipped
        medians = {arm: statistics.median(t) for arm, t in times.items() if t}
        row = {
            "workload": workload,
            "qubits": n,
            "shots": SHOTS,
            "devices": len(backends["all"]._devices()),
            "median_s": medians,
            "times_s": {arm: t for arm, t in times.items() if t},
            "load1_after_each": {arm: l for arm, l in loads.items() if l},
            "branching_speedup_one_gpu": (
                medians["one_per_shot"] / medians["one"] if per_shot else None
            ),
            "all_gpus_vs_one": medians["one"] / medians["all"],
            "counts_equal": len({tuple(sorted(c.items())) for c in counts.values()})
            == 1,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
never_slower = all(r["all_gpus_vs_one"] >= NOISE_FLOOR for r in rows)
out.write_text(
    json.dumps(
        {"noise_floor": NOISE_FLOOR, "never_slower": never_slower, "rows": rows},
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
print("never_slower", never_slower)
sys.exit(0 if all(r["counts_equal"] for r in rows) else 1)
