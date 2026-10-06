# Documentation index

The user guide, tutorials and API reference are the MkDocs site under [mkdocs/](mkdocs/) (built from `mkdocs.yml`);
the pages below are the plain Markdown pages in this folder.

## Understand the CUDA fork

| Page | What it answers |
| --- | --- |
| [cuda-coverage.md](cuda-coverage.md) | Which `runtime="cuda"` requests run on the GPU, and what happens to the rest? |
| [optimisations.md](optimisations.md) | What r9 and r10 change (gate tiles, exact simplification, several GPUs), and why accuracy and memory do not get worse. |
| [CUDA runtime guide](mkdocs/en/api/cupy-simulator.md) | Coverage, precision and the tested environment of the CUDA runtime, in the MkDocs API reference. |

## Check the evidence

| Page | What it answers |
| --- | --- |
| [benchmarks.md](benchmarks.md) | How much faster is the CUDA engine, against which CPU baselines and engine revisions, and how precise is it? |
| [../results/README.md](../results/README.md) | What each results file holds, how it was produced, and how to rerun it. |

## Upstream FatQat (by the FatQat authors)

| Page | What it answers |
| --- | --- |
| [upstream-README.md](upstream-README.md) | The upstream FatQat README: what FatQat is, installation, a first Program, development. |
| [compiler-v0.3-design.md](compiler-v0.3-design.md) | Upstream design note (in Chinese) for FatQat compiler v0.3. |
| [compiler-executable-interface-design.md](compiler-executable-interface-design.md) | Upstream design: how compiler results run directly on the matrix simulators. |
| [compiler-executable-interface-plan.md](compiler-executable-interface-plan.md) | Upstream implementation plan for that executable interface. |
