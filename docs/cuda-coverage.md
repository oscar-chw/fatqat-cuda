# CUDA coverage: which requests run on the GPU?

Which requests run on the GPU, and what happens to the rest: thin solid arrows end in `BackendValidationError`, dotted ones have no CUDA path at all.

```mermaid
flowchart LR
  R(["request with<br/>runtime='cuda'"]):::data
  subgraph YES["Covered: one state per GPU"]
    SV["statevector: ideal, and per-shot<br/>trajectories: channels, reset,<br/>mid-circuit, feedforward"]:::key
    DM["density matrix: channels,<br/>reset, mid-circuit, feedforward"]:::key
    OP["unitary and<br/>superoperator maps"]:::key
    EST["Estimator: exact and<br/>sampled SV / DM"]:::key
    SC["superconducting<br/>hardware profiles"]:::key
    SPREAD["several GPUs: sweep rows,<br/>or a run's shots, per GPU"]:::key
  end
  subgraph NO["Not covered"]
    ATOM["AtomArraySimulator:<br/>occupancy and loss"]:::gate
    PULSE["pulse emulation"]:::ext
    APPLE["Apple GPUs"]:::ext
    MULTI["one state split<br/>across several GPUs"]:::ext
  end
  R ==>|"single_pass and per_shot shapes"| SV
  R ==>|"per-shot runs with shot branching"| DM
  R ==>|"operator shape"| OP
  R ==>|"state stays resident"| EST
  R ==>|"device_id forwarded"| SC
  R ==>|"device_id=(0, 1, ...) or 'all'"| SPREAD
  R -->|"rejected at construction"| ATOM
  R -.->|"no CUDA runtime;<br/>CPU path unchanged"| PULSE
  R -.->|"no Metal engine"| APPLE
  R -.->|"each state lives<br/>on one device"| MULTI

  classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
  classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
  classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
  classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
  classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
  classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
```

Where in the code: `src/fatqat/simulator/simulator.py` (`_validate_runtime_config`), `src/fatqat/simulator/_engine/cupy.py` (`_supported_execution_shapes`, `CupySVEngine`), `src/fatqat/simulator/fake_atom_array.py`, `src/fatqat/simulator/fake_superconducting.py`; coverage table: [CUDA runtime guide](mkdocs/en/api/cupy-simulator.md).
