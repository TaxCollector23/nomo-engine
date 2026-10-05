# Gate 1 report — shared graph and deployment surfaces

Status: Partial.

Built:

- One graph contract expands model configuration into embedding, attention, MLP, and output nodes.
- Train performs bounded/exhaustive layer-aware search with locks and contiguous stages.
- Serve performs a bounded per-node weight/KV precision search with memory, latency, cost, and quality-budget constraints.
- Neuromorphic can download a shared graph contract while preserving the existing compiler boundary.
- The mode shell restores the six-mode navigation and links Neuromorphic to the existing validated dashboard.

Open:

- The historical Neuromorphic dashboard is not yet a shared-graph placement recommendation or compiler adapter. The new export is labelled as a contract surface rather than presented as complete.
- Per-layer serving calibration and task-quality measurements require customer or published data.
