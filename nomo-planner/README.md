# Nomo Planner reference layer

This checkout contains the dependency-free Python reference used to generate
the A1–A3 uncertainty artifact and serious simulation surfaces for the browser
Lab. Published training observations are transcribed exactly from Narayanan et
al. (2021), arXiv:2104.04473, Tables 1–2. The public serving fixture in
`data/serving_sarathi_table4.json` preserves 12 measured Sarathi-Serve Table 4
rows; the derived replay is reported separately and is not promoted to a raw-
trace reproduction.

```powershell
python -m pytest -q
python studies/export_uncertainty.py
python studies/uncertainty_coverage.py
python studies/public_serving_validation.py
```

The bootstrap is stratified by strategy, refits the six calibrated A100
parameters in log-error space, and stores 256 reproducible samples in
`../src/planner/uncertainty.json`. The held-out study uses 128 samples for
each leave-one-out split and writes its detailed report under `docs/`.

`nomo_planner.layers` is the dependency-free reference for the shared model
graph and layer-aware training search. It parses a Hugging Face
`config.json`-shaped object, expands transformer nodes, applies contiguous
stage and lock repair, and keeps FP8/offload values labelled as assumptions.
Graphs with at most four nodes use exact exhaustive enumeration when the
candidate cap is not reached. Larger graphs use a deterministic,
repair-aware constrained NSGA-II search bounded by `max_candidates`; their
Pareto set is useful bounded search output, not a proof of global optimality.

`nomo_planner.experiments.parse_experiment_results_csv` accepts completed rows
from the recommendation-sensitive benchmark template only when their names
match the current candidates. It turns observed step times into explicit local
calibration observations and rejects unknown experiments instead of guessing.

`nomo_planner.artifact_ingestion` safely inspects Hugging Face/Nomo JSON,
safetensors metadata, Prometheus text, and optional ONNX graph artifacts.
Complete Hugging Face configs are lowered to a bounded transformer skeleton
containing only config-derived structure; incomplete configs remain `preview`
and do not receive inferred fields or performance numbers. Nomo graph JSON is
checked for non-empty unique node IDs and well-typed inputs/outputs. When the
optional `onnx` package is installed, `onnx.checker` validates the graph and a
generic structural lowerer emits graph inputs, initializers, operators,
outputs, explicit tensor type/shape metadata, and scalar/list attributes.
Unknown dimensions remain `null` and no FLOPs, bytes, timings, or measurements
are derived. The HTTP/SDK/MCP inspection paths can carry a binary artifact as
`{"filename":"model.onnx","encoding":"base64","base64":"..."}`;
the CLI provides `nomo-platform artifacts inspect-model PATH`. Without the
optional reader, binary ONNX remains an explicit `preview`, and full
framework-specific lowering remains an adapter boundary. None of these paths
load pickle-backed `.pt`/`.pth` files or claim hardware measurements.

The Python run auditor has bounded, versioned semantic fixtures for Megatron
Core 0.19.2, DeepSpeed 0.19.8, TorchTitan 0.2.2 TOML, and vLLM 0.6.2. It
validates only the documented common fields listed in
`../docs/FRAMEWORK_AUDITOR.md`; unknown CLI/JSON/TOML fields remain visible and
are retained in same-format exports. This is not exhaustive framework parsing,
browser parity, target-framework execution, or log calibration.
