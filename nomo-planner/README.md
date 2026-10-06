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
graph and bounded layer-aware training search. It parses a Hugging Face
`config.json`-shaped object, expands transformer nodes, applies contiguous
stage and lock repair, and keeps FP8/offload values labelled as assumptions.

`nomo_planner.experiments.parse_experiment_results_csv` accepts completed rows
from the recommendation-sensitive benchmark template only when their names
match the current candidates. It turns observed step times into explicit local
calibration observations and rejects unknown experiments instead of guessing.

`nomo_planner.artifact_ingestion` safely inspects Hugging Face/Nomo JSON,
safetensors metadata, Prometheus text, and optional ONNX graph nodes. It never
implicitly unpickles `.pt`/`.pth` files; full binary graph lowering remains an
explicit adapter boundary. The platform service exposes the same model
inspection through `artifacts.inspect_model`, and the CLI provides
`nomo-platform artifacts inspect-model PATH`.
