# Nomo Engine — v4 (v0.4)

Supersedes **v3 - fixed**. v4 adds user model upload, per-layer control, presets, six export formats
(PDF, PyTorch, ONNX, Core ML, NIR, C11), and Nomo Copilot on a rebuilt dashboard.

Tri-domain (continuous / spiking / symbolic) hardware-aware architecture search and compiler.
Upload a model, pick a chip and a goal, and Nomo finds the best trade-offs between energy, latency and
accuracy, explains them, and exports the chosen design to PyTorch, ONNX, Core ML, NIR, C11 and a PDF brief.

- Plain-English guide for users: `GUIDE.md`
- Full math and design: `docs/SPEC.md`
- Deployment: `DEPLOY.md`
- Logs and admin: `OBSERVABILITY.md`

## Setup
    pip install -e ".[server,coreml,dev]"
    python setup.py build_ext --inplace     # optional C++ kernel; skipped automatically if no compiler
    pytest -q                               # 97 tests (+1 that runs only where PyTorch is installed)

## Use
    nomo serve --port 8765                                  # API + live telemetry
    cd frontend && npm install && npm run dev               # dashboard at http://localhost:3000
    nomo search  --model attitude_policy --hardware akd1500 --out run.json
    nomo compile --model attitude_policy --hardware akd1500 --genome run.json --out build/

## Live deployment

| | |
|---|---|
| Engine — live dashboard | https://frontend-gray-ten-c3tj1luab7.vercel.app/ |
| Engine — admin panel | https://frontend-gray-ten-c3tj1luab7.vercel.app/admin |
| Engine — live backend | https://nomo-engine.onrender.com/ |
| Main website | https://nomoaiprototype.vercel.app/ |
| Engine — repo | https://github.com/TaxCollector23/nomo-engine |
| Prototype — repo | https://github.com/TaxCollector23/nomo-ai |

## What's new in v0.4
- **Your own models:** upload `.onnx`, PyTorch `state_dict` (`.pt/.pth`, read without executing code) or a
  `nomo.graph/1` JSON graph. Layer shapes, sizes, compute and possible domains are extracted automatically.
- **Control:** per-layer locks on domain and weight precision (hard invariants), domain toggles
  (continuous / spiking / symbolic), spike-coding restrictions, a domain-crossing penalty and a minimum
  saving for mixed designs, custom chip parameters, and search hyperparameters.
- **Presets:** Battery Saver, Ultra-Low Latency, Balanced Edge, Strict Safety.
- **Exports:** PDF audit brief, `deploy_model.py` (PyTorch nn.Module + self-check, numpy fallback), ONNX per
  continuous section (safety guards included), Core ML `.mlpackage`, float NIR for any design (strict +
  extended), integer NIR and header-only C11 for supported designs. Each format says why when unavailable.
- **Nomo Copilot:** answers computed from the run (counterfactual re-evaluation, archive search) with
  one-click actions; optional Claude phrasing when `NOMO_ANTHROPIC_API_KEY` is set.
- **New dashboard:** four-step launcher, plain-English tooltips, trade-off filters, clickable design graph
  with crossing badges and lock-and-rerun, export and Copilot drawers.

## Verification
- `pytest -q`: 97 pass. v4 tests cover policy/pins (repair idempotence under policy, 200 random genomes),
  crossing penalty and threshold, hardware overrides, presets, ONNX ingestion vs ONNX Runtime (< 1e-5),
  malicious-pickle refusal, every export format end to end (PyTorch self-check, stock `nir.read`, ONNX
  sections vs reference, C example compiled and run), Copilot, and the new API endpoints.
- Frontend: strict `tsc`, `next build`, and a headless-browser session (upload, lock, search, inspect,
  Copilot, export download) against a live server.
- Memory: worst case measured (camera model search + all formats in one export) peaks at 382 MB, under
  Render's 512 MB; exports are serialised one at a time.

## Limits (read before quoting numbers)
- Energy/latency coefficients are **placeholders** unless you enter your chip's numbers.
- Accuracy is an **estimate** from per-layer sensitivities, not a measurement; uploaded models use default
  sensitivities.
- The PyTorch path of `deploy_model.py` was not executed in the build environment (PyTorch unavailable);
  the script verifies itself on first run (`--check`). Its numpy path is tested.
- Core ML packages are built and structurally validated on Linux; running them needs macOS/iOS.
- `.pt/.pth` uploads: weights-only `state_dict`; structure is inferred (ReLU between layers, stride 1).
  The `.pt` tests use files synthesised in PyTorch's format; a real-PyTorch test runs where torch exists.
- Not modelled: phase coding, FP16. Not implemented: branching/residual graphs, grouped convolutions,
  surrogate-gradient fine-tuning, multi-chip partitioning, TTFS/conv C11 lowering, MLIR/microTVM output.
