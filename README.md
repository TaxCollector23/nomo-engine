# Nomo Engine — v6 (v0.6)

Supersedes **v5**. v6 adds enterprise profiles and project workspaces, bounded cycle evidence, hardware/deployment
co-search, and an enterprise Workbench on top of the calibration-aware post-training quantisation path and native integer
deployment driver, a six-level hardware/software workbench, architecture-family adapters, operational
modes, RTL/Chisel/EDA release artifacts, and hardware-in-the-loop measurement boundaries.

Tri-domain (continuous / spiking / symbolic) hardware-aware architecture search and compiler with
enterprise design contracts, project workspaces, bounded cycle simulation, and hardware/deployment co-search.
Upload a model, pick a chip and a goal, and Nomo finds the best trade-offs between energy, latency and
accuracy, explains them, and exports the chosen design to a structured package containing model metadata,
runtime code, software SDK files, RTL/EDA artifacts, validation evidence, and an executive PDF brief.

- Plain-English guide for users: `GUIDE.md`
- Full math and design: `docs/SPEC.md`
- Deployment: `DEPLOY.md`
- Logs and admin: `OBSERVABILITY.md`

## Setup
    pip install -e ".[server,dev]"                    # add [coreml] on a supported Apple environment
    python setup.py build_ext --inplace     # optional C++ kernel; skipped automatically if no compiler
    pytest -q --ignore=tests/test_v4.py               # platform-independent suite

## Use
    nomo serve --port 8765                                  # API + live telemetry
    cd frontend && npm install && npm run dev               # dashboard at http://localhost:3000
    nomo search  --model attitude_policy --hardware akd1500 --out run.json
    nomo compile --model attitude_policy --hardware akd1500 --genome run.json --out build/
    nomo-cli pipeline --config nomo.yaml.example --out release.tar.gz
    nomo emulate --artifact artifact.json --config emulator.json --out emulation.json

## Live deployment

| | |
|---|---|
| Engine — live dashboard | https://frontend-gray-ten-c3tj1luab7.vercel.app/ |
| Engine — admin panel | https://frontend-gray-ten-c3tj1luab7.vercel.app/admin |
| Engine — live backend | https://nomo-engine.onrender.com/ |
| Main website | https://nomoaiprototype.vercel.app/ |
| Engine — repo | https://github.com/TaxCollector23/nomo-engine |
| Prototype — repo | https://github.com/TaxCollector23/nomo-ai |

## What's new in v0.6

- **Calibration + PTQ:** attach 100–500 `.npz`, `.npy`, or JSON tensors from the launcher/API. MSE and
  KL threshold selection is recorded per activation edge and used by integer export; labels are retained
  as provenance but never turned into an accuracy claim automatically.
- **Native integer deployment:** the canonical `runtime/deploy_model.py` uses integer accumulators,
  Q16.16 guard math, deterministic saturation, and embedded golden vectors. It is emitted only when the
  selected graph has a supported integer lowering; otherwise the package makes the float reference driver
  explicit.
- **Six-level Workbench:** system topology, domain partitioning, hardware graph, cycle-emulation boundary,
  RTL, and silicon floorplan views are available at `GET /runs/{id}/workbench`. Candidate target weights
  can be changed without silently changing the recorded run.
- **Architecture families:** FNO, ViT/attention, and GNN/message-passing JSON graph blocks preserve their
  spatial, token, or graph contracts. Backends that do not yet lower a family report the limitation instead
  of flattening it invisibly.
- **Hardware release artifacts:** structured exports include SystemVerilog PE/core modules, Chisel boundary,
  generic Yosys/OpenROAD scripts, CMake, a SystemC integration boundary, and proxy PPA marked as proxy.
- **Operational modes and HITL:** low-power neuromorphic, hard real-time, radiation-hardened, and on-chip
  learning contracts are catalogued and carried into manifests. `nomo hitl` accepts a trusted benchmark-agent
  result; no arbitrary remote shell execution is part of the protocol.
- **Enterprise design profiles:** versioned YAML/JSON profiles capture organization, project, safety policy,
  allowed precision, hardware assumptions, objectives, constraints, and required evidence. Start with
  `nomo/enterprise/examples/hard_realtime_robotics.yaml` and validate one with `POST /enterprise/profile/validate`.
- **Project workspaces:** `POST /projects` creates an owner-scoped project and `RunIn.project_id` associates
  searches with it. Set `NOMO_STATE_DB` to a durable SQLite path for local/on-prem persistence; unset means
  intentionally ephemeral storage.
- **Hardware/deployment co-search:** `POST /runs/{id}/co-design` explores bounded PE-array, SRAM, bandwidth,
  and precision choices around existing deployment candidates. Results are analytic priors, not silicon claims.
- **Bounded cycle evidence:** `POST /runs/{id}/emulation` or `nomo emulate` produces deterministic simulated
  cycles, memory references, cache hits/misses, and spike/event counts. Results are explicitly labelled simulated.

## What's retained from v0.4
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
- The platform-independent suite passes with `pytest -q --ignore=tests/test_v4.py`; the v4 full-bundle test
  additionally needs the native Core ML ML-storage extension. On Linux it is reported as unavailable when
  that extension is missing; run the Core ML export test on supported macOS tooling.
- Frontend: strict `tsc --noEmit` and `next build` pass after the enterprise Workbench and calibration UI changes.
- Existing v4 coverage still exercises policy/pins, repair idempotence, crossing penalties, hardware
  overrides, presets, ONNX ingestion, safe checkpoint loading, export self-checks, Copilot, and API routes.
- Memory: worst case measured (camera model search + all formats in one export) peaks at 382 MB, under
  Render's 512 MB; exports are serialised one at a time.

## Evidence boundaries and limits (read before quoting numbers)
- Energy/latency coefficients are **placeholders** unless you enter your chip's numbers.
- Accuracy is an **estimate** from per-layer sensitivities, not a measurement; uploaded models use default
  sensitivities.
- PTQ calibration measures runtime ranges and quantisation/reconstruction fidelity. It does **not** promote
  task accuracy without labeled evaluation and an oracle result.
- PPA, thermal density, and physical power remain proxy values until the generated design is synthesized or
  measured through HITL. The bounded Nomo emulator can provide deterministic **simulated** cycle/cache evidence,
  but simulated cycles are not a board measurement.
- Built-in model weights are synthetic demo weights. Upload trained weights before making model-quality claims.
- FNO/ViT/GNN blocks preserve their graph contracts but only backends listed in their architecture profile
  are enabled; unsupported lowerings are included as explicit `.unavailable.txt` notes in the package.
- The PyTorch path of `deploy_model.py` was not executed in the build environment (PyTorch unavailable);
  the script verifies itself on first run (`--check`). Its numpy path is tested.
- Core ML packages are built and structurally validated on Linux; running them needs macOS/iOS.
- `.pt/.pth` uploads: weights-only `state_dict`; structure is inferred (ReLU between layers, stride 1).
  The `.pt` tests use files synthesised in PyTorch's format; a real-PyTorch test runs where torch exists.
- Not modelled: phase coding, FP16. Not implemented: SystemC/Verilator/Gem5 execution, branching/residual graphs, grouped convolutions,
  surrogate-gradient fine-tuning, multi-chip partitioning, TTFS/conv C11 lowering, full operator-family
  ONNX/Core ML lowering, MLIR/microTVM output, and closed-loop silicon PPA without a target measurement.
