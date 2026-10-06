# Nomo Engine progress

Current release status: the serious simulation-core implementation and the cited public serving comparison are
implemented, tested, and deployed. Customer evidence, raw serving traces, and the neuromorphic compiler adapter remain
explicitly Preview/open where the required artifacts or server contract are absent.

## 2026-10-05 simulation-core implementation

- Added a Python-first serious-simulation layer: operator graph, roofline efficiency curves, topology/collectives,
  training Gantt and memory events, serving request simulation, distributions, calibration/held-out metrics, and
  artifact-driven M4–M11 product simulations.
- Added deterministic TypeScript GraphIR/Worker contracts, Python-core aliases, topology/timeline/distribution
  contracts, a production Worker bundle, and `npm run simulation:golden` parity fixtures (123 checks).
- Added the Lab Simulation Workbench and Evidence registry with explicit Preview labels, source/provenance, training
  and serving timelines, metrics, server-only boundaries, and JSON export.
- Added the SQLite platform store, run comparison, HTML/PDF reports, local/HTTP API, CLI, Python SDK, and MCP-style
  JSON-RPC operations including `simulations.run`.
- Added tested M1/M2 framework projections, M3 audit diff/report exports, and a living `docs/EXECUTION_AUDIT.md`.
- Added a deterministic, repair-aware NSGA-II search utility with population/generation/evaluation provenance and
  focused tests for constrained genomes.
- Honest gate boundary: Study 1 training evidence is present (5.9% PTD-P, 0.94 Spearman, 18/22 interval coverage),
  and 12 measured Sarathi-Serve Table 4 serving rows are imported. The derived replay matches all 12 rows but has
  72.98% MAPE because the paper's raw request trace is unavailable; customer telemetry remains Preview.

## 2026-10-06 evidence and input-boundary pass

- Added the cited Sarathi-Serve Table 4 fixture, reproducible derived replay, 12/12 matching-row report, and Evidence
  dashboard row with a direct source link and explicit 72.98% MAPE Preview boundary.
- Added safe Hugging Face/Nomo JSON structural validation and bounded transformer-skeleton lowering, safetensors metadata,
  Prometheus text, and optional ONNX/state-dict inspection; binary weights are never implicitly unpickled. Exposed model
  inspection through the platform service and CLI.
- Preserved unknown Megatron/vLLM CLI tokens, opaque option values, and nested DeepSpeed JSON fields in same-format audit
  exports while flagging them as semantically unvalidated.
- Verification after this pass: 114 Python tests, production TypeScript build, 14,332 existing parity checks, and 123
  simulation golden checks.

## 2026-10-06 production refresh and live verification

- Pushed engine commits `c777a1f`, `cab46e7`, `b08e5fb`, and `631e040` to `main` after the public-serving and safe-artifact
  boundary pass.
- Vercel deployment `dpl_8zH3TrjGnqdutQJEgTZGuGbZeywz` is READY at
  `https://nomo-engine-cvt11oeaf-rangan-alt.vercel.app/`; the canonical mode-shell alias is assigned to this
  deployment and serves the same `index-8B-nBfyb.js` bundle.
- Live browser verification exercised the restored mode shell, Simulation core, Training/Serving controls, Evidence
  panel, cited Sarathi row, and the Neuromorphic link. The live neuromorphic dashboard then completed an
  Event-camera/AKD1500 search (`/runs/620b864c08bd`) with 550 designs tried, seven trade-offs, layer inspection, and no
  console errors or warnings.
- The landing source tree remains unchanged at `98a91d7`; its pre-existing `research/pilot/` directory remains
  untracked and preserved.

## Completed

- Python-first layer graph with fair global comparison, separate precision/per-layer gain, pipeline/communication/offload accounting, published model presets, and fixed-seed browser parity.
- Shared-graph Train and bounded Serve recommendations with explicit assumptions and locks.
- Restored multi-mode shell with the historical Neuromorphic dashboard link.
- Run auditor for Megatron, DeepSpeed JSON, vLLM, and log metrics, with loss-aware same-format export.
- Browser-local customer CSV scale fit with leave-one-out error, recommendation-sensitive experiment ranking, unknown-option warnings, and a known-preset recommendation bridge.
- Reference product estimators and editable browser Product Studio for chip design, RL scheduling, reliability/goodput, fleet sizing, fine-tuning, and TCO, with JSON/CSV exports.
- Cited cost tracker with model-card compute rows, provider token-price rows, a local physical-cost calculator, and provider-versus-physical comparison.
- Nomo graph-contract re-import and a conservative interval-regret “Safest plan” option for calibrated training.
- Downloadable recommendation-experiment CSVs can be loaded back into the browser-local calibration fit; completed rows are matched to the current candidate names and unknown names are rejected.
- Neuromorphic dashboard typecheck/build is green; the launcher no longer advertises an unimplemented phase-coding control.
- Gate reports for Gates 0–5 in docs/.
- Line-by-line roadmap audit in `docs/ROADMAP_AUDIT.md`.
- File-by-file scope and disposition record in `docs/FILE_AUDIT.md`; linked engine specification, deployment, and observability docs are now present.

## Verification evidence

- Python: 114 tests passed in nomo-planner.
- TypeScript production build: passed.
- Neuromorphic Next.js dashboard: typecheck and production build passed.
- Python/TypeScript parity: npm run verify passed, 14,332 checks, worst relative difference 4.37e-16; the simulation
  golden passed 123 checks including zero-bubble, parallel-collective, and embedding-dependency coverage.
- Existing 22-row calibration artifact and fixed-seed layer goldens remain green.
- Product estimators pass Python physics-sanity tests and browser product golden parity.

## Honest boundaries

- Gate 0 remains partial because current assumptions show 0% additional per-layer gain for Llama 3 8B/70B, even though precision gain is approximately 15.8%.
- Neuromorphic remains the existing validated dashboard, not a new shared-graph compiler adapter.
- Serving and co-design intervals are not calibrated; only A100 training has a published calibration artifact.
- Customer CSV fitting is local preview evidence; leave-one-out error is reported but it is not the published six-parameter hardware refit.
- Product Studio values are user inputs, not measured silicon, queueing, quality, or procurement guarantees.
- Cost tracker rows are hand-entered and must be rechecked before procurement.

## Remaining roadmap work

- Safe ONNX/state-dict inspection and bounded HF/Nomo structural lowering are implemented in the Python boundary; full
  binary graph lowering and the Neuromorphic shared-graph placement/compiler adapter still require the hosted compiler
  contract.
- Model inspection now has matching PlatformService, HTTP, Python SDK, CLI, and MCP entry points.
- Framework-option semantic validation and arbitrary-model/hardware auditor recommendation remain open; opaque
  Megatron/DeepSpeed/vLLM options are now preserved in same-format exports.
- Customer-held-out uncertainty for serving/co-design and server-side/published experiment-result ingestion.
- Product pack joint chip/software optimization, customer calibration, and richer procurement uncertainty.
- Broader cost history/coverage and source-freshness checks.
- Dedicated 1440px/390px live viewport checks pass with no horizontal overflow; a complete every-control accessibility matrix remains open.

## Deployment evidence

- Engine current release commits: c777a1f, cab46e7, b08e5fb, and 631e040 on main, pushed to
  github.com/TaxCollector23/nomo-engine after the a314204 runtime release.
- Mode-shell Vercel production deployment: dpl_8zH3TrjGnqdutQJEgTZGuGbZeywz.
- Mode-shell unique URL: https://nomo-engine-cvt11oeaf-rangan-alt.vercel.app/
- Stable mode-shell alias: https://frontend-gray-ten-c3tj1luab.vercel.app/
- Both mode-shell URLs returned HTTP 200; the canonical stable alias was explicitly reassigned and serves
  `index-8B-nBfyb.js`, `index-DefewnCY.css`, `worker-C-FLG9f1.js`, and `jszip.min-DMBnj76E.js`.
- Live browser smoke showed the Simulation core module, Training/Serving controls, the cited Evidence rows and Preview
  labels, and the historical Neuromorphic dashboard link. The checked stable tab had no console errors or warnings.
- Neuromorphic dashboard deployment: dpl_MN1dV6Cuyd2KhadvmxdNrdwU9TYq; stable alias https://nomo-engine-dashboard.vercel.app/.
- Landing deployment: dpl_BBo1GiaehNAm71M2A7aStC9e1VfJ; stable alias https://nomoailanding.vercel.app/.
- The landing repository had no source change in this simulation release; master remains at 98a91d7 and its existing
  production deployment remains intact. The untracked research/pilot directory was preserved and not deployed.
- Responsive layout code and Worker packaging are present; prior 1440×900 and 390×844 smoke evidence remains recorded.

## Delivery bundles

- C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/nomo-engine-a314204.zip
- C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/nomo-planner-a314204.zip
- C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/nomo-ai-a314204.zip
