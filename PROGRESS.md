# Nomo Engine progress

Current release status: the serious simulation-core implementation and the cited public serving comparison are
implemented, tested, and deployed. Customer evidence, raw serving traces, and the neuromorphic compiler adapter remain
explicitly Preview/open where the required artifacts or server contract are absent.

## 2026-10-08 enterprise surface and deployment refresh

- Added browser-local Enterprise workspace review state: named projects now hold planner/simulation decisions,
  locally labelled Draft/Needs review/Approved statuses, and a module-linked journal for decisions, risks, and
  measurement requests. JSON export/import remains portable; remote sync is still explicit and session-token-only.
- Added shared-platform synchronization for new runs, review metadata, and journal notes as JSON artifacts. The
  dependency-free platform HTTP boundary now exposes bearer-authenticated capabilities, allowlisted CORS, health
  checks, PATCH persistence, and browser-readable authenticated errors; it still does not claim tenancy, backups, or
  permissions beyond deployment configuration.
- Modernized the separate neuromorphic dashboard on commit `19fa250`: Next 16.4.0, React 19.3.0, current React
  Three bindings, strict typecheck, real ESLint configuration, and browser state handling without synchronous
  effect resets. Production audit reports zero vulnerabilities; the full development audit still has Tailwind 3 and
  Next ESLint-plugin advisories whose fixes require a Tailwind major migration.
- Verification: root build, `npm run verify` (14,337 checks; worst relative difference `4.37e-16`),
  `npm run simulation:golden` (123 checks), root production audit, dashboard typecheck/lint/build, dashboard
  production audit, Python compileall, and 135 planner tests all passed.
- Engine commits `19fa250`, `e6e6555`, `d43b498`, and `d569905` are pushed to `main`; landing commit `6a15658` is
  pushed to `master`. Vercel deployments are READY: mode shell `dpl_2zXDvtSHiUyr78z2EkauHMfEzvSk`, neuromorphic
  dashboard `dpl_8qgaHU8QGF3ro9pAsWiA8Vds5XSh`, and landing `dpl_2DPpmWpJnJsXKwcbdAUPro1a3SNK`.

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
  Prometheus text, and checker-backed generic ONNX structural lowering for graph boundaries, operators, attributes, and
  explicit tensor metadata; binary weights are never implicitly unpickled. Exposed model inspection through the platform
  service, HTTP/SDK/MCP base64 binary envelope, and CLI.
- Added explicit versioned bounded semantic-validation fixtures for Megatron Core 0.19.2, DeepSpeed 0.19.8, TorchTitan
  0.2.2 TOML, and vLLM 0.6.2. Unknown CLI tokens, opaque values, and nested JSON/TOML fields remain preserved and
  visibly outside the fixture boundary; see `docs/FRAMEWORK_AUDITOR.md`.
- Verification after this pass: 134 Python tests, production TypeScript build, 14,337 parity checks, and 123
  simulation golden checks.

## 2026-10-06 production refresh and live verification

- Pushed engine commits `c777a1f`, `cab46e7`, `b08e5fb`, `631e040`, `126326c`, `e300b1c`, and `19dd854` to `main` after the public-serving and safe-artifact
  boundary pass.
- Vercel deployment `dpl_88jwbgJb6otFxUwcNe5zUpbT2fDe` is READY at
  `https://nomo-engine-klb7z3drv-rangan-alt.vercel.app/`; the canonical mode-shell alias is assigned to this
  deployment and serves the same `index-c-I02bbG.js` bundle.
- Live browser verification exercised all 11 mode routes at 1440×900 and 390×844, with no document overflow or
  console errors/warnings; it also checked the Simulation core, Training/Serving controls, Evidence panel, cited
  Sarathi row, and the Neuromorphic link. The live neuromorphic dashboard then completed an
  Event-camera/AKD1500 search (`/runs/620b864c08bd`) with 550 designs tried, seven trade-offs, layer inspection, and no
  console errors or warnings.
- The landing source tree remains unchanged at `98a91d7`; its pre-existing `research/pilot/` directory remains
  untracked and preserved.
- Evidence/cost/calibration boundary refresh is pushed as engine commit `27b8824`; the final benchmark-surface
  deployment is recorded below at `dpl_7ezLreq3DhKwv4jj9JfEK4G3YUcH`.

## 2026-10-06 measured verification and benchmark surface

- Re-ran the complete release suite after the Evidence update: 134 Python tests, 14,337 TypeScript parity checks
  (worst relative difference `4.37e-16`), and 123 simulation golden checks all passed.
- The deterministic `npm run bench` snapshot measured `llm_training` at 7,680 plans / 63 ms search, the widened
  13,440-plan training sweep at 150 ms, `llm_inference` at 216 plans / 2 ms, and `arch_codesign` at 1,215 plans /
  13 ms. Explanation took 1 ms, 1 ms, 0 ms, and 0 ms respectively.
- Added the visible **Verification & benchmarks** card to the Evidence module. It shows the test counts, benchmark
  rows, run date, command source, and a clear local-snapshot limitation; it does not present those timings as
  production or hardware guarantees.

## Completed

- Python-first layer graph with fair global comparison, separate precision/per-layer gain, pipeline/communication/offload accounting, published model presets, and fixed-seed browser parity.
- Shared-graph Train and bounded Serve recommendations with explicit assumptions and locks.
- Restored multi-mode shell with the historical Neuromorphic dashboard link.
- Run auditor for Megatron, DeepSpeed JSON, TorchTitan TOML, vLLM, and log metrics, with loss-aware same-format export and
  bounded versioned semantic validation.
- Browser-local customer CSV scale fit with leave-one-out error, recommendation-sensitive experiment ranking, unknown-option warnings, and a known-preset recommendation bridge.
- Reference product estimators and editable browser Product Studio for chip design, RL scheduling, reliability/goodput, fleet sizing, fine-tuning, and TCO, with JSON/CSV exports.
- Cited cost tracker with model-card compute rows, provider token-price rows, a local physical-cost calculator, and provider-versus-physical comparison.
- Cost coverage now explicitly reports two open-model training rows and zero open-model serving/token-price rows;
  checked-in source dates are validated without fetching URLs, and undated model-card rows remain undated.
- Nomo graph-contract re-import and a conservative interval-regret “Safest plan” option for calibrated training.
- Downloadable recommendation-experiment CSVs can be loaded back into the browser-local calibration fit; completed rows are matched to the current candidate names and unknown names are rejected.
- Neuromorphic dashboard typecheck/build is green; the launcher no longer advertises an unimplemented phase-coding control.
- Gate reports for Gates 0–5 in docs/.
- Line-by-line roadmap audit in `docs/ROADMAP_AUDIT.md`.
- File-by-file scope and disposition record in `docs/FILE_AUDIT.md`; linked engine specification, deployment, and observability docs are now present.

## Verification evidence

- Python: 135 tests passed in nomo-planner.
- TypeScript production build: passed.
- Neuromorphic Next.js dashboard: typecheck and production build passed.
- Python/TypeScript parity: npm run verify passed, 14,337 checks, worst relative difference 4.37e-16; the simulation
  golden passed 123 checks including zero-bubble, parallel-collective, and embedding-dependency coverage.
- Existing 22-row calibration artifact and fixed-seed layer goldens remain green.
- Product estimators pass Python physics-sanity tests and browser product golden parity.

## Honest boundaries

- Gate 0 remains partial because current assumptions show 0% additional per-layer gain for Llama 3 8B/70B, even though precision gain is approximately 15.8%.
- Neuromorphic remains the existing validated dashboard, not a new shared-graph compiler adapter.
- Serving and co-design intervals are not calibrated; only A100 training has a published calibration artifact.
- Customer CSV fitting is local preview evidence; leave-one-out error is reported but it is not the published six-parameter hardware refit.
- Customer calibration intervals are labelled empirical in-sample central 90% intervals, not predictive guarantees;
  customer run/cluster identity is required before fitting.
- Product Studio values are user inputs, not measured silicon, queueing, quality, or procurement guarantees.
- Cost tracker rows are hand-entered and must be rechecked before procurement.

## Remaining roadmap work

- Safe ONNX/state-dict inspection, checker-backed generic structural ONNX lowering, and bounded HF/Nomo structural
  lowering are implemented in the Python boundary; framework-specific graph-to-cost lowering and the Neuromorphic
  shared-graph placement/compiler adapter still require the hosted compiler contract.
- Model inspection now has matching PlatformService, HTTP, Python SDK, CLI, and MCP entry points.
- Full framework-option semantic validation, browser parity for the Python fixture reports, and arbitrary-model/hardware
  auditor recommendation remain open; bounded fixture fields are validated and all other options remain preserved.
- Customer-held-out uncertainty for serving/co-design and server-side/published experiment-result ingestion.
- Product pack joint chip/software optimization, customer calibration, and richer procurement uncertainty.
- Broader open-model serving/token-price coverage, historical cost ranges, and network-backed source verification.
- Dedicated 1440px/390px live viewport checks pass with no horizontal overflow; a complete every-control accessibility matrix remains open.

## Deployment evidence

- Engine current runtime release commits: `27b8824`, `3610bb9`, `edd3a49`, and `e34c8c1` on main, with the audit/deployment records also pushed to
  github.com/TaxCollector23/nomo-engine.
- Mode-shell Vercel production deployment: dpl_7ezLreq3DhKwv4jj9JfEK4G3YUcH.
- Mode-shell unique URL: https://nomo-engine-p6ysgomgo-rangan-alt.vercel.app/
- Stable mode-shell alias: https://frontend-gray-ten-c3tj1luab.vercel.app/
- Both mode-shell URLs returned HTTP 200; the canonical stable alias serves the current
  `index-xLodnwaL.js`, `index-Bcup2DCr.css`, `worker-C-FLG9f1.js`, and `jszip.min-D9eUZgo9.js` build.
- Live browser smoke showed the Simulation core module, Training/Serving controls, the cited Evidence rows and Preview
  labels, and the historical Neuromorphic dashboard link. The checked stable tab had no console errors or warnings.
- Neuromorphic dashboard deployment: dpl_MN1dV6Cuyd2KhadvmxdNrdwU9TYq; stable alias https://nomo-engine-dashboard.vercel.app/.
- Landing deployment: dpl_BBo1GiaehNAm71M2A7aStC9e1VfJ; stable alias https://nomoailanding.vercel.app/.
- The landing repository had no source change in this simulation release; master remains at 98a91d7 and its existing
  production deployment remains intact. The untracked research/pilot directory was preserved and not deployed.
- Responsive layout code and Worker packaging are present; all 11 mode routes were directly checked at 1440×900 and
  390×844 with no document overflow or console errors/warnings.

## Current deployment evidence

- Mode-shell commit `d569905`: `dpl_2zXDvtSHiUyr78z2EkauHMfEzvSk`, direct URL
  `https://nomo-engine-6f8xam7te-rangan-alt.vercel.app/`, stable alias
  `https://frontend-gray-ten-c3tj1luab.vercel.app/`. The stable shell serves `index-B11IWhz6.js`; the deployed Lab
  bundle contains `Review journal`, `Optional shared platform`, `Approved locally`, and `workspace-note` markers.
- Neuromorphic dashboard commit `19fa250`: `dpl_8qgaHU8QGF3ro9pAsWiA8Vds5XSh`, stable alias
  `https://nomo-engine-dashboard.vercel.app/`; `/` and `/admin` return HTTP 200.
- Landing commit `6a15658`: `dpl_2DPpmWpJnJsXKwcbdAUPro1a3SNK`, stable alias
  `https://nomoailanding.vercel.app/`; production deployment is READY and returns HTTP 200.

## Delivery bundles

- C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/nomo-engine-a314204.zip
- C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/nomo-planner-a314204.zip
- C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/nomo-ai-a314204.zip
- C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/nomo-engine-27b8824.zip
- C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/nomo-planner-27b8824.zip
- C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/nomo-engine-b1d3bdb.zip
- C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/nomo-planner-b1d3bdb.zip
- C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/nomo-ai-b1d3bdb.zip
