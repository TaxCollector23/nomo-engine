# File-level audit record

Audit date: 2026-10-06 (America/Los_Angeles).

This record is the file-level companion to `ROADMAP_AUDIT.md`. The review used `git ls-files` as the scope, read the
tracked text/config/source files, checked binary assets and generated fixtures by type/size and build usage, searched
for stale links and unlabelled placeholders, and ran the repository's verification/build commands. The untracked
`nomo-ai/research/pilot/` directory was inspected and preserved; it was not silently added to GitHub.

## Inventory and disposition

| Repository / area | Files reviewed | Disposition |
|---|---:|---|
| `nomo-engine` root metadata, package files, entrypoint, README, reports | 12 | Used by build/deploy or documentation; stale deployment evidence refreshed. |
| `nomo-engine/docs/` | 20 after this record | Gate reports, serious-simulation audit/gates, prior-art citations, framework-auditor boundary, evidence/cost/calibration boundary, and reports retained; `SPEC.md`, `DEPLOY.md`, and `OBSERVABILITY.md` resolve the public documentation links. |
| `nomo-engine/frontend/` | 39 | Next.js neuromorphic launcher/dashboard, telemetry client, export UI, and admin surface typecheck/build successfully. The disabled unimplemented phase-coding control was removed. |
| `nomo-engine/nomo-planner/` | 47 | Python reference planner, serious simulation core, platform/API/SDK/CLI, public serving fixture/replay, safe artifact validation/lowering and Prometheus ingestion, studies, data, bounded versioned framework-auditor validation/round-trips, and tests; experiment-result CSV parsing and tests added. |
| `nomo-engine/scripts/` | 7 | Golden fixtures, parity verifier, benchmark runner, and simulation golden runner; `npm run verify` remains green. |
| `nomo-engine/src/` | 51 | Browser planner, Lab modules, product/cost/auditor surfaces, exports, styling, GraphIR, topology/timeline/distribution contracts, Python-core parity, Worker-backed Simulation Workbench, and the measured Verification & benchmarks Evidence panel; calibration file/result ingestion is local-only. |
| `nomo-engine/public/` | 2 | Favicon/logo assets used by the root app. |
| `nomo-ai` tracked tree | 25 | Landing routes, research pages, styling, package/deployment files, and assets reviewed; the only landing source change is the documentation/audit link repair. |

## Checks

- `nomo-planner`: 134 Python tests pass, including the completed-experiment CSV to local-calibration test, public
  Sarathi-Serve serving comparison, HF/Nomo structural lowering, safetensors metadata inspection, safe state-dict
  boundaries, Prometheus parsing, opaque framework-option round-trips, bounded versioned framework validation fixtures,
  HTTP/SDK model inspection, metadata-only source freshness, and explicit open-model cost coverage.
- Serious simulation layer: `nsga2_search` repairs constrained genomes and records its search provenance;
  training/serving/product runtime fixtures, public evidence, and report/export paths are covered.
- Root Lab: production build and Python/TypeScript parity verifier pass (14,337 checks, including cost/calibration
  boundary assertions).
- Browser simulation: `npm run simulation:golden` passes 123 checks, and the production build emits the Worker bundle
  consumed by `SimulationWorkbench` with a deterministic synchronous fallback. The evidence panel now displays the
  cited 12-row Sarathi comparison and its 72.98% replay MAPE, plus the measured verification and benchmark snapshot.
- Neuromorphic dashboard: `npm run typecheck` and `npm run build` pass; a live built-in Event-camera/AKD1500 search
  completed and exposed 550 designs, seven trade-offs, layer inspection, and export controls.
- Browser route matrix: all 11 Lab modules were directly loaded at 1440×900 and 390×844; each named module rendered
  without document-level horizontal overflow or browser error/warning logs.
- Production refresh: engine commits `c777a1f`, `cab46e7`, `b08e5fb`, `631e040`, `126326c`, `e300b1c`, and `19dd854` are pushed, the canonical mode-shell alias
  serves the current Worker bundle, the Simulation/Evidence surfaces were checked live, and the neuromorphic run
  completed without console errors.
- Landing: production build passes; documentation links now resolve to tracked engine documents.
- Release bundles: tracked-file archives for engine, planner, and landing are recorded under
  `release-bundles-a314204/`; the untracked landing `research/pilot/` directory was excluded and preserved.
- No tracked source file contains a newly introduced TODO/FIXME or a silent substitute for missing measurements.

## Remaining boundaries

The audit does not turn data-dependent claims into code claims. Framework-specific ONNX lowering and graph-to-cost
conversion beyond the generic structural boundary, the shared-graph neuromorphic placement/compiler adapter, full framework-option round trips, customer/co-design calibration, measured
product data, broader open-model serving/token-price coverage and network source verification, and a complete every-control browser accessibility matrix remain
explicitly listed as preview/open in `ROADMAP_AUDIT.md`.
