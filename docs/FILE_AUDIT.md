# File-level audit record

Audit date: 2026-10-05 (America/Los_Angeles).

This record is the file-level companion to `ROADMAP_AUDIT.md`. The review used `git ls-files` as the scope, read the
tracked text/config/source files, checked binary assets and generated fixtures by type/size and build usage, searched
for stale links and unlabelled placeholders, and ran the repository's verification/build commands. The untracked
`nomo-ai/research/pilot/` directory was inspected and preserved; it was not silently added to GitHub.

## Inventory and disposition

| Repository / area | Files reviewed | Disposition |
|---|---:|---|
| `nomo-engine` root metadata, package files, entrypoint, README, reports | 12 | Used by build/deploy or documentation; stale deployment evidence refreshed. |
| `nomo-engine/docs/` | 14 after this record | Gate reports and audit retained; `SPEC.md`, `DEPLOY.md`, and `OBSERVABILITY.md` added because the landing page previously linked to missing documents. |
| `nomo-engine/frontend/` | 39 | Next.js neuromorphic launcher/dashboard, telemetry client, export UI, and admin surface typecheck/build successfully. The disabled unimplemented phase-coding control was removed. |
| `nomo-engine/nomo-planner/` | 25 | Python reference planner, studies, data, and tests; experiment-result CSV parsing and tests added. |
| `nomo-engine/scripts/` | 6 | Golden fixtures, parity verifier, and benchmark runner; `npm run verify` remains green. |
| `nomo-engine/src/` | 35 | Browser planner, Lab modules, product/cost/auditor surfaces, exports, and styling; calibration file/result ingestion is local-only. |
| `nomo-engine/public/` | 2 | Favicon/logo assets used by the root app. |
| `nomo-ai` tracked tree | 25 | Landing routes, research pages, styling, package/deployment files, and assets reviewed; the only landing source change is the documentation/audit link repair. |

## Checks

- `nomo-planner`: Python tests pass, including the new completed-experiment CSV to local-calibration test.
- Root Lab: production build and Python/TypeScript parity verifier pass.
- Neuromorphic dashboard: `npm run typecheck` and `npm run build` pass; a live built-in Event-camera/AKD1500 search
  completed and exposed 550 designs, seven trade-offs, layer inspection, and export controls.
- Landing: production build passes; documentation links now resolve to tracked engine documents.
- No tracked source file contains a newly introduced TODO/FIXME or a silent substitute for missing measurements.

## Remaining boundaries

The audit does not turn data-dependent claims into code claims. Binary parser implementation behind the hosted API,
the shared-graph neuromorphic placement/compiler adapter, full framework-option round trips, serving/co-design
calibration, measured product data, broad sourced cost history/freshness, and a complete every-control browser
accessibility matrix remain explicitly listed as preview/open in `ROADMAP_AUDIT.md`.
