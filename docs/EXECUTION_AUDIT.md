# Nomo AI serious-simulation execution audit

Status: active. This file is the single completion ledger for the prompt pasted on 2026-10-05.
It is updated after each implementation and verification pass. A requirement is only marked `DONE`
when code, tests, UI evidence, and deployment evidence exist. Unsupported claims stay `PREVIEW` or
`OPEN`; no synthetic measurement is promoted to validation evidence.

## Design decision

The simulation core is Python-first in `nomo-planner/nomo_planner/` so calibration, numerical tests,
and reference behavior remain inspectable. A deterministic TypeScript/Web Worker port is required for
browser-visible simulations and is verified against Python goldens. Large searches and uncertainty
sampling may run through the Python API/CLI; the browser uses a Worker for responsive local runs and
shows measured runtime. Representative Python workloads are recorded at 2.12 ms training, 13.16 ms serving, and
1.24 ms product; the TypeScript golden run is 173.90 ms including bundling, and the production Worker dispatch path
is smoke-tested. Large searches remain server-side because their input and sampling sizes are not a browser budget.

## Layer 1 — simulation core and Gate A

| ID | Requirement | Status | Evidence / next action |
|---|---|---|---|
| C1 | Full forward/backward operator graph | IMPLEMENTED / PREVIEW | Python `simcore.py` and TS GraphIR account for embeddings, GEMMs, attention, norms, activations, loss, optimizer, MoE/router, FLOPs, bytes, and lifetimes. Core tests, browser goldens, and the Lab workbench verify it; default values remain assumptions. |
| C2 | Shape/precision-aware roofline kernel model | IMPLEMENTED / PREVIEW | Roofline functions support precision/shape efficiency curves and provenance; `simcalibration.py` imports measured operator/topology CSV. No customer microbenchmark artifact is checked in, so absolute hardware numbers stay Preview. |
| C3 | Hardware topology and collective model | IMPLEMENTED / PREVIEW | Topology IR includes devices, hosts, switches, NIC/fabric links, bandwidth/latency, path routing, and ring/tree/hierarchical planning. Default browser links are placeholders until measured topology input is supplied. |
| C4 | Training discrete-event simulator | IMPLEMENTED / PREVIEW | Compute/communication streams, dependencies, pipeline transfers, GPipe/1F1B/interleaved/zero-bubble policy labels, data/tensor/sequence/context/expert parallel, ZeRO/FSDP flags, recompute/offload, and memory timelines are implemented and tested. Zero-bubble ordering is explicitly analytical, not a runtime scheduler. |
| C5 | Request-level serving simulator | IMPLEMENTED / PREVIEW | `serving_sim.py` supports replay/generated arrivals, prompt/answer distributions, batching, paged KV, eviction/preemption, prefix caching, chunked prefill, speculation, PD disaggregation, TP/replicas, and TTFT/ITL/goodput/KV distributions. The cited Sarathi-Serve Table 4 fixture imports 12 measured rows; the derived replay matches all rows but reports 72.98% MAPE because the raw trace is unavailable. |
| C6 | Parameter distributions and interval reporting | IMPLEMENTED / PREVIEW | Product distributions and deterministic intervals are implemented; `simcalibration.py` fits measured groups, bootstrap intervals, held-out error, rank, and coverage. Checked-in Study 1 interval coverage is 18/22 (81.8%) against a nominal 90% target. |
| C7 | Calibration pipeline | PARTIAL | The 22-row Narayanan training artifact and 12-row Sarathi-Serve serving fixture have provenance checks, metadata-only source reports, strategy-specific comparison, bootstrap/held-out reporting, and explicit interval/evidence labels. Customer telemetry, raw serving traces, and co-design calibration remain open. |
| Gate A | Study 1 parity, serving validation, rendered timelines, performance budget | PARTIAL | Study 1 evidence records 5.9% held-out PTD-P and 0.94 Spearman; 12 cited Sarathi-Serve measurements are imported and compared by a derived replay. Raw trace fidelity and customer validation remain Preview. |

## Layer 2 — modules

| ID | Requirement | Status | Core dependency | Verification required |
|---|---|---|---|---|
| M1 | Train on core; Gantt/memory/intervals; Megatron/DeepSpeed/torchtitan exports | IMPLEMENTED / PREVIEW | Gate A | Core simulator, schedule tests, golden parity, and `simulation_exports.py`; calibrated evidence is training-only and browser defaults are assumptions. |
| M2 | Serve on core; latency/goodput/KV timeline; vLLM/SGLang/TensorRT-LLM exports | IMPLEMENTED / PREVIEW | Gate A | Serving simulator, trace replay, exports, UI metrics, and 12-row Sarathi-Serve comparison are present; raw-trace fidelity and customer held-out validation remain Preview. |
| M3 | Audit real configs/logs; current-vs-best; same-format diff; PDF/HTML report | IMPLEMENTED / PREVIEW | M1/M2 | Existing parsers plus `audit_reports.py` provide lossless diff, same-format exports, escaped HTML, and dependency-free PDF. Python adds versioned bounded semantic fixtures for common Megatron, DeepSpeed, TorchTitan TOML, and vLLM fields; unknown options/fields remain preserved. Full framework-version coverage, browser parity, target-framework dry runs, and customer-held-out logs remain unsupplied. |
| M4 | Architecture search with multiple scaling laws and joint train/serve costs | IMPLEMENTED / PREVIEW | M1/M2 | `product_simulations.py` carries scaling-law artifacts, train/serve timelines, intervals, constrained Pareto search, deterministic repair, and a reusable NSGA-II search path; validation is Preview without measured architecture rows. |
| M5 | Joint chip/software design-space search and report | IMPLEMENTED / PREVIEW | M1/M2 | Chip/software candidates, Pareto/bottleneck outputs, sensitivities, provenance, and exports exist; PPA is not claimed measured. |
| M6 | RL rollout/reward/train/sync simulation and scheduling search | IMPLEMENTED / PREVIEW | M1/M2 | Phase timelines, asynchronous staleness, interval outputs, and schedule search are tested; measured RL loop validation is open. |
| M7 | Monte Carlo reliability/goodput and Young/Daly validation | IMPLEMENTED / PREVIEW | M1 | Failure/checkpoint Monte Carlo, hot-spare fields, Young/Daly checks, and exports are tested; fleet failure traces are open. |
| M8 | Multi-day serving-fleet simulation and autoscaling policy search | IMPLEMENTED / PREVIEW | M2 | Multi-day traffic/cold-start/autoscaling replay and policy comparisons are tested; customer fleet replay validation is open. |
| M9 | Full/LoRA/QLoRA core simulation and framework exports | IMPLEMENTED / PREVIEW | M1/M2 | Full/adapter/quantized memory-throughput rows and framework config exports exist; measured quality-boundary data is open. |
| M10 | Multi-year procurement/TCO Monte Carlo | IMPLEMENTED / PREVIEW | M1/M2/M8 | Cash-flow, demand/price distributions, buy/lease rows, and intervals are tested; procurement inputs are user-supplied assumptions. |
| M11 | Open-model cost tracker derived from M1/M2 | IMPLEMENTED / PREVIEW | M1/M2 | Open-model scope is enforced, serving/training artifacts are required, source/provenance labels and metadata-only freshness statuses are exported; checked-in coverage is two open-model training rows and zero open-model serving/token-price rows. |

## Layer 3 — platform

| ID | Requirement | Status | Evidence / next action |
|---|---|---|---|
| P1 | Projects, saved artifacts, history, side-by-side timeline diffs | IMPLEMENTED | SQLite project/artifact/run store, run comparison, persisted reports, and evidence panel are implemented and tested. |
| P2 | Real model/topology/CSV/log/Prometheus/trace inputs | IMPLEMENTED / PREVIEW | JSON/Hugging Face graph contracts have structural validation and bounded transformer-skeleton lowering; CSV evidence, logs, serving traces, Prometheus text, safetensors metadata, and checker-backed generic ONNX structural lowering are implemented with provenance. Framework-specific ONNX lowering, graph-to-cost conversion, and unsafe pickle loading remain intentionally unsupported. |
| P3 | Shareable PDF/HTML reports with methods/evidence | IMPLEMENTED / PREVIEW | Escaped HTML and valid dependency-free PDF audit/run reports are exported; report content remains explicit about assumptions/evidence. |
| P4 | API, CLI, Python SDK, MCP with UI-equivalent capabilities | IMPLEMENTED | Local/HTTP API, CLI, SDK, MCP-style JSON-RPC, safe model inspection, run comparison, and preview simulations share the platform service. Model inspection is exposed at `POST /artifacts/inspect-model` and through `PlatformClient.inspect_model`. |
| P5 | Evidence dashboard with held-out error/coverage/sources | IMPLEMENTED / PREVIEW | Lab Simulation Workbench and Evidence module show Study 1 source/error/rank/nominal-90%-interval coverage, Sarathi-Serve 12-row MAPE and match count, source-date status, and server-only boundaries; customer rows remain visibly Preview. |

## Cross-cutting gates

- [x] Prior-art review and citations: Calculon, ASTRA-sim, Vidur, Megatron-LM, Korthikanti et al.,
  Zero Bubble, FlashAttention, vLLM/PagedAttention, Sarathi, DistServe, Splitwise, speculative decoding,
  and data-constrained scaling.
- [x] Python reference first, deterministic TypeScript parity, golden verification.
- [x] No invented measurements; Preview labels for every module missing any serious criterion.
- [x] `PROGRESS.md` updated at every gate.
- [x] Every module has hand-checked cases, physics sanity tests, evidence-dashboard validation, tested exports,
  and 1440px/390px browser evidence.
- [x] Updated engine/planner/landing archives and final report.

## Verification log

| Date | Check | Result | Notes |
|---|---|---|---|
| 2026-10-08 | Enterprise workspace, dashboard hardening, and API boundary refresh | PASS / PREVIEW | Browser-local workspace review journal, explicit platform sync for runs/review metadata/notes, browser-readable bearer errors, bounded request bodies with HTTP 413 rejection and CLI configuration, threaded in-memory store coverage, Next 16.4.0 + React 19.3.0 dashboard, and SPA shell cache policy are implemented. Root build, 14,337 parity checks, 123 simulation goldens, 138 Python tests, dashboard typecheck/lint/build, and production audits passed. Full development audit still reports Tailwind 3 / Next ESLint-plugin advisories that require a major Tailwind migration. |
| 2026-10-08 | Current production deployments | PASS | Engine UI commit `f46905e` is deployed as mode shell `dpl_5BTQes4WsxCBopPzjUjFiRMHEVpu`; dashboard `dpl_8qgaHU8QGF3ro9pAsWiA8Vds5XSh` and landing commit `fc4325d` deployment `dpl_H3Jj5LqT7EVDz87eZCESJ1BecLAe` are READY. Stable mode-shell, dashboard, and landing URLs return HTTP 200, and the Lab bundle contains the new workspace markers, import safety limit, and secure URL guard. |
| 2026-10-05 | Baseline audit | STARTED | Existing v5 surface is mostly closed-form; this prompt requires a new simulation layer. |
| 2026-10-05 | Python simulator suite | PASS | 134 tests passed from `nomo-planner`; product, serving, calibration, artifact-ingestion, core, platform, export, regression, auditor round-trip, repaired-search, source-freshness, and cost-boundary coverage are green. |
| 2026-10-05 | Browser simulation suite | PASS | `npm run build`; `npm run verify`; `npm run simulation:golden`; GraphIR and Python-core fixtures pass. |
| 2026-10-05 | Browser simulation suite | PASS | Production build emits the dedicated Worker bundle; the Lab workbench dispatches its local run through `src/simulation/worker.ts` with a synchronous fallback. |
| 2026-10-05 | Golden and Python search suite | PASS | TypeScript golden: 123 checks; Python suite: 134 tests; deterministic repaired NSGA-II search and public serving comparison have focused tests. |
| 2026-10-05 | Representative runtime | PASS | Python train 2.12 ms / 172 events, serving 13.16 ms / 32 requests, product 1.24 ms / 32 candidates; TS golden 173.90 ms including bundling. |
| 2026-10-05 | Production deployment and browser smoke | PASS | Engine commit `aa266fa` is pushed; Vercel deployment `dpl_9DwMQDg45MiAvNktnV7QHMyaMBFf` is live. Stable mode-shell HTTP 200 serves the Simulation Worker bundle; Training, Serving, Preview labels, and Neuromorphic link were checked with no console errors/warnings. |
| 2026-10-05 | Release archives and report | PASS | Engine, planner, and landing tracked-file archives plus this final report are recorded under `release-bundles-aa266fa`. |
| 2026-10-05 | Public serving validation | PASS / PREVIEW | Sarathi-Serve Table 4: 12/12 rows matched by the reproducible Nomo replay; 72.98% MAPE is reported honestly. The input trace is derived from published length summaries because the raw trace is not included. |
| 2026-10-06 | Safe artifact and audit-boundary pass | PASS / PREVIEW | HF/Nomo structural lowering, checker-backed generic ONNX graph lowering, base64 binary transport, nested/opaque framework-option preservation, bounded versioned framework validation fixtures, metadata-only source freshness, explicit cost coverage, and HTTP/SDK model inspection are covered by the 134-test planner suite; exhaustive framework validation and framework-specific binary lowering remain Preview. |
| 2026-10-06 | Production alias refresh and live browser verification | PASS | Runtime commits `c777a1f`, `cab46e7`, `b08e5fb`, `631e040`, `126326c`, `e300b1c`, and `19dd854` are pushed, with the audit/deployment records alongside them; Vercel deployment `dpl_88jwbgJb6otFxUwcNe5zUpbT2fDe` and the canonical mode-shell alias both return HTTP 200 and serve the current Worker bundle. All 11 Lab routes were directly checked at 1440×900 and 390×844 with no document overflow or console errors/warnings; a live Event-camera/AKD1500 run completed with 550 designs and seven trade-offs. |
| 2026-10-06 | Release archives and report refresh | PASS | Current tracked-file engine, planner, and landing archives are recorded under `release-bundles-a314204`; the landing `research/pilot/` directory remains preserved and untracked. |
| 2026-10-06 | Evidence/cost/calibration boundary release | PASS / PREVIEW | Commit `27b8824` is pushed; Vercel deployment `dpl_5sxB78cDXJhiV1JaEi22aJ2wrKq4` and the stable mode-shell alias return HTTP 200. The deployed bundle contains the 14,337 count, metadata-only freshness labels, explicit 2-training/0-serving-token coverage boundary, and empirical in-sample interval disclaimer; open-model serving data and network verification remain blocked by missing cited data. |
| 2026-10-06 | Measured verification and benchmark surface | PASS | After the Evidence update, 134 Python tests, 14,337 parity checks (worst relative difference `4.37e-16`), and 123 simulation goldens passed. `npm run bench` measured 7,680 plans / 63 ms, 13,440 / 150 ms, 216 / 2 ms, and 1,215 / 13 ms; the Evidence module now displays these rows with local-snapshot limitations. |
| 2026-10-06 | Final benchmark deployment and browser verification | PASS | Commit `e34c8c1` is pushed. Vercel deployment `dpl_7ezLreq3DhKwv4jj9JfEK4G3YUcH` is READY at `https://nomo-engine-p6ysgomgo-rangan-alt.vercel.app/`; the canonical no-7 alias was explicitly rebound to it. Both return HTTP 200, the live Evidence panel shows the four measured rows, and the browser reports no error/warning logs. |
| 2026-10-06 | Final tracked-file archives | PASS | Archives regenerated from the final engine and landing trees at `release-bundles-a314204/nomo-engine-b1d3bdb.zip`, `nomo-planner-b1d3bdb.zip`, and `nomo-ai-b1d3bdb.zip`; the landing `research/pilot/` directory remains excluded and preserved. |
