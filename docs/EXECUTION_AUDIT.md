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
shows measured runtime. This decision is provisional until the first benchmark records both paths.

## Layer 1 — simulation core and Gate A

| ID | Requirement | Status | Evidence / next action |
|---|---|---|---|
| C1 | Full forward/backward operator graph | IMPLEMENTED / PREVIEW | Python `simcore.py` and TS GraphIR account for embeddings, GEMMs, attention, norms, activations, loss, optimizer, MoE/router, FLOPs, bytes, and lifetimes. Core tests, browser goldens, and the Lab workbench verify it; default values remain assumptions. |
| C2 | Shape/precision-aware roofline kernel model | IMPLEMENTED / PREVIEW | Roofline functions support precision/shape efficiency curves and provenance; `simcalibration.py` imports measured operator/topology CSV. No customer microbenchmark artifact is checked in, so absolute hardware numbers stay Preview. |
| C3 | Hardware topology and collective model | IMPLEMENTED / PREVIEW | Topology IR includes devices, hosts, switches, NIC/fabric links, bandwidth/latency, path routing, and ring/tree/hierarchical planning. Default browser links are placeholders until measured topology input is supplied. |
| C4 | Training discrete-event simulator | IMPLEMENTED / PREVIEW | Compute/communication streams, dependencies, pipeline transfers, GPipe/1F1B/interleaved/zero-bubble policy labels, data/tensor/sequence/context/expert parallel, ZeRO/FSDP flags, recompute/offload, and memory timelines are implemented and tested. Zero-bubble ordering is explicitly analytical, not a runtime scheduler. |
| C5 | Request-level serving simulator | IMPLEMENTED / PREVIEW | `serving_sim.py` supports replay/generated arrivals, prompt/answer distributions, batching, paged KV, eviction/preemption, prefix caching, chunked prefill, speculation, PD disaggregation, TP/replicas, and TTFT/ITL/goodput/KV distributions. No public serving measurement fixture is imported yet. |
| C6 | Parameter distributions and interval reporting | IMPLEMENTED / PREVIEW | Product distributions and deterministic intervals are implemented; `simcalibration.py` fits measured groups, bootstrap intervals, held-out error, rank, and coverage. Checked-in Study 1 interval coverage is 18/22 (81.8%) against a nominal 90% target. |
| C7 | Calibration pipeline | PARTIAL | The 22-row Narayanan training artifact, provenance checks, strategy-specific fits, bootstrap intervals, and held-out reports are wired. Customer telemetry and cited serving rows are intentionally not fabricated or promoted. |
| Gate A | Study 1 parity, serving validation, rendered timelines, performance budget | PARTIAL | Study 1 evidence records 5.9% held-out PTD-P and 0.94 Spearman; browser/Python timelines and runtime are checked. Serving validation remains open until a cited numeric serving trace is imported. |

## Layer 2 — modules

| ID | Requirement | Status | Core dependency | Verification required |
|---|---|---|---|---|
| M1 | Train on core; Gantt/memory/intervals; Megatron/DeepSpeed/torchtitan exports | IMPLEMENTED / PREVIEW | Gate A | Core simulator, schedule tests, golden parity, and `simulation_exports.py`; calibrated evidence is training-only and browser defaults are assumptions. |
| M2 | Serve on core; latency/goodput/KV timeline; vLLM/SGLang/TensorRT-LLM exports | IMPLEMENTED / PREVIEW | Gate A | Serving simulator, trace replay, exports, and UI metrics are present; numeric public held-out validation is still required. |
| M3 | Audit real configs/logs; current-vs-best; same-format diff; PDF/HTML report | IMPLEMENTED / PREVIEW | M1/M2 | Existing parsers plus `audit_reports.py` provide lossless diff, same-format exports, escaped HTML, and dependency-free PDF. Customer-held-out logs remain unsupplied. |
| M4 | Architecture search with multiple scaling laws and joint train/serve costs | IMPLEMENTED / PREVIEW | M1/M2 | `product_simulations.py` carries scaling-law artifacts, train/serve timelines, intervals, constrained Pareto search, deterministic repair, and a reusable NSGA-II search path; validation is Preview without measured architecture rows. |
| M5 | Joint chip/software design-space search and report | IMPLEMENTED / PREVIEW | M1/M2 | Chip/software candidates, Pareto/bottleneck outputs, sensitivities, provenance, and exports exist; PPA is not claimed measured. |
| M6 | RL rollout/reward/train/sync simulation and scheduling search | IMPLEMENTED / PREVIEW | M1/M2 | Phase timelines, asynchronous staleness, interval outputs, and schedule search are tested; measured RL loop validation is open. |
| M7 | Monte Carlo reliability/goodput and Young/Daly validation | IMPLEMENTED / PREVIEW | M1 | Failure/checkpoint Monte Carlo, hot-spare fields, Young/Daly checks, and exports are tested; fleet failure traces are open. |
| M8 | Multi-day serving-fleet simulation and autoscaling policy search | IMPLEMENTED / PREVIEW | M2 | Multi-day traffic/cold-start/autoscaling replay and policy comparisons are tested; customer fleet replay validation is open. |
| M9 | Full/LoRA/QLoRA core simulation and framework exports | IMPLEMENTED / PREVIEW | M1/M2 | Full/adapter/quantized memory-throughput rows and framework config exports exist; measured quality-boundary data is open. |
| M10 | Multi-year procurement/TCO Monte Carlo | IMPLEMENTED / PREVIEW | M1/M2/M8 | Cash-flow, demand/price distributions, buy/lease rows, and intervals are tested; procurement inputs are user-supplied assumptions. |
| M11 | Open-model cost tracker derived from M1/M2 | IMPLEMENTED / PREVIEW | M1/M2 | Open-model scope is enforced, serving/training artifacts are required, and source/provenance labels are exported; coverage/freshness is open. |

## Layer 3 — platform

| ID | Requirement | Status | Evidence / next action |
|---|---|---|---|
| P1 | Projects, saved artifacts, history, side-by-side timeline diffs | IMPLEMENTED | SQLite project/artifact/run store, run comparison, persisted reports, and evidence panel are implemented and tested. |
| P2 | Real model/topology/CSV/log/Prometheus/trace inputs | PARTIAL | JSON contracts, CSV evidence ingestion, logs, and serving trace replay are implemented with provenance; Prometheus/ONNX/state-dict connectors remain open. |
| P3 | Shareable PDF/HTML reports with methods/evidence | IMPLEMENTED / PREVIEW | Escaped HTML and valid dependency-free PDF audit/run reports are exported; report content remains explicit about assumptions/evidence. |
| P4 | API, CLI, Python SDK, MCP with UI-equivalent capabilities | IMPLEMENTED | Local/HTTP API, CLI, SDK, MCP-style JSON-RPC, run comparison, and preview simulations share the platform service. |
| P5 | Evidence dashboard with held-out error/coverage/sources | IMPLEMENTED / PREVIEW | Lab Simulation Workbench and Evidence module show source, held-out error/rank/coverage, and server-only boundaries; serving/customer rows remain visibly Preview. |

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
| 2026-10-05 | Baseline audit | STARTED | Existing v5 surface is mostly closed-form; this prompt requires a new simulation layer. |
| 2026-10-05 | Python simulator suite | PASS | 94 tests passed from `nomo-planner`; product, serving, calibration, core, platform, export, regression, and repaired-search coverage are green. |
| 2026-10-05 | Browser simulation suite | PASS | `npm run build`; `npm run verify`; `npm run simulation:golden`; GraphIR and Python-core fixtures pass. |
| 2026-10-05 | Browser simulation suite | PASS | Production build emits the dedicated Worker bundle; the Lab workbench dispatches its local run through `src/simulation/worker.ts` with a synchronous fallback. |
| 2026-10-05 | Golden and Python search suite | PASS | TypeScript golden: 123 checks; Python suite: 94 tests; deterministic repaired NSGA-II search has a focused test. |
| 2026-10-05 | Representative runtime | PASS | Python train 2.12 ms / 172 events, serving 13.16 ms / 32 requests, product 1.24 ms / 32 candidates; TS golden 173.90 ms including bundling. |
| 2026-10-05 | Production deployment and browser smoke | PASS | Engine commit `aa266fa` is pushed; Vercel deployment `dpl_9DwMQDg45MiAvNktnV7QHMyaMBFf` is live. Stable mode-shell HTTP 200 serves the Simulation Worker bundle; Training, Serving, Preview labels, and Neuromorphic link were checked with no console errors/warnings. |
| 2026-10-05 | Release archives and report | PASS | Engine, planner, and landing tracked-file archives plus this final report are recorded under `release-bundles-aa266fa`. |

