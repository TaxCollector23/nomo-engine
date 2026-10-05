# Nomo roadmap line-by-line audit

Audit source: `C:/Users/Rangan Balaji/.codex/attachments/d5a12488-92cd-429d-97f9-9fd6df383380/Pasted text.txt` (the 165-line roadmap supplied on 2026-10-04). The earlier `b14457f8` copy is identical in its roadmap section. This audit is intentionally stricter than a release summary: **built** means there is code and verification evidence; **preview** means a useful, bounded slice is shipped; **open** means the requirement is not claimed.

## Rules

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 26–29 | Python reference first, TypeScript port, goldens, fixed seeds | Built | `nomo-planner/`, `src/planner/`, `npm run verify`; 14,332 checks after the auditor and contract fixtures. |
| 30–31 | No invented data; cite rows; omit unknowns | Built | `nomo-planner/data/`, model-card URLs, cost rows, parser warnings, and explicit user-input labels. |
| 32–33 | Honest provenance labels | Built | Guided/Explore/Rigor labels; customer input, assumption, calibrated, spec, and placeholder boundaries are visible. |
| 34–35 | Held-out validation for predictions/calibration | Preview | A100 study: 18/22 (81.8%) held out; customer CSV now reports leave-one-out error. Serving/co-design still have no published held-out data. |
| 36 | Formula citations in code, Methods, exports | Built | Planner docstrings, `src/lab/Methods.tsx`, export `references.bib`; product references include Young, Daly, and Erlang. |
| 37–39 | Three modes, accessibility, reduced motion, adaptive formatting, export tree | Built | Lab mode shell, labelled controls/dialog, `prefers-reduced-motion`, adaptive formatters, and five-folder planner export. Product packs also export JSON/CSV. |
| 40–41 | Study 1 regression numbers | Built | `npm run verify`; 5.9%, Spearman 0.94, and 6.4% LOO remain in the evidence/docs. |
| 42–43 | Progress kept current | Built | Root `PROGRESS.md`, gate reports, and this audit. |
| 44–45 | Integrity fallback is clearly marked | Built | Preview/assumption labels and open-item list are not hidden. |

## Gate 0 — per-layer training

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 61–63 | Fair global baseline; precision gain separated from layer gain | Built | `layers.py` / `layers.ts`; Llama 3 8B and 70B goldens. |
| 64–65 | Bubble, communication, and CPU transfer charged with labelled bandwidth | Built | Layer metrics and customer-overridable hardware assumptions. |
| 67–71 | Safety tests and fixed-seed reproducibility | Built | `tests/test_layers.py`, `scripts/layer-golden.json`, browser parity. |
| 72–73 | Published Llama 3 8B/70B and Mixtral presets; 8B default | Built | `model_configs.py`, `layers.ts`, linked Hugging Face sources. |
| 74 | Per-step and whole-run adaptive totals | Built | Layer UI formatters and whole-run metrics. |
| 75 | Remove old “90% interval unavailable” answer-card label | Built | No such UI label remains; uncalibrated boundaries are stated without that placeholder. |
| 77–80 | Gate checklist | Preview | Tests, build, verify, regression, and responsive browser checks pass. The requested positive additional per-layer gain is not demonstrated by the current assumptions (0% for the two large presets), so it is reported rather than manufactured. |

## Phase 1 — shared model graph

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 85–87 | One upload and graph component for all tabs | Preview | Hugging Face config JSON and Nomo graph-contract JSON now re-import into the shared graph; node accounting, map, locks, and what-if surfaces are shared. ONNX and binary state-dict parsing are not bundled in the browser and remain open. |
| 88–90 | Train per-layer decisions, repair, locks, verified browser port | Preview | Per-layer precision/recompute/offload/stages and absolute locks are implemented and tested. The current bounded search is deterministic exhaustive/capped enumeration, not a full NSGA-II implementation. |
| 91–92 | Serve per-layer weight/KV precision with cost recommendation | Built | `serving.py` / `serving.ts`, bounded serving search, quality effects labelled assumptions. |
| 93 | Neuromorphic shared graph through existing engine with unchanged exports | Preview | Contract JSON export and historical dashboard link are live. The cross-repository compiler/server adapter and placement recommendation are not claimed. |
| 94–95 | Gate 1 | Partial | Train/Serve are real; Neuromorphic is contract-only until the existing server contract and supported model are provided. |

## Phase 2 — auditor and calibration

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 100–101 | Parse Megatron, DeepSpeed JSON, vLLM, logs; reject/flag unknowns | Preview | Core topology/precision fields and selected step/throughput/memory metrics parse in both engines. Unknown CLI flags and JSON fields are now surfaced; exhaustive framework-option coverage is open. |
| 102–103 | Current vs recommended, exact changes, same-format correction, readable/exportable audit | Preview | Same-format loss-aware export and CSV experiment template are shipped. Known published models connect to the shared bounded recommendation; arbitrary model/hardware current-vs-dollar diff still needs a complete framework mapping. |
| 104–105 | Browser-local customer refit and calibration badge | Preview | Local multiplicative refit, 90% range, in-sample coverage, and leave-one-out error are displayed; no data leaves the browser. It is not the six-parameter published hardware refit. |
| 106–107 | Round-trip fixtures and synthetic calibration recovery | Built for the supported core fields | `scripts/auditor-golden.json`, Python tests, and `npm run verify` cover Megatron, DeepSpeed, and vLLM core fields. Full framework-option preservation remains open by design. |

## Phase 3 — uncertainty

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 112 | Hardware distribution | Built for calibrated A100 training | Deterministic stratified bootstrap artifact in `src/planner/uncertainty.json`. |
| 113–114 | Median + 90% interval and held-out coverage | Preview | Training predictions show intervals; coverage is reported as 18/22 (81.8%) against 90%. Serving/co-design lack published calibration rows. |
| 115–116 | Probability-best and safest plan | Preview | Probability-best is posterior-draw based for calibrated training; Rigor mode now offers a conservative interval-regret “Safest plan” selection. It is not a guarantee and is not available for uncalibrated packs. |
| 117–118 | Recommendation-sensitive experiment designer, exact configs, CSV, upload into calibration | Preview | Ranking and CSV template are shipped; a customer can copy results into the local calibration CSV. Automatic one-click experiment-result ingestion remains open. |
| 119–120 | Gate 3 | Partial | Coverage study and training interval evidence pass; “intervals everywhere” cannot be honestly closed without serving/co-design observations. |

## Phase 4 — six paid products

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 125–130 | Chip design, workload suite, best mapping, design-space export | Preview | Chip bottleneck/Pareto estimator, editable chip/workload inputs, and JSON/CSV export are live. Joint software mapping, area/power calibration, and workload-suite optimization are not yet a measured product. |
| 131–134 | RL post-training scheduling with answer-length input and outputs | Built as reference estimator | Editable p50/p95, GPU split, sync/async, colocation, batch, KV precision, samples/hour, cost, and schedule outputs. Rates remain customer inputs. |
| 135–136 | Reliability/goodput and Young formula | Built as reference estimator | Customer failure/checkpoint/restart inputs, tested interval search, Young equation, and citation. |
| 137–138 | Fleet sizing, hourly profile, p99 approximation, calibration boundary | Built as reference estimator | Editable hourly JSON profile, queueing approximation, p99 target, and visible calibration limitation. |
| 139–140 | Full/LoRA/QLoRA, memory/time/cost/GPU, quality assumptions | Built as reference estimator | Editable modes and quality-loss inputs; no quality claim is made without customer evaluations. |
| 141–142 | Buy/rent TCO, utilization, depreciation, uncertainty, cited/user prices | Preview | Editable buy/rent, utilization, energy, maintenance/depreciation proxy, and price range. Customer procurement data and richer uncertainty remain open. |
| 143–145 | Per-pack tests, goldens, Methods, equations, exports, calibration labels | Preview | Python physics tests, browser goldens, Methods references, and JSON/CSV exports pass. No pack has customer-held-out calibration; each says so. |

## Phase 5 — visibility

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 150–152 | Public cited training rows, provider prices, physical serving comparison | Preview | Five hand-entered cited rows and a local physical-cost calculator are live. Rows are not scraped; broader model history, freshness checks, and explicit row-by-row provider/physical ratio UI remain open. |
| 153 | Gate 5 | Partial | Source links/as-of notes and browser page pass; broader coverage/freshness are not claimed. |

## Final delivery

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 158–160 | Newcomer navigation grouped by model life, infrastructure, chip, trust | Built | Lab rail exposes graph/train/serve/design, neuromorphic, auditor/products/costs, evidence, and Methods. |
| 161 | Every module at 1440px and 390px with no console errors | Preview | Desktop/mobile smoke covered the mode shell, Lab, Product Studio, Auditor, Cost Tracker, and graph surfaces; a full scripted every-control accessibility matrix is still open. |
| 162 | README, Methods, Evidence, docs, PROGRESS | Built | Updated repository documentation and gate reports. |
| 163–164 | Feature-by-feature final report, verification, assumptions, limitations | Built | `FINAL_REPORT.md` plus this line audit. |
| 165 | Three updated codebase zips | Built | `release-bundles-*/nomo-engine-*.zip`, `nomo-planner-*.zip`, and `nomo-ai-*.zip`; landing archive contains no changes beyond its preserved pre-existing files. |

## Verdict

The release is materially more complete than the previous audit, but it is not honest to call the roadmap 100% complete. The remaining blockers are not hidden implementation oversights: binary ONNX/state-dict ingestion, a true neuromorphic server/compiler adapter, full framework-option round trips, customer/published serving and co-design calibration, measured chip/quality/queueing data, broader sourced cost history/freshness, and a complete every-control browser/accessibility matrix. Closing those requires the corresponding formats, server contract, hardware measurements, customer logs/evaluations, and source disclosures.
