# Nomo roadmap line-by-line audit

Audit source: `C:/Users/Rangan Balaji/.codex/attachments/d5a12488-92cd-429d-97f9-9fd6df383380/Pasted text.txt` (the 165-line roadmap supplied on 2026-10-04). The earlier `b14457f8` copy is identical in its roadmap section. This audit is intentionally stricter than a release summary: **built** means there is code and verification evidence; **preview** means a useful, bounded slice is shipped; **open** means the requirement is not claimed.

## Rules

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 26–29 | Python reference first, TypeScript port, goldens, fixed seeds | Built | `nomo-planner/`, `src/planner/`, `npm run verify`; 14,337 checks after the auditor, evidence, and contract fixtures. |
| 30–31 | No invented data; cite rows; omit unknowns | Built | `nomo-planner/data/`, model-card URLs, cost rows, parser warnings, and explicit user-input labels. |
| 32–33 | Honest provenance labels | Built | Guided/Explore/Rigor labels; customer input, assumption, calibrated, spec, and placeholder boundaries are visible. |
| 34–35 | Held-out validation for predictions/calibration | Preview | A100 study: 18/22 (81.8%) held out; customer CSV reports leave-one-out error; the cited Sarathi-Serve fixture has 12 measured rows and a 12/12 derived replay with 72.98% MAPE. Serving raw-trace fidelity and co-design still lack held-out data. |
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
| 85–87 | One upload and graph component for all tabs | Preview | Hugging Face config JSON and Nomo graph-contract JSON receive structural validation and bounded lowering; when the optional reader is installed, ONNX receives checker-backed generic structural lowering for graph boundaries, operators, attributes, and explicit tensor metadata. Browser binary lowering, framework-specific semantics, and cost-bearing conversion remain open; node accounting, map, locks, and what-if surfaces are shared for JSON inputs. |
| 88–90 | Train per-layer decisions, repair, locks, verified browser port | Preview | Per-layer precision/recompute/offload/stages and absolute locks are implemented and tested. Up to four nodes use exact exhaustive enumeration when uncapped; larger searches use deterministic repair-aware constrained NSGA-II bounded by `max_candidates`, so the returned Pareto set is not globally exhaustive. |
| 91–92 | Serve per-layer weight/KV precision with cost recommendation | Built | `serving.py` / `serving.ts`, bounded serving search, quality effects labelled assumptions. |
| 93 | Neuromorphic shared graph through existing engine with unchanged exports | Preview | Contract JSON export and historical dashboard link are live. The cross-repository compiler/server adapter and placement recommendation are not claimed. |
| 94–95 | Gate 1 | Partial | Train/Serve are real; Neuromorphic is contract-only until the existing server contract and supported model are provided. |

## Phase 2 — auditor and calibration

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 100–101 | Parse Megatron, DeepSpeed JSON, vLLM, logs; reject/flag unknowns | Preview | Python now parses Megatron, DeepSpeed JSON, TorchTitan v0.2.2 TOML, vLLM, and selected step/throughput/memory metrics. Versioned bounded fixtures validate documented common fields and cross-field conflicts; unknown CLI flags, opaque values, and nested fields are surfaced and preserved. Exhaustive/version-selected coverage, browser parity, and customer-log calibration remain open. See `docs/FRAMEWORK_AUDITOR.md`. |
| 102–103 | Current vs recommended, exact changes, same-format correction, readable/exportable audit | Preview | Same-format loss-aware export and CSV experiment template are shipped. Known published models connect to the shared bounded recommendation; arbitrary model/hardware current-vs-dollar diff still needs a complete framework mapping. |
| 104–105 | Browser-local customer refit and calibration badge | Preview | Local multiplicative refit, labelled empirical 90% in-sample range, coverage, and leave-one-out error are displayed; CSV identity is validated and no data leaves the browser. It is not the six-parameter published hardware refit. |
| 106–107 | Round-trip fixtures and synthetic calibration recovery | Built for bounded fixture fields | `scripts/auditor-golden.json` and Python tests cover the existing browser-parity core fields; Python auditor tests add versioned Megatron, DeepSpeed, TorchTitan TOML, and vLLM validation/round trips. Opaque options and nested fields round-trip. Full framework coverage and browser consumption of the Python reports remain open by design. |

## Phase 3 — uncertainty

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 112 | Hardware distribution | Built for calibrated A100 training | Deterministic stratified bootstrap artifact in `src/planner/uncertainty.json`. |
| 113–114 | Median + 90% interval and held-out coverage | Preview | Training predictions show intervals; coverage is reported as 18/22 (81.8%) against 90%. Serving/co-design lack published calibration rows. |
| 115–116 | Probability-best and safest plan | Preview | Probability-best is posterior-draw based for calibrated training; Rigor mode now offers a conservative interval-regret “Safest plan” selection. It is not a guarantee and is not available for uncalibrated packs. |
| 117–118 | Recommendation-sensitive experiment designer, exact configs, CSV, upload into calibration | Preview | Ranking and a downloadable CSV template are shipped; completed experiment-result CSV rows can now be loaded locally, matched to the current candidate names, and fed into the local calibration fit without guessing. Server-side result ingestion and published calibration remain open. |
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
| 150–152 | Public cited training rows, provider prices, physical serving comparison | Preview | Five hand-entered cited rows, a local physical-cost calculator, row-by-row provider/physical ratios, metadata-only freshness labels, and explicit open-model coverage counts are live. Rows are not scraped; broader open-model serving/token-price coverage, model history, and network verification remain open. |
| 153 | Gate 5 | Partial | Source links/as-of notes and browser page pass; date metadata is validated without fetching URLs, but the two model-card rows remain undated and open-model serving/token-price coverage is absent. |

## Final delivery

| Lines | Requirement | Status | Evidence / boundary |
|---|---|---|---|
| 158–160 | Newcomer navigation grouped by model life, infrastructure, chip, trust | Built | Lab rail exposes graph/train/serve/design, neuromorphic, auditor/products/costs, evidence, and Methods. |
| 161 | Every module at 1440px and 390px with no console errors | Built | Direct route checks covered all 11 Lab modules at 1440×900 and 390×844; each named module rendered with no document-level horizontal overflow or browser error/warning logs. A deeper every-control interaction matrix remains an optional follow-up. |
| 162 | README, Methods, Evidence, docs, PROGRESS | Built | Updated repository documentation and gate reports. |
| 163–164 | Feature-by-feature final report, verification, assumptions, limitations | Built | `FINAL_REPORT.md` plus this line audit. |
| 165 | Three updated codebase zips | Built | Current tracked-file archives are recorded under `release-bundles-a314204/` for engine, planner, and landing; landing contains no source changes beyond its preserved pre-existing files. |

## Verdict

The release is materially more complete than the previous audit, but it is not honest to call the roadmap 100% complete. The remaining blockers are not hidden implementation oversights: framework-specific ONNX lowering and binary graph-to-cost conversion beyond the generic structural boundary, a true neuromorphic server/compiler adapter, full framework-option round trips, customer/raw-trace serving and co-design calibration, measured chip/quality/queueing data, broader open-model serving/token-price coverage and network source verification, and a complete every-control browser/accessibility matrix. Closing those requires the corresponding formats, server contract, hardware measurements, customer logs/evaluations, and source disclosures.
