# Nomo Engine delivery report

## Release scope

### Serious simulation-core expansion (2026-10-05)

The new implementation adds a Python-first simulation reference and a deterministic TypeScript/Worker contract.
It covers operator graphs (forward/backward, attention, MoE, optimizer, FLOPs/bytes/lifetimes), roofline and
topology/collective models, training timelines/memory, serving request timelines/KV/cache/SLO metrics, calibrated
distributions and held-out reporting, plus artifact-driven M4–M11 product simulations. The Lab now exposes a
Simulation Workbench and Evidence registry, and the platform layer persists projects, artifacts, runs, comparisons,
HTML/PDF reports, and preview simulations through a CLI, HTTP API, SDK, and MCP-style JSON-RPC surface.

The serious-criteria boundary is explicit. The checked-in Study 1 training artifact retains 5.9% held-out PTD-P,
0.94 Spearman, and 18/22 (81.8%) nominal-90% interval coverage. The serving simulator is fully implemented and tested
against 12 measured Sarathi-Serve Table 4 rows. The derived replay matches all 12 rows with 72.98% MAPE; because the
paper's raw request trace was not supplied, this is traceable comparison evidence rather than a claim of raw-trace
reproduction. Customer validation remains Preview and is not presented as measured.

This release closes the remaining safe, testable slices identified by the strict roadmap audit without presenting
estimates as measurements. It does not claim that data-dependent or cross-repository compiler work is complete.

The shared model graph now powers layer-aware Train and a bounded per-layer Serve search. The engine reports precision
gain separately from additional layer gain, charges pipeline bubble, inter-stage transfer, and CPU activation offload,
and preserves locks and bounded-search labels. The historical Neuromorphic dashboard remains linked from the mode shell,
but is not misrepresented as a new shared-graph compiler integration. The dashboard launcher now removes an
unimplemented phase-coding control and states the estimate/provenance boundary before a search begins.

The run auditor parses Megatron commands, DeepSpeed JSON, vLLM commands, and selected log metrics into one canonical
record, flags unrecognized options, round-trips supported core fields, and connects known published model names to a
bounded shared-graph recommendation. Customer CSV calibration now reports leave-one-out error; the fit and
recommendation-sensitive experiment ranking run in the browser and remain visibly labelled as local preview evidence.

The Product Studio adds editable reference estimators and JSON/CSV exports for chip bottlenecks, RL scheduling,
reliability/checkpoint goodput, serving fleet sizing, fine-tuning modes, and TCO. The Cost Tracker adds cited
model-card GPU-hour and provider-token price rows plus a local physical-cost calculation and provider/physical
comparison from user-entered GPU rate and measured throughput. Nomo graph contracts can be re-imported, and calibrated
training exposes a conservative interval-regret Safest plan option.
Recommendation-sensitive experiment templates can now be downloaded and completed-result CSVs can be loaded back into
the local calibration fit with candidate-name matching and explicit errors for unknown experiments.
The landing page now links to real engine specification, deployment, observability, and roadmap-audit documents.

## Verification

- Python reference: 116 tests passed.
- TypeScript production build: passed.
- Neuromorphic Next.js dashboard typecheck and production build: passed.
- Landing page production build: passed.
- Python/TypeScript planner, product, layer, and auditor parity: 14,332 checks passed; worst relative difference 4.37e-16.
- Serious simulation golden: 123 checks passed, including zero-bubble, parallel-collective, embedding-dependency, and
  repaired NSGA-II coverage.
- Public serving validation: Sarathi-Serve Table 4 fixture contains 12 measured rows; the reproducible replay matched
  12/12 and reported 72.98% MAPE without filling missing data.
- Artifact boundaries: Hugging Face/Nomo JSON structural validation and bounded transformer-skeleton lowering, safetensors
  metadata, Prometheus text, and safe ONNX/state-dict preview inspection are tested; full framework-specific ONNX
  lowering and unsafe pickle loading remain explicit boundaries.
- Auditor boundaries: opaque Megatron/vLLM tokens and nested DeepSpeed fields round-trip in same-format exports; their
  semantics remain explicitly unvalidated without versioned framework fixtures.
- Platform parity: safe model inspection is available through the shared service, `POST /artifacts/inspect-model`, the
  Python SDK, CLI, and MCP tool.
- Production Lab build emitted and exercised a dedicated Simulation Worker bundle with synchronous fallback.
- Existing 22-row calibration artifact and layer golden cases remain green.
- Product estimator physics-sanity tests passed.
- Complete Lab route matrix: all 11 modules rendered at 1440×900 and 390×844 with no document-level horizontal
  overflow and no browser error or warning logs.

## Gate status

| Gate | Status | Evidence / limitation |
|---|---|---|
| Gate 0 layer-aware training | Partial | Fair baseline and parity shipped; current assumptions show 0% extra per-layer gain; 1440px/390px live viewport checks pass |
| Gate 1 shared graph | Partial | Train and Serve are real; Neuromorphic graph-contract export is shipped, but placement/compiler integration remains |
| Gate 2 auditor/calibration | Partial | Parsers, unknown-option warnings, core round-trip fixtures, same-format exports, known-preset recommendation bridge, local CSV fit, experiment ranking, downloadable templates, and local completed-result ingestion shipped; full framework-option and six-parameter refit remain |
| Gate 3 uncertainty | Partial foundation | Calibrated A100 bootstrap, probability-best, held-out coverage, and interval-regret Safest plan shipped; serving/co-design validation remains |
| Gate 4 product packs | Partial Preview | Six reference estimators, editable forms, Python/browser goldens, JSON/CSV preview exports, and Methods equations shipped; joint chip optimization and customer calibration remain |
| Gate 5 cost tracker | Partial Preview | Five cited rows, physical-cost calculation, provider/physical comparison, and source/as-of labels shipped; broader coverage and freshness checks remain |

Detailed evidence is in docs/GATE_0_REPORT.md through docs/GATE_5_REPORT.md.

## Repositories and deployment

- Engine repository: https://github.com/TaxCollector23/nomo-engine
- Landing repository: https://github.com/TaxCollector23/nomo-ai
- Mode shell: https://frontend-gray-ten-c3tj1luab.vercel.app/
- Historical Neuromorphic dashboard: https://nomo-engine-dashboard.vercel.app/
- Landing page: https://nomoailanding.vercel.app/

The earlier landing repair is already present in the landing repository. This serious simulation release changes the
engine repository only; the landing repository's pre-existing untracked `research/pilot` directory was preserved and
was not committed or deployed as source.

Historical baseline details retained for traceability: engine commit 29bbd00 and landing commit 98a91d7 were pushed
to their respective default branches. The baseline mode-shell production deployment was dpl_V495J7gBMhyQQ83HTXP585hPz6AM at
https://nomo-engine-hdw7htt2j-rangan-alt.vercel.app/. The stable mode-shell alias
https://frontend-gray-ten-c3tj1luab.vercel.app/ returned HTTP 200 and served the new
index-Cy47kWqD.js bundle with index-BYBpg4SO.css. The unique URL also returned HTTP 200. Browser module smoke tests
were run after cache-busting the stable alias. The live Product Studio was also checked at
1440×900 and 390×844; both viewports contained the document without horizontal overflow, and the
shared graph tablist fit at 390px.

The neuromorphic dashboard deployment dpl_MN1dV6Cuyd2KhadvmxdNrdwU9TYq and landing deployment
dpl_BBo1GiaehNAm71M2A7aStC9e1VfJ both completed successfully and were aliased to their stable URLs.

The line-by-line requirement map is [docs/ROADMAP_AUDIT.md](docs/ROADMAP_AUDIT.md); the file-level review is in
[docs/FILE_AUDIT.md](docs/FILE_AUDIT.md). Source bundles are available at
C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-29bbd00/ (historical engine, planner, and landing archives),
generated from the baseline audited tree after its pushed source commit.

## Latest release and live verification

Engine runtime commits `c777a1f`, `cab46e7`, `b08e5fb`, `631e040`, `126326c`, `e300b1c`, and `19dd854` were pushed to `main` after the `a314204` runtime release; the audit/deployment records were pushed alongside them. Vercel
production deployment `dpl_88jwbgJb6otFxUwcNe5zUpbT2fDe` is live at
https://nomo-engine-klb7z3drv-rangan-alt.vercel.app/ and the canonical mode-shell alias is
https://frontend-gray-ten-c3tj1luab.vercel.app/. The canonical alias was explicitly reassigned after deployment and
serves `index-c-I02bbG.js`, `index-DR3mSem2.css`, `worker-C-FLG9f1.js`, and `jszip.min-DuFMEDAl.js`.

Browser smoke verified all 11 Lab routes at 1440×900 and 390×844, the Simulation core module, Training and Serving
modes, the cited Sarathi Evidence row and explicit Preview evidence labels, and the Neuromorphic link to https://nomo-engine-dashboard.vercel.app/. A live
Event-camera/AKD1500 dashboard search completed at `/runs/620b864c08bd`, reporting 550 designs tried and seven
trade-offs with layer inspection available. The checked tabs had no console errors or warnings.
The landing repository stayed at pushed commit `98a91d7` with its existing deployment; no landing source change was
needed for this engine-only simulation release.

The current release archives are at
C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-a314204/.
