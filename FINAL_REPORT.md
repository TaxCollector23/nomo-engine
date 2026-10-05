# Nomo Engine delivery report

## Release scope

This release closes the remaining safe, testable slices identified by the strict roadmap audit without presenting
estimates as measurements. It does not claim that data-dependent or cross-repository compiler work is complete.

The shared model graph now powers layer-aware Train and a bounded per-layer Serve search. The engine reports precision
gain separately from additional layer gain, charges pipeline bubble, inter-stage transfer, and CPU activation offload,
and preserves locks and bounded-search labels. The historical Neuromorphic dashboard remains linked from the mode shell,
but is not misrepresented as a new shared-graph compiler integration.

The run auditor parses Megatron commands, DeepSpeed JSON, vLLM commands, and selected log metrics into one canonical
record, flags unrecognized options, round-trips supported core fields, and connects known published model names to a
bounded shared-graph recommendation. Customer CSV calibration now reports leave-one-out error; the fit and
recommendation-sensitive experiment ranking run in the browser and remain visibly labelled as local preview evidence.

The Product Studio adds editable reference estimators and JSON/CSV exports for chip bottlenecks, RL scheduling,
reliability/checkpoint goodput, serving fleet sizing, fine-tuning modes, and TCO. The Cost Tracker adds cited
model-card GPU-hour and provider-token price rows plus a local physical-cost calculation and provider/physical
comparison from user-entered GPU rate and measured throughput. Nomo graph contracts can be re-imported, and calibrated
training exposes a conservative interval-regret Safest plan option.

## Verification

- Python reference: 36 tests passed.
- TypeScript production build: passed.
- Python/TypeScript planner, product, layer, and auditor parity: 14,332 checks passed; worst relative difference 4.37e-16.
- Existing 22-row calibration artifact and layer golden cases remain green.
- Product estimator physics-sanity tests passed.

## Gate status

| Gate | Status | Evidence / limitation |
|---|---|---|
| Gate 0 layer-aware training | Partial | Fair baseline and parity shipped; current assumptions show 0% extra per-layer gain; 1440px/390px live viewport checks pass |
| Gate 1 shared graph | Partial | Train and Serve are real; Neuromorphic graph-contract export is shipped, but placement/compiler integration remains |
| Gate 2 auditor/calibration | Partial | Parsers, unknown-option warnings, core round-trip fixtures, same-format exports, known-preset recommendation bridge, local CSV fit, and experiment ranking shipped; full framework-option and six-parameter refit remain |
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

The engine repository is the only repository changed in this release. The landing repository has no required frontend
change; its pre-existing untracked research/pilot directory was preserved.

Production deployment: dpl_FjUvGXnLiCr4U5GoTSDWf1zsCKoe at
https://nomo-engine-7mnxyawpw-rangan-alt.vercel.app/. The stable mode-shell alias
https://frontend-gray-ten-c3tj1luab.vercel.app/ returned HTTP 200 and served the new
index-Bvqy1GcW.js bundle. The unique URL also returned HTTP 200. Browser module smoke tests
were run after cache-busting the stable alias. The live Product Studio was also checked at
1440×900 and 390×844; both viewports contained the document without horizontal overflow.

The line-by-line requirement map is [docs/ROADMAP_AUDIT.md](docs/ROADMAP_AUDIT.md). Source bundles are available at
C:/Users/Rangan Balaji/Desktop/Nomo AI/release-bundles-29b8503/ (engine, planner, and landing archives); they will be
regenerated after this audit commit so the archives exactly match the pushed tree.
