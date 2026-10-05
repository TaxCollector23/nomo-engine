# Nomo Engine delivery report

## Release scope

This release completes the remaining safe, testable slices of the roadmap without presenting estimates as measurements.

The shared model graph now powers layer-aware Train and a bounded per-layer Serve search. The engine reports precision
gain separately from additional layer gain, charges pipeline bubble, inter-stage transfer, and CPU activation offload,
and preserves locks and bounded-search labels. The historical Neuromorphic dashboard remains linked from the mode shell,
but is not misrepresented as a new shared-graph compiler integration.

The run auditor parses Megatron commands, DeepSpeed JSON, vLLM commands, and selected log metrics into one canonical
record. It can emit a corrected same-format representation of parsed fields. Customer CSV calibration and
recommendation-sensitive experiment ranking run in the browser and are visibly labelled as local preview evidence.

The Product Studio adds reference estimators for chip bottlenecks, RL scheduling, reliability/checkpoint goodput,
serving fleet sizing, fine-tuning modes, and TCO. The Cost Tracker adds cited model-card GPU-hour and provider-token
price rows plus a local physical-cost calculation from user-entered GPU rate and measured throughput.

## Verification

- Python reference: 32 tests passed.
- TypeScript production build: passed.
- Python/TypeScript planner and product parity: 14,289 checks passed; worst relative difference 4.37e-16.
- Existing 22-row calibration artifact and layer golden cases remain green.
- Product estimator physics-sanity tests passed.

## Gate status

| Gate | Status | Evidence / limitation |
|---|---|---|
| Gate 0 layer-aware training | Partial | Fair baseline and parity shipped; current assumptions show 0% extra per-layer gain; viewport evidence pending |
| Gate 1 shared graph | Partial | Train and Serve are real; Neuromorphic graph-contract export is shipped, but placement/compiler integration remains |
| Gate 2 auditor/calibration | Partial | Parsers, logs, same-format exports, local CSV fit, and experiment ranking shipped; held-out refit remains |
| Gate 3 uncertainty | Partial foundation | Calibrated A100 bootstrap shipped; serving/co-design and safest-plan study remain |
| Gate 4 product packs | Partial Preview | Six reference estimators, Python/browser goldens, and JSON preview export shipped; richer forms and customer data remain |
| Gate 5 cost tracker | Partial Preview | Five cited rows plus physical-cost calculator; broader coverage and freshness checks remain |

Detailed evidence is in docs/GATE_0_REPORT.md through docs/GATE_5_REPORT.md.

## Repositories and deployment

- Engine repository: https://github.com/TaxCollector23/nomo-engine
- Landing repository: https://github.com/TaxCollector23/nomo-ai
- Mode shell: https://frontend-gray-ten-c3tj1luab.vercel.app/
- Historical Neuromorphic dashboard: https://nomo-engine-dashboard.vercel.app/
- Landing page: https://nomoailanding.vercel.app/

The engine repository is the only repository changed in this release. The landing repository has no required frontend
change; its pre-existing untracked research/pilot directory was preserved.

Deployment IDs and smoke-test URLs are recorded in the release commit after the final production deployment.
